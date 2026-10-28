# -*- coding: utf-8 -*-
"""S6 概览（turbo 可选）：分块要点 → 总览文件「概览.md」，供人快速查阅。

双路径（limix-2 实测 + 1M 上下文模型修订，2026-10-28）：
  DIRECT  正文估算 ≤ gate_tokens(128k)：单次调用直接读全文（不经过有损中间层）
          + MinerU 图注原文 + 公式清单 → 概览。零截断、信息量最大。
  MAPRED  超长论文（百页综述级）：按 target_tokens(40k) 文档序装箱切块，
          逐块要点抽取（并发 4，与 S4 重叠）→ reduce = 全部要点 + 首块原文
          + 图表/公式清单 → 概览。

护栏（实测教训并入）：
  - 提示词硬约束「仅凭所给材料」——模型对热门论文会夹带外部记忆
    （limix-2 实测幻觉 "d=256"，全文无此数）
  - reduce/direct 恒附首块原文：机构/链接/代际定位等前置元信息在要点压缩中会丢失
  - S6 失败隔离：单块失败跳过、终稿失败写占位文件；绝不影响主产物与退出码
  - 缓存 cache_dir/summary.json（key=提示词版本+模型+材料hash），改提示词
    只重算概览；重跑零计费；断点续跑同 translations 语义
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .clients.llm import LLMClient, log_event
from .clients.ratelimit import SlidingWindowLimiter
from .s5_assemble import DISCLAIMER

PROMPT_VER = "v1"

SYSTEM_PROMPT = (
    "你是学术文献综述撰写者。用户给出一篇论文的原文材料"
    "（按原文顺序的正文，可能附图表清单与公式清单）。"
    "请写一份供人快速查阅的中文总览，输出 Markdown，结构："
    "## 一句话概括 / ## 问题与动机 / ## 方法（模型·数据·训练，保留关键参数数值） / "
    "## 实验与结果（保留关键数字） / ## 创新点与局限。硬性规则："
    "1) 只允许使用给定材料中出现的信息；你对该主题的先验知识（包括参数规模、"
    "排行榜、发布渠道等细节）一律不得写入，材料没有就省略该项；"
    "2) 忠实于材料，关键数值保留原始数字，不得改写或推算；"
    "3) 材料间冲突时并列指出；4) 总长 800-1400 字，信息密集，不写客套句。"
)

MAP_PROMPT = (
    "你是学术文献精读助手。针对给出的论文片段（论文按原文顺序切块，你只看到其中"
    "一块），抽取该片段的关键信息，供后续生成全文总览使用。要求："
    "1) 保留：研究问题/动机、方法与模型结构（含材料中出现的具体名称与参数数值）、"
    "数据集与来源、实验设置、关键数值结果（保留原始数字）、结论与创新点、局限；"
    "2) 只允许使用片段中出现的信息，禁止补充任何片段之外的先验知识；"
    "3) 公式只记一句话作用，不抄 LaTeX；"
    "4) 输出不超过 700 字的中文要点列表，信息密集；片段首尾半句话属截断噪声，忽略。"
    "只输出要点。"
)


def _est_tokens(s: str) -> int:
    """保守估算：中英混合 ~2.5 字符/token（宁高勿低，配合限流预留）。"""
    return int(len(s) / 2.5)


def _render(text: str) -> str:
    """概览成文：标题 + 与译文一致的 AI 幻觉免责行（复用 s5.DISCLAIMER）。"""
    return f"# 论文概览\n\n{DISCLAIMER}\n\n{text}\n"


def pack_chunks(units, target_tokens: int):
    """正文单元按估算 token 文档序装箱；单元本身超目标时独立成块。"""
    body = [u for u in units if u.kind in ("text", "title", "ref", "equation")]
    chunks, cur, cur_tok = [], [], 0
    for u in body:
        t = _est_tokens(u.src)
        if cur and cur_tok + t > target_tokens:
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


def _materials(units, formulas, visuals) -> tuple[str, str]:
    """返回 (正文全文, 附加材料串)。附加=图表清单（图注原文）+公式清单。
    公式用 S2 注册表（id+页码），不抄 LaTeX——概览是给人读的。"""
    body = "\n\n".join(u.src for u in units
                       if u.kind in ("text", "title", "ref", "equation"))
    cap_lines = []
    for v in visuals:
        if not v.skipped:
            cap_lines.append(f"- {v.id}（第{v.page}页，{v.kind}）：{(v.caption or '（无图注）')[:160]}")
    eq_lines = [f"- {f.id}（第{f.page}页）" for f in formulas][:60]
    add = "【图表清单（原文图注）】\n" + ("\n".join(cap_lines) or "（无）")
    add += "\n\n【公式清单（数量与位置）】\n" + ("\n".join(eq_lines) or "（无）")
    return body, add


def run_summary(units: list, formulas: list, visuals: list,
                client: LLMClient, out_dir: Path, cache_dir: Path, log_path: Path,
                limiter: SlidingWindowLimiter | None = None,
                cfg: dict | None = None) -> dict:
    """生成 概览.md。恒不抛穿；返回 {"status": "ok|fallback|fail|cached",
    "calls": n, "tokens": "+", "wall_s": x, "path": str}。
    visuals 传 collect_visuals 的 Visual 对象（用 id/page/kind/caption，均为本地
    解析字段，与 S3 后台线程写入的结果字段不重叠，读取安全）。"""
    cfg = cfg or {}
    gate = int(cfg.get("gate_tokens", 128_000))
    target = int(cfg.get("target_tokens", 40_000))
    map_conc = int(cfg.get("map_concurrency", 4))
    out_name = cfg.get("out_name", "概览.md")
    sys_prompt = cfg.get("system_prompt") or SYSTEM_PROMPT
    map_prompt = cfg.get("map_prompt") or MAP_PROMPT
    max_tokens = int(cfg.get("max_output_tokens", 3000))

    out_path = out_dir / out_name
    cache_path = cache_dir / "summary.json"
    t0 = time.perf_counter()
    body, add = _materials(units, formulas, visuals)
    total_tok = _est_tokens(body)
    first_chunk = body[:20000]   # 首块锚定原文（前置元信息保底），direct 路径 body 本身全含

    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    def key_for(mode: str, material_hash_src: str) -> str:
        return hashlib.sha1(f"{PROMPT_VER}|{mode}|{client.model}|"
                            f"{len(sys_prompt)}|{material_hash_src}".encode()).hexdigest()[:16]

    stats = {"calls": 0, "tokens": "0+0", "in": 0, "out": 0}
    lock = threading.Lock()

    def _chat(system: str, user: str, cap: int) -> dict:
        entry = None
        if limiter is not None:
            entry = limiter.acquire(_est_tokens(user) + cap + 200)
        try:
            r = client.chat(system, user, max_tokens=cap)
        except Exception:
            if limiter is not None:
                limiter.correct(entry, 0)
            raise
        if limiter is not None:
            limiter.correct(entry, r["tok_in"] + r["tok_out"])
        with lock:
            stats["calls"] += 1
            stats["in"] += r["tok_in"]
            stats["out"] += r["tok_out"]
        return r

    try:
        if total_tok <= gate:
            # ---------------- DIRECT：全文直读 ----------------
            user = f"【论文正文（原文，按顺序完整给出）】\n{body}\n\n{add}"
            k = key_for("direct", user)
            if isinstance(cache.get(k), dict) and cache[k].get("text"):
                out_path.write_text(_render(cache[k]["text"]), encoding="utf-8")
                return {"status": "cached", "calls": 0, "tokens": "0+0",
                        "wall_s": round(time.perf_counter() - t0, 1), "path": str(out_path)}
            print(f"  [s6] direct 模式：正文 ~{total_tok:,} est-tok ≤ 门控 {gate:,}"
                  f"，单次全文总览", flush=True)
            r = _chat(sys_prompt, user, max_tokens)
            text = r["text"]
            status = "ok"
        else:
            # ---------------- MAPRED：40k 块要点 → reduce ----------------
            chunks = pack_chunks(units, target)
            print(f"  [s6] map-reduce 模式：正文 ~{total_tok:,} est-tok > 门控 {gate:,}"
                  f" → {len(chunks)} 块（目标 {target // 1000}k/块，并发 {map_conc}）", flush=True)
            notes: list[str | None] = [None] * len(chunks)

            def _map_key(i: int) -> str:
                src = "\n\n".join(u.src for u in chunks[i])
                return hashlib.sha1(f"m|{PROMPT_VER}|{client.model}|{len(map_prompt)}|{src}"
                                    .encode()).hexdigest()[:16]

            def _map(i: int):
                ck = _map_key(i)
                if isinstance(cache.get(ck), dict) and cache[ck].get("note"):
                    notes[i] = cache[ck]["note"]   # 块要点缓存命中：不重新计费
                    return
                seg = (f"这是论文第 {i + 1}/{len(chunks)} 块（原文顺序）：\n\n"
                       + "\n\n".join(u.src for u in chunks[i]))
                try:
                    txt = _chat(map_prompt, seg, 1300)["text"]
                    notes[i] = f"### 块 {i + 1}\n{txt}"
                    cache[ck] = {"note": notes[i]}
                except Exception as e:
                    log_event(log_path, {"stage": "s6", "level": "warn",
                                         "item": f"map#{i + 1}", "error": str(e)[:200]})
                    print(f"  [s6 map#{i + 1}] FAIL（跳过，其余块不受影响）: {str(e)[:80]}",
                          flush=True)

            with ThreadPoolExecutor(max_workers=map_conc) as ex:
                list(ex.map(_map, range(len(chunks))))
            n_ok = sum(1 for x in notes if x)
            if n_ok == 0:
                raise RuntimeError("所有分块要点抽取均失败")
            user = ("【分块要点（按原文顺序）】\n" + "\n\n".join(x for x in notes if x)
                    + f"\n\n【论文开头原文（锚定标题/机构/摘要等前置信息）】\n{first_chunk}\n\n{add}")
            k = key_for("mapred", user)
            if isinstance(cache.get(k), dict) and cache[k].get("text"):
                out_path.write_text(_render(cache[k]["text"]), encoding="utf-8")
                return {"status": "cached", "calls": stats["calls"], "tokens": "0+0",
                        "wall_s": round(time.perf_counter() - t0, 1), "path": str(out_path)}
            r = _chat(sys_prompt, user, max_tokens)
            text = r["text"]
            status = "ok" if n_ok == len(chunks) else "fallback"
            if n_ok < len(chunks):
                text += (f"\n\n> ⚠️ {len(chunks) - n_ok}/{len(chunks)} 个分块要点抽取失败"
                         "（已跳过，重跑自动补算）。")

        out_path.write_text(_render(text), encoding="utf-8")
        cache[k] = {"text": text}
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0),
                              encoding="utf-8")
        log_event(log_path, {"stage": "s6", "level": "info", "msg": "done",
                             "mode": "direct" if total_tok <= gate else "mapred",
                             "calls": stats["calls"], "tokens_in": stats["in"],
                             "tokens_out": stats["out"]})
        return {"status": status, "calls": stats["calls"],
                "tokens": f"{stats['in']}+{stats['out']}",
                "wall_s": round(time.perf_counter() - t0, 1), "path": str(out_path)}
    except Exception as e:
        # S6 失败不影响主产物：写占位 + 日志，返回状态
        out_path.write_text(f"# 概览\n\n> ❌ 概览生成失败：{str(e)[:200]}\n>\n"
                            "> 重跑 turbo 将自动补算（其余产物不受影响）。\n", encoding="utf-8")
        log_event(log_path, {"stage": "s6", "level": "error", "error": str(e)[:300]})
        return {"status": "fail", "calls": stats["calls"],
                "tokens": f"{stats['in']}+{stats['out']}",
                "wall_s": round(time.perf_counter() - t0, 1), "path": str(out_path)}
