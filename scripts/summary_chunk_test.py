# -*- coding: utf-8 -*-
"""turbo 概览功能预研（付费实测）：map-reduce 切块能否给模型足够的信息量。

问题：论文很长（页数/字数远超上下文友好区）时，按 ~9k token 分块抽取要点、
再汇总生成总览——模型只见到「块要点」这层压缩信息，是否足以产出高质量总览？

方法（对应方案设计的 S6，本脚本先行验证可行性与成本）：
  [1] 统计：units / 总字符 / 估算 token，切块方案（文档序装箱，段落边界）
  [2] MAP   ：每块一次 LLM 调用「抽取要点」（并发 4，共享限流器）
  [3] REDUCE：块要点 + 图注清单 + 公式注册表摘要 → 终稿概览（一次长调用）
  [4] 评测  ：把「概览 vs 论文自带 Abstract」交给模型对比，列出概览遗漏的
              摘要要点、以及概览中摘要未提及的信息——覆盖率是信息量是否
              充足的代理指标（作者自己认为重要的点，管线是否都传到了）
  另附 [5] 对照基线：只喂 Abstract+图注（不切块）生成的短概览，
      用于说明「切块管线相对裸摘要的增量信息」是否存在。

用法（仓库根目录，需 config.yaml 的硅基 key）：
  .venv\\Scripts\\python.exe scripts\\summary_chunk_test.py \
      --cache output/_cache/9e5fc979a813f2cc
可选：--chunk-tokens 9000  --map-conc 4  --max-chunks 16

产出：output/_summary_test/<hash>/  {overview.md, abstract_only.md,
      evaluation.md, chunks_stat.json}
成本预估：map n 块×(9k入+0.6k出) + reduce ~12k入+2.5k出 + 评测 ~6k
      —— 40 页论文约 400k token ≈ ¥1.5 内（DeepSeek-V4.1-Flash 价）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paper_translate.clients.llm import LLMClient
from paper_translate.clients.ratelimit import SlidingWindowLimiter
from paper_translate.config import load_config
from paper_translate.s2_formulas import extract_formulas
from paper_translate.s3_figures import collect_visuals
from paper_translate.s4_translate import build_units

# ---- 学科中立默认模板（真实功能里放 config.summary.system_prompt 可自定义） ----
MAP_SYS = (
    "你是学术文献精读助手。针对给出的论文片段（论文按原文顺序切块，"
    "你只看到其中一块），抽取该片段的关键信息，供后续生成全文总览使用。要求："
    "1) 保留：研究问题/动机、方法与模型结构（含具体名称、参数规模、架构要点）、"
    "数据集与数据来源、实验设置、关键数值结果（保留原始数字）、结论与创新点、"
    "局限或未来工作；2) 公式只记编号与一句话作用，不抄 LaTeX；"
    "3) 输出不超过 400 字的中文要点列表，信息密集，不要客套与总结句；"
    "4) 片段开头/结尾可能是半句话，忽略截断噪声。只输出要点。"
)
REDUCE_SYS = (
    "你是学术文献综述撰写者。用户给出：一篇论文的分块要点（按原文顺序）、"
    "图表清单、公式清单。请写出一份供人快速查阅的中文总览，结构："
    "## 一句话概括 / ## 问题与动机 / ## 方法（模型·数据·训练，含关键参数数值） / "
    "## 实验与结果（保留关键数字） / ## 创新点与局限。要求：忠实于给定材料，"
    "不编造；材料间有冲突时并列说明；总长 600-1000 字；输出 Markdown。"
)
EVAL_SYS = (
    "你是质检员。对比「模型生成的论文总览」与「论文作者自写的英文摘要」，"
    "回答两点，输出 Markdown：1) 摘要中出现而总览遗漏的关键信息点（逐条列出，"
    "没有则写无）；2) 总览中额外包含而摘要未提及的信息（判断这些是增量价值"
    "还是可疑幻觉，各举 2-3 例）。不要评价文笔。"
)


def est_tokens(s: str) -> int:
    return int(len(s) / 2.5)


def pack_chunks(units, chunk_tokens: int, max_chunks: int):
    """文档序装箱：text/title 单元按估算 token 累积到 ~chunk_tokens 切一刀。
    超出 max_chunks 时把目标块尺寸按比例放大（合并策略），保证调用次数封顶。"""
    body = [u for u in units if u.kind in ("text", "title", "ref", "equation")]
    total = sum(est_tokens(u.src) for u in body)
    target = max(chunk_tokens, est_tokens(" ".join(u.src for u in body)) // max_chunks
                 if max_chunks else chunk_tokens)
    chunks, cur, cur_tok = [], [], 0
    for u in body:
        t = est_tokens(u.src)
        if cur and cur_tok + t > target:
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks, total, target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="middle_json 所在解析缓存目录")
    ap.add_argument("--chunk-tokens", type=int, default=9000)
    ap.add_argument("--max-chunks", type=int, default=16, help="map 调用次数封顶（控成本）")
    ap.add_argument("--map-conc", type=int, default=4)
    ap.add_argument("--out", default="output/_summary_test")
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    mj = json.loads((cache_dir / "middle_json.json").read_text(encoding="utf-8"))
    cfg = load_config()
    api = cfg["api"]
    client = LLMClient(api["base_url"], api["api_key"], api["llm_model"],
                       proxy=api.get("proxy") or None, enable_thinking=False)
    limiter = SlidingWindowLimiter(rpm=180, tpm=1_800_000)
    out_dir = ROOT / args.out / cache_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    usage = {"in": 0, "out": 0, "calls": 0, "fails": 0}

    # ---------- [1] 切块统计 ----------
    units = build_units(mj, [])
    chunks, total_tok, target = pack_chunks(units, args.chunk_tokens, args.max_chunks)
    sizes = [sum(est_tokens(u.src) for u in c) for c in chunks]
    print(f"[1] 单元 {len(units)}，可切块文本约 {total_tok:,} est-tokens → "
          f"{len(chunks)} 块（目标块大小 {target:,}），块大小 "
          f"min/avg/max = {min(sizes)}/{int(statistics.mean(sizes))}/{max(sizes)}")
    (out_dir / "chunks_stat.json").write_text(json.dumps({
        "units": len(units), "est_total_tokens": total_tok, "n_chunks": len(chunks),
        "target": target, "chunk_sizes": sizes}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # 图注清单 / 公式清单（reduce 的附加上下文，本地零成本）
    visuals = [v for v in collect_visuals(mj, cache_dir) if not v.skipped]
    fig_lines = [f"- {v.id}（第{v.page}页，{v.kind}）：{v.caption[:160] or '（无图注）'}"
                 for v in visuals]
    registry = extract_formulas(mj)
    eq_lines = [f"- {f.id or ('eq' + str(i))}（第{f.page}页）" for i, f in enumerate(registry)][:40]
    print(f"    图注 {len(fig_lines)} 条 / 公式 {len(registry)} 条")

    # ---------- [2] MAP：并发块要点抽取 ----------
    print(f"[2] map：{len(chunks)} 块 × 并发{args.map_conc}", flush=True)
    notes: list[str | None] = [None] * len(chunks)

    def _map(i: int):
        text = "\n\n".join(u.src for u in chunks[i])
        seg = (f"这是论文第 {i + 1}/{len(chunks)} 块"
               f"（原文顺序，起始章节语境：{chunks[i][0].kind}）：\n\n{text}")
        entry = limiter.acquire(est_tokens(seg) + 1000)
        try:
            r = client.chat(MAP_SYS, seg, max_tokens=900)
        except Exception as e:
            limiter.correct(entry, 0)
            with lock:
                usage["fails"] += 1
            print(f"  map#{i + 1} FAIL {str(e)[:80]}", flush=True)
            return
        limiter.correct(entry, r["tok_in"] + r["tok_out"])
        notes[i] = f"### 块 {i + 1}（est {sizes[i]:,} tok）\n{r['text']}"
        with lock:
            usage["in"] += r["tok_in"]; usage["out"] += r["tok_out"]; usage["calls"] += 1
        print(f"  map#{i + 1}/{len(chunks)} {r['duration_s']}s in={r['tok_in']} "
              f"out={r['tok_out']}", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.map_conc) as ex:
        list(ex.map(_map, range(len(chunks))))
    map_wall = time.perf_counter() - t0
    n_ok = sum(1 for x in notes if x)
    print(f"    map 完成 {n_ok}/{len(chunks)}，墙钟 {map_wall:.0f}s")

    # ---------- [3] REDUCE：终稿概览 ----------
    material = (
        "【分块要点（按原文顺序）】\n" + "\n\n".join(x for x in notes if x)
        + "\n\n【图表清单】\n" + "\n".join(fig_lines)
        + "\n\n【公式清单（数量与位置）】\n"
        + ("\n".join(eq_lines) or "（无）"))
    entry = limiter.acquire(est_tokens(material) + 2500)
    r = client.chat(REDUCE_SYS, material, max_tokens=2500)
    limiter.correct(entry, r["tok_in"] + r["tok_out"])
    with lock:
        usage["in"] += r["tok_in"]; usage["out"] += r["tok_out"]; usage["calls"] += 1
    overview = r["text"]
    (out_dir / "overview.md").write_text(overview, encoding="utf-8")
    print(f"[3] reduce：{r['duration_s']}s in={r['tok_in']} out={r['tok_out']} "
          f"→ overview.md")

    # ---------- [4] 评测：概览 vs 作者 Abstract ----------
    from paper_translate.s2_formulas import _content_to_text
    abstract = ""
    pages = mj.get("pages", [])
    if pages:
        for b in pages[0].get("blocks", [])[:14]:
            if b.get("type") == "text":
                abstract += _content_to_text(b.get("content", "")).strip() + " "
    m = re.search(r"(?:a abstract|abstract)[:：]?\s*(.{100,})", abstract, re.I)
    abstract = (m.group(1) if m else abstract)[:1400]
    entry = limiter.acquire(est_tokens(overview) + 1600)
    ev = client.chat(EVAL_SYS, f"总览：\n{overview}\n\n作者摘要：\n{abstract}",
                     max_tokens=1200)
    limiter.correct(entry, ev["tok_in"] + ev["tok_out"])
    with lock:
        usage["in"] += ev["tok_in"]; usage["out"] += ev["tok_out"]; usage["calls"] += 1
    (out_dir / "evaluation.md").write_text(ev["text"], encoding="utf-8")
    print(f"[4] 评测完成 {ev['duration_s']}s → evaluation.md")

    # ---------- [5] 对照基线：只喂 Abstract+图注 ----------
    fig_block = "\n".join(fig_lines) if fig_lines else "（无）"
    base_user = (f"论文标题（首块）与作者摘要：\n{abstract}\n\n"
                 f"图表清单：\n{fig_block}\n\n"
                 "请仅依据以上内容写快速总览（其余同上要求）。")
    entry = limiter.acquire(est_tokens(base_user) + 2000)
    b = client.chat(REDUCE_SYS, base_user, max_tokens=1800)
    limiter.correct(entry, b["tok_in"] + b["tok_out"])
    with lock:
        usage["in"] += b["tok_in"]; usage["out"] += b["tok_out"]; usage["calls"] += 1
    (out_dir / "abstract_only.md").write_text(b["text"], encoding="utf-8")
    print(f"[5] 基线（裸摘要版）完成 → abstract_only.md")

    print(f"\n汇总：API 调用 {usage['calls']}（map 失败 {usage['fails']}）"
          f"，tokens {usage['in']:,}+{usage['out']:,}，"
          f"限流窗口 {limiter.stats(limiter.created)}")
    print(f"产物目录：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
