# -*- coding: utf-8 -*-
"""S4 翻译：middle_json 块 → 翻译单元 → LLM 翻译（术语表+校验+缓存）。

块分类：
    doc_title / paragraph_title → title（翻译）
    text                        → text（翻译）
    ref_text                    → ref（保留原文，学术惯例）
    equation                    → equation（LaTeX 原样，不翻译）
    image / chart / table       → visual（关联 S3 的 figures.json 描述）
    aside_text / page_footnote  → skip（arXiv 侧栏水印 / 脚注，v1 不译）

校验（不通过则重试一次，再失败标记 flag）：
    - 行内 LaTeX 命令数量一致（如 mathbf 命令）
    - 引用编号 [12] / [1–8] 数量一致

并发（turbo 模式）：
    translate_units(concurrency=N>1, limiter=全局限流器) 时线程池并发翻译；
    - 缓存写盘加锁、每 8 单元合并落盘一次（降 I/O；Ctrl+C 尾部落盘，重启自动续跑）
    - 单项彻底失败（重试耗尽）→ 标记 unit.error 不中断整体，译文回退占位符可续跑
    - limiter：请求前按估算预留 token、返回后按实际校准，S3/S4 共享时总量不超限额
    concurrency=1（default/free/local 路径）保持原串行行为逐条落盘。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .clients.llm import LLMClient, log_event
from .clients.ratelimit import SlidingWindowLimiter
from .s2_formulas import _content_to_text, tidy_inline_math, to_markdown_math

LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")
CITE_RE = re.compile(r"\[\d+(?:\s*[–,-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–,-]\s*\d+)?)*\]")
# 行内数学段 $...$（不含展示公式 $$），用于译文与原文数量一致性校验
INLINE_MATH_RE = re.compile(r"(?<!\$)\$[^$\n]+\$(?!\$)")

PROMPT_VER = "v2"
SYSTEM_PROMPT = (
    "你是资深学术论文翻译引擎，将英文学术论文段落翻译成中文。规则："
    "1) 忠实原文，不增删信息，不加译注和总结；"
    "2) 术语表中的译名强制采用；未列入术语表的专业术语首次出现时在括号内附英文原词；"
    "3) 行内公式片段以 `$...$` 包裹（如 $d _ { k }$、$\\mathbf { x } [ n ]$），"
    "`$` 定界符与公式内容必须逐字符原样保留，不得翻译、改写、删除、增删空格或调换位置；"
    "4) 引用编号如 [12]、[1–8] 原样保留；数字、单位、百分比不变；"
    "5) 专有名词（模型名、数据集名、机构名）保留英文；"
    "6) 只输出译文正文本身，不要任何前言、解释或 Markdown 格式。"
)


@dataclass
class Unit:
    seg_id: str
    kind: str            # title | text | ref | equation | visual | skip
    page_idx: int
    block_index: int
    src: str = ""
    zh: str = ""
    meta: dict = field(default_factory=dict)   # visual: {figure_id, kind}; check flags 等


def _block_text(block: dict) -> str:
    # content 可能是纯字符串（equation 等）或 span 列表（text 等），统一处理
    return _content_to_text(block.get("content", "")).strip()


def build_units(middle_json: dict, figures_registry: list[dict]) -> list[Unit]:
    """遍历块生成翻译单元；visual 块按 (page_idx, kind, 顺序) 对齐 figures.json。"""
    # figures.json 与 s3.collect_visuals 同序；s3 的计数器按 kind 全局递增，此处保持一致
    kcount: dict[str, int] = {}
    fig_by_key: dict[tuple, dict] = {}
    for f in figures_registry:
        if f.get("skipped"):
            continue
        kcount[f["kind"]] = kcount.get(f["kind"], 0) + 1
        fig_by_key[(f["page_idx"], f["kind"], kcount[f["kind"]])] = f
    bcount: dict[str, int] = {}

    units: list[Unit] = []
    n = 0
    for page in middle_json.get("pages", []):
        pidx = page.get("page_idx", 0)
        for bi, b in enumerate(page.get("blocks", [])):
            t = b.get("type")
            idx = b.get("index", bi)
            if t in ("doc_title", "paragraph_title"):
                n += 1
                units.append(Unit(f"S-{n:03d}", "title", pidx, idx, _block_text(b)))
            elif t == "text":
                txt = _block_text(b)
                if not txt:
                    continue
                n += 1
                units.append(Unit(f"S-{n:03d}", "text", pidx, idx, txt))
            elif t == "ref_text":
                txt = _block_text(b)
                if txt:
                    n += 1
                    units.append(Unit(f"S-{n:03d}", "ref", pidx, idx, txt))
            elif t == "equation":
                n += 1
                units.append(Unit(f"S-{n:03d}", "equation", pidx, idx,
                                  _block_text(b), meta={"kind": "equation"}))
            elif t in ("image", "chart", "table"):
                bcount[t] = bcount.get(t, 0) + 1
                f = fig_by_key.get((pidx, t, bcount[t]))
                n += 1
                units.append(Unit(f"S-{n:03d}", "visual", pidx, idx,
                                  meta={"figure_id": f["id"] if f else None, "kind": t}))
            # aside_text / page_footnote → skip
    return units


def _load_glossary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in (data.get("terms") or {}).items()}


def _glossary_block(gloss: dict[str, str]) -> str:
    if not gloss:
        return "（无术语表）"
    return "\n".join(f"{k} → {v}" for k, v in gloss.items())


def _check(src: str, zh: str, kind: str = "text") -> list[str]:
    problems = []
    if len(LATEX_CMD_RE.findall(src)) != len(LATEX_CMD_RE.findall(zh)):
        problems.append("latex_cmd_mismatch")
    if len(INLINE_MATH_RE.findall(src)) != len(INLINE_MATH_RE.findall(zh)):
        problems.append("inline_math_mismatch")
    if len(CITE_RE.findall(src)) != len(CITE_RE.findall(zh)):
        problems.append("citation_mismatch")
    # 标题译成中文天然更短，不作长度检查
    if kind != "title" and len(zh) < 0.25 * len(src):
        problems.append("too_short")
    return problems


def _translate_serial(todo, keys, cache, cache_path, client, usage,
                      gloss_block, user_tpl, doc_title, log_path, retry_once) -> dict:
    """原串行路径（default/free/local）：逐条调用、逐条落盘、异常抛穿（行为与旧版一致）。"""
    n_todo = len(todo) - sum(1 for u in todo if keys[u.seg_id] in cache)
    for u in todo:
        key = keys[u.seg_id]
        if key in cache:
            u.zh = cache[key]
            usage["cache_hit"] += 1
            continue
        print(f"  [s4 {usage['calls'] + 1}/{n_todo}] {u.seg_id} ...", end="", flush=True)
        r = client.chat(SYSTEM_PROMPT,
                        user_tpl.format(title=doc_title, gloss=gloss_block, seg=u.src))
        usage["in"] += r["tok_in"]
        usage["out"] += r["tok_out"]
        usage["calls"] += 1
        zh = r["text"]
        # 兜底清洗：模型偶尔把行内公式改写成 \( \) 或在 $ 内侧加空格（KaTeX 不识别）
        zh = tidy_inline_math(to_markdown_math(zh))
        problems = _check(u.src, zh, u.kind)
        print(f" {r['duration_s']}s" + (f" flag={','.join(problems)}" if problems else ""),
              flush=True)
        if problems and retry_once:
            fix_hint = {"latex_cmd_mismatch": "行内 LaTeX 公式记号被改动，必须原样保留。",
                        "inline_math_mismatch": "行内公式 `$...$`（含定界符）被改动、丢失或增删，必须原样保留。",
                        "citation_mismatch": "引用编号被改动或丢失，必须原样保留。",
                        "too_short": "译文过短，疑似漏译，请完整翻译。"}
            r2 = client.chat(SYSTEM_PROMPT,
                             user_tpl.format(title=doc_title, gloss=gloss_block, seg=u.src)
                             + "\n\n注意：上一次输出存在问题——"
                             + "；".join(fix_hint.get(p, p) for p in problems) + " 请重新翻译。")
            usage["in"] += r2["tok_in"]
            usage["out"] += r2["tok_out"]
            usage["calls"] += 1
            zh2 = r2["text"]
            if not _check(u.src, zh2, u.kind):
                zh = zh2
            else:
                u.meta["flag"] = problems  # 保留首次译文并标记人工复核
                usage["flags"] += 1
        u.zh = zh
        cache[key] = zh
        log_event(log_path, {"stage": "s4", "level": "warn" if u.meta.get("flag") else "info",
                             "item": u.seg_id, "engine": client.model,
                             "duration_ms": int(r["duration_s"] * 1000),
                             "tokens_in": r["tok_in"], "tokens_out": r["tok_out"],
                             "problems": problems or None})
        # 每段落落盘一次缓存
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return usage


def _est_tokens(src: str, gloss_block: str, title: str, max_tokens: int = 2000) -> int:
    """限流预留估算：中英混合按 ~2.5 字符/token 粗估 prompt，加 max_tokens 预留。
    宁可高估（请求即滑窗占用），返回后 correct() 按实际多退少补。"""
    return int((len(src) + len(gloss_block) + len(title)) / 2.5) + max_tokens + 100


def translate_units(units: list[Unit], client: LLMClient, gloss_path: Path,
                    doc_title: str, cache_path: Path, log_path: Path,
                    retry_once: bool = True,
                    concurrency: int = 1,
                    limiter: SlidingWindowLimiter | None = None) -> dict:
    """翻译所有 title/text 单元。带磁盘缓存（断点续跑不重复计费）。

    concurrency>1（turbo）：ThreadPoolExecutor 并发 + 全局限流 + 批量落盘；
    单元素彻底失败不抛穿，标记 u.error（zh 留空 → S5 显示 ⏳ 占位，重跑自动补译）。
    """
    gloss = _load_glossary(gloss_path)
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    usage = {"in": 0, "out": 0, "calls": 0, "cache_hit": 0, "flags": 0, "fails": 0}
    gloss_block = _glossary_block(gloss)
    user_tpl = ("全文标题（已译）：{title}\n术语表（强制采用译名）：\n{gloss}\n\n"
                "待翻译段落：\n{seg}")

    todo = [u for u in units if u.kind in ("title", "text")]

    def _key(u: Unit) -> str:
        return hashlib.sha1(
            f"{PROMPT_VER}|{client.model}|{gloss_block}|{u.src}".encode()).hexdigest()[:16]

    # 预扫缓存：让用户开局就知道要调多少次 API（其余直接命中缓存跳过）
    keys = {u.seg_id: _key(u) for u in todo}
    n_cached = sum(1 for k in keys.values() if k in cache)
    print(f"  [s4] {len(todo)} units: {n_cached} cached, "
          f"{len(todo) - n_cached} to translate"
          + (f" (concurrency={concurrency})" if concurrency > 1 else ""), flush=True)

    # 串行路径（default/free/local）：保持原行为——逐条落盘、异常直接抛穿
    if concurrency <= 1:
        return _translate_serial(
            todo, keys, cache, cache_path, client, usage,
            gloss_block, user_tpl, doc_title, log_path, retry_once)

    # ---------- 并发路径（turbo） ----------
    pending = [u for u in todo if keys[u.seg_id] not in cache]
    for u in todo:
        if keys[u.seg_id] in cache:
            u.zh = cache[keys[u.seg_id]]
            usage["cache_hit"] += 1
    if not pending:
        return usage

    lock = threading.Lock()
    n_done = 0
    dirty = False
    total = len(pending)

    def _flush(force: bool = False):
        nonlocal dirty, n_done
        # 每 8 个完成单元或收尾时落盘一次：降 I/O，Ctrl+C 最多丢 7 条重译
        if dirty and (force or n_done % 8 == 0):
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0),
                                  encoding="utf-8")
            dirty = False

    def _one(u: Unit):
        nonlocal n_done, dirty
        # 单次 API 调用封装：带限流预留与重试统计
        def _call(seg: str) -> dict:
            entry = None
            if limiter is not None:
                entry = limiter.acquire(_est_tokens(seg, gloss_block, doc_title))
            try:
                r = client.chat(SYSTEM_PROMPT, user_tpl.format(
                    title=doc_title, gloss=gloss_block, seg=seg))
            except Exception:
                if limiter is not None:
                    limiter.correct(entry, 0)  # 失败请求也占额度，但保守清零重试压力
                raise
            if limiter is not None:
                limiter.correct(entry, r["tok_in"] + r["tok_out"])
            return r

        try:
            r = _call(u.src)
            zh = tidy_inline_math(to_markdown_math(r["text"]))
            problems = _check(u.src, zh, u.kind)
            if problems and retry_once:
                fix_hint = {"latex_cmd_mismatch": "行内 LaTeX 公式记号被改动，必须原样保留。",
                            "inline_math_mismatch": "行内公式 `$...$`（含定界符）被改动、丢失或增删，必须原样保留。",
                            "citation_mismatch": "引用编号被改动或丢失，必须原样保留。",
                            "too_short": "译文过短，疑似漏译，请完整翻译。"}
                try:
                    r2 = _call(u.src + "\n\n注意：上一次输出存在问题——"
                               + "；".join(fix_hint.get(p, p) for p in problems) + " 请重新翻译。")
                    zh2 = tidy_inline_math(to_markdown_math(r2["text"]))
                    problems2 = _check(u.src, zh2, u.kind)
                    if not problems2:
                        zh, problems = zh2, []      # 重试洗清：不再标记
                    r = {"tok_in": r["tok_in"] + r2["tok_in"],
                         "tok_out": r["tok_out"] + r2["tok_out"],
                         "duration_s": r["duration_s"] + r2["duration_s"],
                         "retries": r["retries"] + r2["retries"]}
                except Exception as e2:
                    # 补试调用本身失败：保留首次译文并标记（好过整单元判死）
                    problems = problems or [f"retry_call_failed: {str(e2)[:60]}"]
            with lock:
                usage["in"] += r["tok_in"]
                usage["out"] += r["tok_out"]
                usage["calls"] += 1
                if problems:
                    u.meta["flag"] = problems  # 保留译文并标记人工复核
                    usage["flags"] += 1
                u.zh = zh
                cache[keys[u.seg_id]] = zh
                dirty = True
                n_done += 1
                done_n = n_done
                log_event(log_path, {
                    "stage": "s4", "level": "warn" if problems else "info",
                    "item": u.seg_id, "engine": client.model,
                    "duration_ms": int(r["duration_s"] * 1000),
                    "tokens_in": r["tok_in"], "tokens_out": r["tok_out"],
                    "problems": problems or None})
                _flush()
            print(f"  [s4 {done_n}/{total}] {u.seg_id} {r['duration_s']}s"
                  + (f" flag={','.join(problems)}" if problems else ""), flush=True)
        except Exception as e:
            # 单项彻底失败：不抛穿（否则一整篇白跑），标记后 S5 用 ⏳ 占位，重跑续译
            with lock:
                u.meta["error"] = str(e)[:200]
                usage["fails"] += 1
                log_event(log_path, {"stage": "s4", "level": "error",
                                     "item": u.seg_id, "error": u.meta["error"]})
            print(f"  [s4 FAIL] {u.seg_id}: {str(e)[:90]}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(_one, pending))
    _flush(force=True)
    return usage
