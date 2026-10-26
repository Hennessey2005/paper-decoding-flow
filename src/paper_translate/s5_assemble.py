# -*- coding: utf-8 -*-
"""S5 组装：翻译单元 + figures.json → 译文.zh.md / 译文.bilingual.md / segments.json + 自检。"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _figure_block(unit, fig: dict, cache_dir: Path, out_dir: Path) -> list[str]:
    """visual 单元 → 引用块（图注译文 + 图 + VLM 描述 + 数据表）。"""
    img = cache_dir / fig.get("image_path", "")
    rel = os.path.relpath(img, out_dir).replace(os.sep, "/")
    label = "Table" if unit.meta.get("kind") == "table" else "Figure"
    no = f" {fig['fig_no']}" if fig.get("fig_no") else ""
    lines = [
        f"> **{label}{no}**（{fig['id']}）{fig.get('caption_zh') or fig.get('caption', '')}",
        ">",
        f"> ![{fig['id']}]({rel})",
        ">",
    ]
    desc = fig.get("description") or ""
    if desc:
        lines += ["> " + line for line in desc.splitlines()] + [">"]
    if fig.get("data_table"):
        lines += ["> **数据表**：", ">"] + ["> " + l for l in fig["data_table"].splitlines()] + [">"]
    if fig.get("confidence", 1) < 0.7:
        lines.append("> ⚠️ 描述置信度较低，建议对照原图。")
    while lines and lines[-1] == ">":
        lines.pop()
    return lines


DISCLAIMER = "> ⚠️ AI读图翻译可能存在幻觉，请以原文为准。"


def assemble(units, figures_by_id: dict, cache_dir: Path, out_dir: Path,
             zh_name: str = "译文.zh.md", bi_name: str | None = "译文.bilingual.md",
             title_zh: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    zh_lines: list[str] = []
    bi_lines: list[str] = ["# 双语对照（中 / EN）", DISCLAIMER, ""]
    segments = []
    title_done = False

    for u in units:
        seg = {"seg_id": u.seg_id, "kind": u.kind, "page_idx": u.page_idx,
               "block_index": u.block_index}
        if u.kind == "title":
            head = "# " if not title_done else "## "
            title_done = True
            text = title_zh if (head == "# " and title_zh) else (u.zh or u.src)
            zh_lines += [head + text, ""]
            if head == "# ":  # 篇首一级标题下：AI 产物免责行（双语版已在文首）
                zh_lines += [DISCLAIMER, ""]
            bi_lines += [f"**{u.zh or u.src}** / {u.src}", ""]
            seg.update({"src": u.src, "zh": u.zh or u.src})
        elif u.kind == "text":
            zh_lines += [u.zh or f"⏳ {u.src}", ""]
            if u.zh:
                bi_lines += [u.zh, "", f"> {u.src}", ""]
            seg.update({"src": u.src, "zh": u.zh, "flag": u.meta.get("flag")})
        elif u.kind == "ref":
            zh_lines += [u.src, ""]
            seg.update({"src": u.src, "zh": None, "note": "references-kept"})
        elif u.kind == "equation":
            zh_lines += [f"$${u.src}$$", ""]
            seg.update({"src": u.src, "zh": None, "note": "latex-kept"})
        elif u.kind == "visual":
            fid = u.meta.get("figure_id")
            fig = figures_by_id.get(fid)
            if fig:
                zh_lines += _figure_block(u, fig, cache_dir, out_dir) + [""]
            seg.update({"figure_id": fid})
        segments.append(seg)

    (out_dir / zh_name).write_text("\n".join(zh_lines), encoding="utf-8")
    if bi_name:
        (out_dir / bi_name).write_text("\n".join(bi_lines), encoding="utf-8")
    (out_dir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 自检 ----
    zh_md = "\n".join(zh_lines)
    # 展示公式按整行 $$...$$ 计数，避免正文行内 `$a$$b$` 相连时误计
    display_n = sum(1 for ln in zh_md.splitlines()
                    if ln.strip().startswith("$$") and ln.strip().endswith("$$")
                    and len(ln.strip()) > 4)
    checks = {
        "无残留占位符": "\u27e6" not in zh_md,
        "公式数与解析一致": display_n == sum(1 for u in units if u.kind == "equation"),
        "图均插入": sum(1 for u in units if u.kind == "visual"
                    and u.meta.get("figure_id")) ==
            sum(1 for u in units if u.kind == "visual"),
        "图片链接存在": all(
            (out_dir / l).exists() for l in
            [p for p in _extract_images(zh_md, out_dir)]),
    }
    flags = [u.seg_id for u in units if u.meta.get("flag")]
    return {"checks": checks, "manual_review": flags,
            "segments": len(segments)}


def _extract_images(md: str, out_dir: Path):
    import re
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
        yield m.group(1)
