# -*- coding: utf-8 -*-
"""S3 图表理解：视觉块配对图注 → VLM 生成中文描述与数据表 → figures.json/md。

MinerU 4.x 实测结构：
- 视觉块 content 内已含图注 span（Figure N: ... / Table N: ...）
- 裁剪图路径可确定性构造：images/page_{page_idx}_{type}_{index}.jpg
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .clients.vlm import VLMClient, parse_model_json, log_event
from .s2_formulas import _content_to_text, tidy_inline_math, to_markdown_math

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.?|Table)\s*(\d+)\s*[:.]", re.I)

# 系统提示（防幻觉约束 + 结构化输出）
SYSTEM_PROMPT = (
    "你是论文图表解析助手。只描述图中可见信息，禁止推测图中不存在的内容；"
    "图中数字必须与原图一致。始终输出一个 JSON 对象，不要输出任何其他文字。"
)


def _prompt_for(kind: str, caption: str, context: str) -> str:
    kind_hint = {
        "image": "这是论文中的插图（可能是示意图/流程图/照片/频谱图等）",
        "chart": "这是论文中的数据图表（折线/柱状/散点等）",
        "table": "这是论文中的表格截图",
    }[kind]
    data_rule = (
        "若图中可读出数据点/数值，将其整理为 Markdown 表格放入 data_table；读不出则填 null"
        if kind in ("chart", "table") else "data_table 固定为 null"
    )
    return f"""请解析这张论文{kind_hint}，输出以下 JSON：
{{
 "figure_type": "chart|diagram|photo|spectrogram|table|schematic|other",
 "caption_zh": "图注的中文翻译",
 "description": "给人类读者的中文内容描述：图里画了什么、坐标轴/变量/模块的含义、关键趋势或结论（150-300字）",
 "data_table": "Markdown表格字符串或null",
 "confidence": 0到1的小数
}}
本图的图注原文：{caption or "（未找到，请依据图内容判断）"}
图注中的描述若与图不符，以图为准。
{data_rule}
上下文（论文原文摘录，帮助理解术语）：{context[:300]}"""


@dataclass
class Visual:
    id: str                     # FIG-002-01 / TAB-003-01
    kind: str                   # image | chart | table
    fig_no: int | None          # 图注中的编号（Figure 3 → 3）
    page: int
    page_idx: int
    bbox: list
    image_path: str             # 相对 cache 的路径（images/...）
    caption: str = ""
    context_before: str = ""
    context_after: str = ""
    skipped: bool = False
    skip_reason: str = ""
    # ---- VLM 结果 ----
    figure_type: str = ""
    caption_zh: str = ""
    description: str = ""
    data_table: str | None = None
    confidence: float = 0.0
    usage: dict = field(default_factory=dict)
    error: str = ""


def _span_text(s: dict) -> str:
    """span 文本提取，兼容三层形态：str / list[dict] / 嵌套序列化的 list-repr 字符串
    （MinerU 4.x 视觉块的 caption span 实测为 repr 字符串，需再剥一层）。"""
    c = s.get("content", "")
    if isinstance(c, str) and c.lstrip().startswith("["):
        try:
            import json_repair
            inner = json_repair.loads(c)
            if isinstance(inner, list):
                parts = [str(d.get("content", "")) for d in inner if isinstance(d, dict)]
                joined = " ".join(p for p in parts if p).strip()
                if joined:
                    return joined
        except Exception:
            pass
        return c.strip()
    if isinstance(c, list):
        return _content_to_text(c)
    return str(c).strip()


def collect_visuals(middle_json: dict, cache_dir: Path,
                    min_area: float = 0.02) -> list[Visual]:
    """收集视觉块并配对图注。装饰性小图（无图注且面积过小）标记 skipped。"""
    visuals: list[Visual] = []
    counters: dict[str, int] = {}
    for page in middle_json.get("pages", []):
        pidx = page.get("page_idx", 0)
        blocks = page.get("blocks", [])
        for i, b in enumerate(blocks):
            kind = b.get("type")
            if kind not in ("image", "chart", "table"):
                continue
            # 图注：MinerU 4.x 视觉块 content 内含 image_caption/chart_caption/table_caption span
            spans = b.get("content") or []
            cap = ""
            fig_no = None
            img_path = ""
            for s in spans:
                if not isinstance(s, dict):
                    continue
                if not img_path and s.get("image_path"):
                    img_path = str(s["image_path"])
                t = _span_text(s)
                if t and CAPTION_RE.match(t):
                    cap, fig_no = t, int(CAPTION_RE.match(t).group(2))
                    break
            if not cap:  # 兜底：邻块找图注
                for j in (i + 1, i - 1, i + 2, i - 2):
                    if 0 <= j < len(blocks) and blocks[j].get("type") in ("text", "paragraph_title"):
                        t = _content_to_text(blocks[j].get("content", "")).strip()
                        m = CAPTION_RE.match(t)
                        if m:
                            cap, fig_no = t, int(m.group(2))
                            break

            # 装饰性过滤：无图注且面积过小
            x0, y0, x1, y1 = (b.get("bbox") or [0, 0, 0, 0])
            area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
            idx = b.get("index")
            counters.setdefault(kind, 0)
            counters[kind] += 1
            prefix = {"image": "FIG", "chart": "FIG", "table": "TAB"}[kind]
            v = Visual(
                id=f"{prefix}-{pidx + 1:03d}-{counters[kind]:02d}",
                kind=kind, fig_no=fig_no, page=pidx + 1, page_idx=pidx,
                bbox=[round(x, 4) for x in b.get("bbox", [])],
                image_path=img_path or f"images/page_{pidx}_{kind}_{idx}.jpg",
                caption=cap,
                context_before=(_content_to_text(blocks[i - 1].get("content", ""), 300)
                                if i > 0 and blocks[i - 1].get("type") in ("text",) else ""),
                context_after=(_content_to_text(blocks[i + 1].get("content", ""), 300)
                               if i + 1 < len(blocks) and blocks[i + 1].get("type") in ("text",) else ""),
            )
            if not cap and area < min_area:
                v.skipped = True
                v.skip_reason = f"无图注且面积过小({area:.3f})"
            # 校验图片文件存在
            if not (cache_dir / v.image_path).exists():
                hits = list((cache_dir / "images").glob(f"page_{pidx}_{kind}_*.jpg"))
                if hits:
                    v.image_path = f"images/{hits[0].name}"
                else:
                    v.skipped = True
                    v.skip_reason = "裁剪图缺失"
            visuals.append(v)
    return visuals


def analyze(visuals: list[Visual], cache_dir: Path, client: VLMClient,
            log_path: Path) -> dict:
    """顺序调用 VLM 解析每个未跳过的视觉块。返回用量汇总。"""
    total = {"in": 0, "out": 0, "calls": 0, "fail": 0}
    todo = [v for v in visuals if not v.skipped]
    print(f"  [s3] {len(todo)} visuals to analyze", flush=True)
    for i, v in enumerate(todo, 1):
        print(f"  [s3 {i}/{len(todo)}] {v.id} ({v.kind}) ...", end="", flush=True)
        # 本地 Ollama（6GB 实测）：table 长输出需 4096 上下文，chart/image 用 2048 更快
        num_ctx = 4096 if v.kind == "table" else 2048
        resp = client.chat_image(
            cache_dir / v.image_path, SYSTEM_PROMPT,
            _prompt_for(v.kind, v.caption, v.context_before + " " + v.context_after),
            max_tokens=4096 if v.kind == "table" else 2000,
            num_ctx=num_ctx if client.local else None,
        )
        total["in"] += resp["usage"]["in"]
        total["out"] += resp["usage"]["out"]
        total["calls"] += 1
        v.usage = resp["usage"]
        try:
            obj = parse_model_json(resp["text"])
            # VLM 倾向用 LaTeX 定界符 \( \) / \[ \]，KaTeX（mdview）只认 $ / $$，统一转写
            def _md(s: str) -> str:
                return tidy_inline_math(to_markdown_math(s))
            v.figure_type = str(obj.get("figure_type", ""))[:24]
            v.caption_zh = _md(str(obj.get("caption_zh", ""))[:300])
            v.description = _md(str(obj.get("description", ""))[:1200])
            dt = obj.get("data_table")
            v.data_table = _md(str(dt)[:4000]) if dt not in (None, "null", "", "None") else None
            v.confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
        except Exception as e:
            v.error = f"JSON解析失败: {e}"
            total["fail"] += 1
        print(f" {resp['duration_s']}s" + (f" FAIL {v.error[:60]}" if v.error else ""),
              flush=True)
        log_event(log_path, {
            "stage": "s3", "level": "info" if not v.error else "warn",
            "item": v.id, "engine": client.model,
            "duration_ms": int(resp["duration_s"] * 1000),
            "tokens_in": resp["usage"]["in"], "tokens_out": resp["usage"]["out"],
            "error": v.error or None,
        })
    total["fail"] = total["fail"] + sum(1 for v in visuals if v.error)
    return total


def emit_figures(visuals: list[Visual], out_dir: Path, cache_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures.json").write_text(
        json.dumps([asdict(v) for v in visuals], ensure_ascii=False, indent=1),
        encoding="utf-8")

    lines = ["# 图表内容理解", "",
             f"共 {sum(1 for v in visuals if not v.skipped)} 个有效图表"
             f"（跳过 {sum(1 for v in visuals if v.skipped)} 个装饰性/缺失项）。", ""]
    for v in visuals:
        if v.skipped:
            lines += [f"## {v.id}（已跳过：{v.skip_reason}）", ""]
            continue
        label = f"{'Table' if v.kind == 'table' else 'Figure'} {v.fig_no}" if v.fig_no else v.kind
        warn = "\n\n> ⚠️ 以下描述置信度较低，建议对照原图核验。" if v.confidence < 0.7 else ""
        lines += [
            f"## {v.id}（{label}，第 {v.page} 页，{v.figure_type or v.kind}）", "",
            f"![{v.id}](../{cache_dir.name}/{v.image_path})" if False else
            f"![{v.id}](\"{os.path.relpath(cache_dir / v.image_path, out_dir).replace(os.sep, '/')}\")", "",
            f"- **图注原文**: {v.caption or '（未检出）'}",
            f"- **图注译文**: {v.caption_zh or '—'}",
            f"- **置信度**: {v.confidence:.2f}", "",
            v.description or "（描述生成失败）" + warn, "",
        ]
        if v.data_table:
            lines += ["**数据表**", "", v.data_table, ""]
        if v.error:
            lines += [f"> ❌ {v.error}", ""]
    (out_dir / "figures.md").write_text("\n".join(lines), encoding="utf-8")
