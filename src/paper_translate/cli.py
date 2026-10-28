# -*- coding: utf-8 -*-
"""paper-translate 命令行入口。

用法:
    paper-translate paper.pdf                 # 全流程（解析+公式+图表理解+翻译）
    paper-translate paper.pdf --offline       # 仅本地阶段（解析+公式），不调 API
    paper-translate paper.pdf -o out/         # 指定输出目录
    paper-translate paper.pdf --cache-dir d/  # 复用已有解析缓存（跳过最贵的 S1）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .orchestrator import run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="paper-translate",
        description="PDF 论文 → 深度理解包（中文译文 + LaTeX 公式注册表 + 图表内容描述）")
    p.add_argument("pdf", help="输入 PDF 路径")
    p.add_argument("-o", "--output", default=None,
                   help="输出目录（默认 output/<论文标题>/，标题取自 PDF 解析结果）")
    p.add_argument("--cache-dir", default=None, help="解析缓存目录（复用可跳过 S1）")
    p.add_argument("--tier", default=None, help="MinerU 档位 basic|standard|flash（默认读配置）")
    p.add_argument("--mode", default="default",
                   choices=["default", "turbo", "free", "local", "minimum"],
                   help="运行模式：default(硅基云·串行) | turbo(硅基云·并发重叠，实验) | "
                        "free(智谱免费云) | local(Ollama本地翻译) | minimum(纯解析)")
    p.add_argument("--free", dest="free", action="store_true",
                   help="[兼容别名] 等价于 --mode free")
    p.add_argument("--offline", dest="offline", action="store_true",
                   help="[兼容别名] 等价于 --mode minimum")
    p.add_argument("--summary", dest="summary", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="turbo 模式下生成「概览.md」全文总览；--summary/--no-summary "
                        "覆盖 config.summary.enabled（默认跟随配置文件；仅 turbo 模式生效）")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config/config.yaml）")
    p.add_argument("--open", dest="open_out", action="store_true",
                   help="完成后用系统文件管理器打开输出目录（run.bat 使用）")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.tier:
        cfg["tier"] = args.tier

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"输入不存在: {pdf}")
        return 2
    # 未指定 -o 时由流水线按论文真实标题定名：output/<标题>（v2.1 P0-A）
    out_dir = Path(args.output) if args.output else None

    # 解析最终 mode：--mode 优先；否则兼容别名 --free / --offline
    mode = args.mode
    if mode == "default" and args.free:
        mode = "free"
    if mode in ("default", "free") and args.offline:
        mode = "minimum"

    print(f"paper-translate: {pdf.name}  tier={cfg.get('tier')}  mode={mode}")
    try:
        report = run_pipeline(pdf, out_dir, cfg,
                              cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                              mode=mode, summary=args.summary)
    except KeyboardInterrupt:
        # Ctrl+C：已完成的 S3 项 / S4 单元均已落盘缓存，重跑自动从断点继续
        print("\n[interrupted] 已手动中止。已完成的单元有缓存，重新运行将从断点继续。")
        return 130
    print(f"\n完成（{report['total_seconds']}s）")
    for k, v in report["outputs"].items():
        print(f"  {k}: {v}")
    ok = all(report["checks"].values())
    print("自检:", "PASS" if ok else "FAIL")
    if args.open_out:
        _open_in_file_manager(Path(report["outputs"]["输出目录"]))
    return 0 if ok else 1


def _open_in_file_manager(path: Path) -> None:
    """跨平台尽力而为：失败静默（打开目录只是便利，不影响产物）。"""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
