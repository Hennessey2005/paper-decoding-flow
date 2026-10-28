# -*- coding: utf-8 -*-
"""全流程编排：S1 解析 → S2 公式 → S3 图表 → S4 翻译 → S5 组装。

断点复用：
    - cache_dir 下已有 middle_json.json → 跳过 S1（最贵的本地解析）
    - cache_dir 下已有 translations.json → S4 全部走缓存（0 API 调用）

turbo 模式（并发调度，参数来自 scripts/concurrency_bench.py 实测）：
    S3（并发 6）提交后台线程后立即进入 S4（并发 8），两阶段重叠执行、
    共享一个滑动窗口限流器（RPM+TPM），总请求严格压在供应商限额内
    （硅基流动 200RPM/2M TPM → 软限 180/1.8M）。
    default 模式行为完全不变（串行），turbo 稳定前互不影响。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .clients.llm import LLMClient, log_event
from .clients.ratelimit import SlidingWindowLimiter
from .clients.vlm import VLMClient
from .config import load_config
from .s2_formulas import extract_formulas, emit as emit_formulas
from .s3_figures import analyze as analyze_figures, collect_visuals, emit_figures
from .s4_translate import build_units, translate_units
from .s5_assemble import assemble
from .title_naming import extract_doc_title, resolve_out_dir, write_hash_marker


def _find_middle_json(cache_dir: Path) -> Path | None:
    direct = cache_dir / "middle_json.json"
    if direct.exists():
        return direct
    hits = list(cache_dir.glob("*middle*.json"))
    return hits[0] if hits else None


def _ollama_unload(base_url: str, model: str) -> None:
    """local 模式串行保证：S3 跑完立即卸载 VLM，释放显存给 S4 翻译模型。

    6GB 显存放不下 minicpmv26(~5.2GB)+gemma4b(~2.8GB) 双常驻，
    不显式卸载则 Ollama 在 keep_alive(5min) 内反复换入换出，两边都变慢。
    """
    import httpx
    root = base_url.rsplit("/v1", 1)[0]
    try:
        httpx.Client(trust_env=False, timeout=30).post(
            f"{root}/api/generate",
            json={"model": model, "keep_alive": 0})
        print(f"  [local] 已卸载 {model}，显存让位翻译模型", flush=True)
    except Exception as e:
        print(f"  [local] 卸载 {model} 失败（不影响产物）: {e}", flush=True)


def run_pipeline(pdf: Path, out_dir: Path | None, cfg: dict,
                 cache_dir: Path | None = None, mode: str = "default",
                 log_dir: Path | None = None) -> dict:
    """out_dir=None → S1 解析后按论文标题自动命名 output/<标题>（v2.1 P0-A，title_naming）。

    mode（平级模式，run.bat 菜单 / --mode 指定）：
      default  - 云端硅基流动（原默认，S3/S4 串行）
      turbo    - 云端硅基流动并发版：S3(6)∥S4(8) 重叠 + 全局 RPM/TPM 限流
      free     - 智谱 GLM-4.6V/4.7-Flash（免费云）
      local    - 本地 Ollama 翻译模型（本地 GPU），S3 图表理解跳过
      minimum  - 纯解析，不调任何 API
    """
    t_start = time.perf_counter()
    pdf = Path(pdf)
    # PDF 内容哈希：既隔离缓存，也作为输出目录的身份标记（title_naming）
    import hashlib
    pdf_hash = hashlib.sha1(pdf.read_bytes()).hexdigest()[:16]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        # 默认按 PDF 内容哈希隔离缓存：同一论文重跑自动复用，不同论文互不污染
        cache_dir = Path("output") / "_cache" / pdf_hash
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir else Path("log")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    mode = (mode or "default").lower()
    offline = (mode == "minimum")
    api = cfg.get("api", {})
    enable_thinking = True   # 本地/Ollama 端点不识别 enable_thinking，默认省略该参数
    vlm_max_retries, llm_max_retries = 3, 3
    # turbo 并发参数（2026-10-28 基准实测：conc=8/6 零 429；16 会顶 RPM 限故配限流器兜底）
    turbo = (mode == "turbo")
    s3_conc = 6 if turbo else 1
    s4_conc = 8 if turbo else 1
    limiter: SlidingWindowLimiter | None = None

    if mode == "free":
        fp = cfg.get("free", {})
        zk = (api.get("zhipu_api_key") or os.environ.get("ZHIPU_API_KEY", "")).strip()
        if not zk:
            raise RuntimeError(
                "free 模式需要 api.zhipu_api_key（或在环境变量 ZHIPU_API_KEY 中）")
        api = {
            "base_url": fp.get("base_url", "https://open.bigmodel.cn/api/paas/v4"),
            "api_key": zk,
            "vlm_model": fp.get("vlm_model", "GLM-4.6V-Flash"),
            "llm_model": fp.get("llm_model", "GLM-4.7-Flash"),
            "proxy": api.get("proxy"),
        }
        enable_thinking = False
        vlm_max_retries = llm_max_retries = 8
        print(f"  [mode] free (Zhipu): vlm={api['vlm_model']} llm={api['llm_model']}")
    elif mode == "local":
        lp = cfg.get("local", {})
        api = {
            "base_url": lp.get("base_url", "http://localhost:11434/v1"),
            "api_key": lp.get("api_key", "ollama"),
            "vlm_model": lp.get("vlm_model") or None,   # 填 Ollama 多模态模型（如 minicpmv26）解锁本地 S3
            "llm_model": lp.get("llm_model") or "qwen2.5:14b-instruct",
            "proxy": None,
            "max_concurrency": lp.get("max_concurrency", 1),
        }
        # enable_thinking 保持 True（不发送该字段）→ 兼容 Ollama
        if api["vlm_model"]:
            print(f"  [mode] local (Ollama): vlm={api['vlm_model']} llm={api['llm_model']}"
                  "  (S3 本地串行，6GB 显存下较慢，属预期)")
        else:
            print(f"  [mode] local (Ollama): llm={api['llm_model']}  (未配 vlm_model，S3 图表理解跳过)")
    elif mode == "minimum":
        print("  [mode] minimum (parse only, no API)")
    else:  # default / turbo（同一云端配置，仅调度方式不同）
        enable_thinking = False
        if turbo:
            rl = api.get("rate_limit") or {}
            rpm = int(rl.get("rpm", 180))
            tpm = int(rl.get("tpm", 1_800_000))
            limiter = SlidingWindowLimiter(rpm=rpm, tpm=tpm)
            print(f"  [mode] turbo (SiliconFlow concurrent): S3∥{s3_conc} S4∥{s4_conc} "
                  f"overlap, rate limit {rpm}RPM/{tpm:,}TPM (soft) <- provider 200/2,000,000")
        else:
            mode = "default"
        print(f"  [mode] {mode} ({api.get('provider', 'cloud')}): "
              f"vlm={api.get('vlm_model')} llm={api.get('llm_model')}")

    report: dict = {"pdf": pdf.name, "pdf_hash": pdf_hash,
                    "cache_dir": str(cache_dir), "stages": {}}

    def mark(stage: str, **kw):
        report["stages"][stage] = kw
        print(f"  [{stage}] " + " ".join(f"{k}={v}" for k, v in kw.items()))

    # ---------- S1 解析（MinerU basic 档，CPU） ----------
    os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    mj_path = _find_middle_json(cache_dir)
    if mj_path:
        print(f"  [s1] 复用缓存 {mj_path}")
        mark("s1", mode="cache", file=mj_path.name)
    else:
        from mineru.parser import parse
        from mineru.parser.writer import FileBasedDataWriter

        tier = cfg.get("tier", "basic")
        t0 = time.perf_counter()
        result = parse(str(pdf), tier=tier)
        result.save(FileBasedDataWriter(str(cache_dir)))
        mark("s1", mode="parse", tier=tier, seconds=round(time.perf_counter() - t0, 1))
        mj_path = _find_middle_json(cache_dir)
    assert mj_path, "middle_json 未产出"
    mj = json.loads(mj_path.read_text(encoding="utf-8"))
    assert mj.get("schema") == "docvortex.middle", "协议版本不兼容，请检查 MinerU 版本"

    # ---------- 输出目录定名（未显式 -o 时，按论文真实标题，S1 产物零成本） ----------
    if out_dir is None:
        doc_title = extract_doc_title(mj)
        out_dir = resolve_out_dir(Path("output"), doc_title, pdf.stem, pdf_hash)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_hash_marker(out_dir, pdf_hash)
        report["title"] = doc_title or pdf.stem
        mark("dir", out=str(out_dir), title=(doc_title or "")[:60] or "(回退文件名)")

    # ---------- S2 公式剥离（本地） ----------
    registry = extract_formulas(mj)
    emit_formulas(registry, out_dir / "formulas")
    mark("s2", formulas=len(registry),
         syntax_ok=sum(1 for f in registry if f.syntax["strict_ok"] or f.syntax["lenient_ok"]))

    # ---------- S3 图表理解（API；offline 跳过；local 配了 vlm_model 则本地 GPU 串行） ----------
    visuals = collect_visuals(mj, cache_dir)

    def _figs_snapshot(with_results: bool) -> list[dict]:
        """with_results=False：仅对齐字段（turbo 重叠期给 build_units 用）；
        True：S3 完成后的全量结果（给 S5 组装用）。"""
        out = []
        for v in visuals:
            d = dict(id=v.id, kind=v.kind, page_idx=v.page_idx, fig_no=v.fig_no,
                     bbox=v.bbox, image_path=v.image_path, skipped=v.skipped)
            if with_results:
                d.update(caption=v.caption, caption_zh=v.caption_zh,
                         description=v.description, data_table=v.data_table,
                         confidence=v.confidence)
            out.append(d)
        return out

    s3_thread = None
    s3_box: dict = {}          # 后台线程产物 {usage, error?}
    if offline:
        emit_figures(visuals, out_dir / "figures", cache_dir)
        mark("s3", mode="offline", figures=sum(1 for v in visuals if not v.skipped))
    elif mode == "local" and not api["vlm_model"]:
        emit_figures(visuals, out_dir / "figures", cache_dir)
        mark("s3", mode="skip", figures=sum(1 for v in visuals if not v.skipped))
    else:
        local_vlm = (mode == "local")
        vlm = VLMClient(api["base_url"], api["api_key"], api["vlm_model"],
                        proxy=api.get("proxy"), max_retries=vlm_max_retries,
                        enable_thinking=enable_thinking,
                        local=local_vlm, timeout=600.0 if local_vlm else 120.0)
        if turbo:
            # S3 提交后台：analyze 原地写 Visual 对象，内部单项失败已隔离；
            # 线程级异常兜底捕获进 s3_box，join 后统一报告（不让 S4 成果陪葬）
            import threading

            def _s3_bg():
                try:
                    s3_box["usage"] = analyze_figures(visuals, cache_dir, vlm, log_path,
                                                      concurrency=s3_conc, limiter=limiter)
                except Exception as e:  # noqa: BLE001
                    s3_box["error"] = str(e)[:300]
            s3_thread = threading.Thread(target=_s3_bg, name="s3-turbo", daemon=True)
            s3_thread.start()
        else:
            usage = analyze_figures(visuals, cache_dir, vlm, log_path)
            emit_figures(visuals, out_dir / "figures", cache_dir)
            if local_vlm:
                _ollama_unload(api["base_url"], api["vlm_model"])
            mark("s3", figures=sum(1 for v in visuals if not v.skipped),
                 tokens=f"{usage['in']}+{usage['out']}", fail=usage["fail"])

    figures_dicts = _figs_snapshot(with_results=(s3_thread is None))

    # ---------- S4 翻译（API；offline 跳过；turbo 并发+与 S3 重叠+共享限流） ----------
    units = build_units(mj, figures_dicts)
    title_src = next((u.src for u in units if u.kind == "title"), pdf.stem)
    if offline:
        mark("s4", mode="offline")
    else:
        llm = LLMClient(api["base_url"], api["api_key"],
                        api.get("llm_model") or api["vlm_model"],
                        proxy=api.get("proxy"), max_retries=llm_max_retries,
                        enable_thinking=enable_thinking)
        usage = translate_units(units, llm,
                                gloss_path=Path("config/glossary.yaml"),
                                doc_title=title_src,
                                cache_path=cache_dir / "translations.json",
                                log_path=log_path,
                                concurrency=s4_conc, limiter=limiter)

        # ---- turbo：等 S3 后台收敛，再产出 figures 并重建全量快照供 S5 ----
        if s3_thread is not None:
            if limiter is not None:
                print(f"  [s4] 限流实测: {limiter.stats(limiter.created)}", flush=True)
            s3_thread.join()
            if "error" in s3_box:
                print(f"  [s3] ⚠️ 后台异常（不影响译文，figures 按已完成部分产出）: "
                      f"{s3_box['error'][:120]}", flush=True)
            u3 = s3_box.get("usage", {"in": 0, "out": 0, "fail":
                                      sum(1 for v in visuals if v.error)})
            mark("s3", figures=sum(1 for v in visuals if not v.skipped),
                 tokens=f"{u3['in']}+{u3['out']}", fail=u3["fail"])
            figures_dicts = _figs_snapshot(with_results=True)
        mark("s4", calls=usage["calls"], cache_hit=usage["cache_hit"],
             tokens=f"{usage['in']}+{usage['out']}", flags=usage["flags"],
             fails=usage.get("fails", 0))

    if s3_thread is not None:
        # figures 产物统一在主线程 emit（S3 后台线程只写内存对象，避免并发写盘）
        emit_figures(visuals, out_dir / "figures", cache_dir)

    # ---------- S5 组装 ----------
    fig_by_id = {f["id"]: f for f in figures_dicts}
    title_unit = next((u for u in units if u.kind == "title"), None)
    check = assemble(units, fig_by_id, cache_dir, out_dir,
                     title_zh=title_unit.zh if title_unit else None)
    mark("s5", **{k: (v if isinstance(v, (int, str)) else "PASS" if v else "FAIL")
                  for k, v in check["checks"].items()},
         review=check["manual_review"] or "无")

    report["checks"] = check["checks"]
    report["outputs"] = {
        "输出目录": str(out_dir),
        "译文": str(out_dir / "译文.zh.md"),
        "对照": str(out_dir / "译文.bilingual.md"),
        "公式": str(out_dir / "formulas"),
        "图表": str(out_dir / "figures"),
        "溯源": str(out_dir / "segments.json"),
    }
    report["total_seconds"] = round(time.perf_counter() - t_start, 1)
    log_event(log_path, {"stage": "run", "level": "info", "msg": "done",
                         "pdf": pdf.name, "total_seconds": report["total_seconds"],
                         "checks": report["checks"]})
    return report
