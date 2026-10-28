# -*- coding: utf-8 -*-
"""并发调度基准测试（独立脚本，不动业务代码）。

背景：S3/S4 目前完全串行，长耗时单元（大表 VLM、长段落）阻塞后续全部请求；
硅基流动限额 200 RPM / 2,000,000 TPM，单请求平均仅 ~2-25s，实际远未打满限额。
本脚本用真实模型、真实论文数据测量不同并发档位下的吞吐与延迟，
为管线改造（S3∥S4 重叠 + 单元级并发 + 全局限流）提供参数依据。

场景（--scenarios 逗号分隔）：
  s4     翻译单元并发扫描（档位 × 样本）
  s3     图表 VLM 并发扫描
  mixed  S3+S4 同时跑的混合负载（模拟改造后的重叠执行）

用法（在仓库根目录）:
  .venv\\Scripts\\python.exe scripts\\concurrency_bench.py --quick          # 冒烟：档位 1,4 小样本
  .venv\\Scripts\\python.exe scripts\\concurrency_bench.py                  # 完整扫描
  .venv\\Scripts\\python.exe scripts\\concurrency_bench.py --scenarios s3 --levels 1,4,8

产出：
  终端汇总表 + log/concurrency_bench-<ts>.jsonl（逐请求明细，可复盘）
成本：默认全程约 (2+16+16+16)*~700 tok + VLM 若干，< ¥0.5；--quick 约 1/4。
缓存：结果只写 log/，不写 translations.json / figures.json，业务缓存零污染。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paper_translate.clients import llm as llm_mod
from paper_translate.clients import vlm as vlm_mod
from paper_translate.clients.llm import LLMClient
from paper_translate.clients.vlm import VLMClient
from paper_translate.config import load_config
from paper_translate.s3_figures import SYSTEM_PROMPT as VLM_SYS, _prompt_for, collect_visuals
from paper_translate.s4_translate import SYSTEM_PROMPT as LLM_SYS, build_units

# ---------------------------------------------------------------- 限流计数探针
# 猴子补丁 _warn_retry：客户端每次重试（含 429 退避）都会经过这里，
# 用它统计重试/429 次数，不改业务代码即可拿到限流压力信号。
PROBE_LOCK = threading.Lock()
PROBE = {"retries": 0, "rate429": 0}


def _patched_warn_retry(orig):
    def wrapper(attempt, max_retries, wait, err):
        with PROBE_LOCK:
            PROBE["retries"] += 1
            resp = getattr(err, "response", None)
            if resp is not None and getattr(resp, "status_code", None) == 429:
                PROBE["rate429"] += 1
        orig(attempt, max_retries, wait, err)
    return wrapper


for _mod in (llm_mod, vlm_mod):
    _mod._warn_retry = _patched_warn_retry(_mod._warn_retry)


# ---------------------------------------------------------------- 滑动窗口限流器
class SlidingWindowLimiter:
    """RPM + TPM 双滑窗（60s），线程安全。

    - 请求发出前：检查两个窗口余量，不足则等待最早的条目滑出。
    - TPM 采用「先按估算预留、返回后按实际校准」：窗口内条目记 reserved，
      实际用量与预留的差额通过 extra 修正（多退少补），防止大输出打穿限额。
    """

    def __init__(self, rpm: int, tpm: int, window: float = 60.0):
        self.rpm, self.tpm, self.window = rpm, tpm, window
        self._req: deque[float] = deque()          # 请求时间戳
        self._tok: deque[tuple[float, int]] = deque()  # (时间戳, 预留token)
        self._lock = threading.Lock()

    def _evict(self, now: float):
        cutoff = now - self.window
        while self._req and self._req[0] < cutoff:
            self._req.popleft()
        while self._tok and self._tok[0][0] < cutoff:
            self._tok.popleft()

    def acquire(self, est_tokens: int) -> tuple[float, int]:
        """阻塞直到可以发出一个预留 est_tokens 的请求。返回 (入窗时间戳, id)。"""
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict(now)
                wait_req = 0.0
                if len(self._req) >= self.rpm:
                    wait_req = self._req[0] + self.window - now
                tok_sum = sum(t for _, t in self._tok)
                wait_tok = 0.0
                if tok_sum + est_tokens > self.tpm:
                    # 需要滑出的最早条目数
                    s = tok_sum
                    for ts, t in self._tok:
                        s -= t
                        if s + est_tokens <= self.tpm:
                            wait_tok = ts + self.window - now
                            break
                wait = max(wait_req, wait_tok, 0.0)
                if wait <= 0:
                    self._req.append(now)
                    ent = (now, est_tokens)
                    self._tok.append(ent)
                    return now, id(ent)
            time.sleep(min(wait, 2.0) + 0.01)

    def correct(self, entry_id: int, actual_tokens: int):
        """按实际用量校准预留条目（找不到=已滑出窗口，忽略即可）。"""
        with self._lock:
            for i, ent in enumerate(self._tok):
                if id(ent) == entry_id:
                    self._tok[i] = (ent[0], actual_tokens)
                    return

    def stats(self, since: float) -> dict:
        with self._lock:
            now = time.monotonic()
            self._evict(now)
            span = max(1e-6, now - since)
            return {
                "req_in_window": len(self._req),
                "tok_in_window": sum(t for _, t in self._tok),
                "achieved_rpm": round(len(self._req) / span * 60, 1),
                "achieved_tpm": round(sum(t for _, t in self._tok) / span * 60, 1),
            }


# ---------------------------------------------------------------- 估算
def est_prompt_tokens_llm(src: str, gloss: str, title: str) -> int:
    # 中英混合粗略估算：字符数 / 2.5，再加固定开销。宁可高估保安全。
    return int((len(src) + len(gloss) + len(title)) / 2.5) + 150


def est_prompt_tokens_vlm(img_bytes: int, caption: str) -> int:
    # 图片 base64 很大，但计费按视觉 token（Qwen3-VL 上限约 1.5k）+ 文本
    return 2000 + len(caption) // 2


# ---------------------------------------------------------------- 工作负载
def make_s4_tasks(client, cfg_api, sample, limiter):
    """从缓存论文构建翻译任务（真实段落，但不读写业务缓存）。"""
    gloss = "（术语表省略）"
    units = sample
    def run(u):
        entry_ts, eid = limiter.acquire(est_prompt_tokens_llm(u.src, gloss, "bench") + 1000)
        r = client.chat(LLM_SYS,
                        f"全文标题（已译）：基准测试\n术语表（强制采用译名）：\n{gloss}\n\n"
                        f"待翻译段落：\n{u.src}", max_tokens=1000)
        limiter.correct(eid, r["tok_in"] + r["tok_out"])
        return {"kind": "s4", "id": u.seg_id, "dur": r["duration_s"],
                "tok_in": r["tok_in"], "tok_out": r["tok_out"],
                "retries": r["retries"]}
    return units, run


def make_s3_tasks(client, cache_dir, sample, limiter):
    mj = json.loads((cache_dir / "middle_json.json").read_text(encoding="utf-8"))
    visuals = [v for v in collect_visuals(mj, cache_dir) if not v.skipped]
    visuals = visuals[:sample]

    def run(v):
        img = cache_dir / v.image_path
        est = est_prompt_tokens_vlm(img.stat().st_size, v.caption) + 2500
        _, eid = limiter.acquire(est)
        r = client.chat_image(img, VLM_SYS,
                              _prompt_for(v.kind, v.caption,
                                          v.context_before + " " + v.context_after),
                              max_tokens=2000)
        limiter.correct(eid, r["usage"]["in"] + r["usage"]["out"])
        return {"kind": "s3", "id": v.id, "dur": r["duration_s"],
                "tok_in": r["usage"]["in"], "tok_out": r["usage"]["out"],
                "retries": r["retries"]}
    return visuals, run


# ---------------------------------------------------------------- 执行器
def bench_level(name, concurrency, items, run, limiter, logf) -> dict:
    PROBE["retries"] = PROBE["rate429"] = 0
    results, errors = [], []
    t0 = time.monotonic()

    def one(x):
        s = time.perf_counter()
        try:
            rec = run(x)
        except Exception as e:  # 重试耗尽后仍失败
            rec = {"kind": name, "id": getattr(x, "seg_id", None) or getattr(x, "id", "?"),
                   "dur": round(time.perf_counter() - s, 1), "error": str(e)[:200]}
            errors.append(rec)
        rec["t_start"] = round(time.monotonic() - t0, 1)
        results.append(rec)
        logf.write(json.dumps({"bench": name, "conc": concurrency, **rec},
                              ensure_ascii=False) + "\n")
        logf.flush()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(one, items))
    wall = time.monotonic() - t0

    durs = sorted(r["dur"] for r in results if "error" not in r)
    tok = sum(r.get("tok_in", 0) + r.get("tok_out", 0) for r in results)
    lstats = limiter.stats(t0)
    out = {
        "bench": name, "concurrency": concurrency, "n": len(items),
        "wall_s": round(wall, 1),
        "ok": len(results) - len(errors), "fail": len(errors),
        "dur_avg": round(statistics.mean(durs), 1) if durs else None,
        "dur_p50": durs[len(durs) // 2] if durs else None,
        "dur_p90": durs[max(0, int(len(durs) * 0.9) - 1)] if durs else None,
        "dur_max": durs[-1] if durs else None,
        "tokens_total": tok,
        "throughput_req_min": round(len(items) / wall * 60, 1),
        "throughput_tok_min": round(tok / wall * 60),
        "retries": PROBE["retries"], "http_429": PROBE["rate429"],
        "limiter_window": lstats,
    }
    print(f"  [{name} conc={concurrency:>2}] wall={wall:7.1f}s  "
          f"ok={out['ok']}/{out['n']} fail={out['fail']}  "
          f"dur avg/p50/p90/max={out['dur_avg']}/{out['dur_p50']}/{out['dur_p90']}/{out['dur_max']}s  "
          f"吞吐={out['throughput_req_min']}req/min {out['throughput_tok_min']}tok/min  "
          f"429={out['http_429']} 重试={out['retries']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="output/_cache/a639448e61be3ab2",
                    help="提供测试数据的解析缓存目录（含 middle_json/images）")
    ap.add_argument("--levels", default="1,4,8,12", help="并发档位，逗号分隔")
    ap.add_argument("--scenarios", default="s4,s3,mixed", help="s4,s3,mixed 子集")
    ap.add_argument("--sample-s4", type=int, default=16, help="s4 每档位样本单元数")
    ap.add_argument("--sample-s3", type=int, default=5, help="s3 每档位样本图表数")
    ap.add_argument("--rpm", type=int, default=180, help="RPM 软上限（限额 200 留余量）")
    ap.add_argument("--tpm", type=int, default=1_800_000, help="TPM 软上限（限额 2M 留余量）")
    ap.add_argument("--quick", action="store_true", help="冒烟档：levels=1,4 sample=6/3 仅 s4")
    args = ap.parse_args()

    levels = [1, 4] if args.quick else [int(x) for x in args.levels.split(",")]
    scenarios = ["s4"] if args.quick else args.scenarios.split(",")
    n_s4 = 6 if args.quick else args.sample_s4
    n_s3 = 3 if args.quick else args.sample_s3

    cfg = load_config()
    api = cfg["api"]
    print(f"bench: {api['base_url']}  llm={api['llm_model']}  vlm={api['vlm_model']}")
    print(f"levels={levels}  scenarios={scenarios}  限额 rpm<={args.rpm} tpm<={args.tpm:,}")

    cache_dir = ROOT / args.cache
    mj = json.loads((cache_dir / "middle_json.json").read_text(encoding="utf-8"))

    limiter = SlidingWindowLimiter(args.rpm, args.tpm)
    llm_client = LLMClient(api["base_url"], api["api_key"], api["llm_model"],
                           proxy=api.get("proxy") or None, enable_thinking=False)
    vlm_client = VLMClient(api["base_url"], api["api_key"], api["vlm_model"],
                           proxy=api.get("proxy") or None, enable_thinking=False)

    # 准备任务池（s4 单元 / s3 图表）。每档位重复执行同一批任务以对比并发效应；
    # 服务端无翻译缓存可用（提示词含时间戳无关内容），重复执行=真实重复计费，
    # 故对 >1 的档位复用结果不做（必须真发请求才有意义），但把样本限制在小规模。
    units = [u for u in build_units(mj, []) if u.kind == "text"][:n_s4]
    if len(units) < n_s4:
        print(f"  (缓存论文 text 单元不足，实际每档 {len(units)} 个)")

    log_dir = ROOT / "log"
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"concurrency_bench-{ts}.jsonl"

    summary: list[dict] = []
    t_all = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        if "s4" in scenarios:
            print("\n== S4 翻译（LLM）并发扫描 ==")
            items, run = make_s4_tasks(llm_client, api, units, limiter)
            for c in levels:
                summary.append(bench_level("s4", c, items, run, limiter, logf))

        if "s3" in scenarios:
            print("\n== S3 图表（VLM）并发扫描 ==")
            items3, run3 = make_s3_tasks(vlm_client, cache_dir, n_s3, limiter)
            print(f"  样本: {[v.id for v in items3]}")
            for c in levels:
                if c == 1:  # VLM 单请求贵，档位>1 时只跑一半样本控制成本
                    sample = items3
                else:
                    sample = items3[:max(2, len(items3) // 2)]
                summary.append(bench_level("s3", c, sample, run3, limiter, logf))

        if "mixed" in scenarios:
            # 混合负载 = 改造后形态：S3 长请求在跑时 S4 持续吞吐。
            # 用两个独立线程池（模拟阶段重叠）+ 同一个全局限流器。
            print("\n== MIXED：S3∥S4 重叠（共享限流） ==")
            c = levels[-1] if len(levels) > 1 else 4
            _, run4 = make_s4_tasks(llm_client, api, units[:8], limiter)
            items3, run3 = make_s3_tasks(vlm_client, cache_dir, n_s3, limiter)

            PROBE["retries"] = PROBE["rate429"] = 0
            t0 = time.monotonic()
            recs = []
            with ThreadPoolExecutor(max_workers=c) as ex4, \
                 ThreadPoolExecutor(max_workers=max(1, c // 2)) as ex3:
                # 先提交 S3 长请求，再提交 S4 —— 模拟重叠执行
                futs = [ex3.submit(run3, v) for v in items3]
                futs += [ex4.submit(run4, u) for u in units[:8]]
                for f in futs:
                    try:
                        r = f.result()
                        recs.append(r)
                        logf.write(json.dumps({"bench": "mixed", "conc": c, **r},
                                              ensure_ascii=False) + "\n")
                    except Exception as e:
                        recs.append({"dur": 0, "error": str(e)[:200]})
            wall = time.monotonic() - t0
            ok = [r for r in recs if "error" not in r]
            durs = sorted(r["dur"] for r in ok)
            tok = sum(r.get("tok_in", 0) + r.get("tok_out", 0) for r in ok)
            m = {"bench": "mixed", "concurrency": c, "n": len(recs),
                 "wall_s": round(wall, 1), "ok": len(ok), "fail": len(recs) - len(ok),
                 "dur_avg": round(statistics.mean(durs), 1) if durs else None,
                 "dur_p50": durs[len(durs) // 2] if durs else None,
                 "dur_p90": durs[max(0, int(len(durs) * 0.9) - 1)] if durs else None,
                 "dur_max": durs[-1] if durs else None,
                 "tokens_total": tok,
                 "throughput_req_min": round(len(recs) / wall * 60, 1),
                 "throughput_tok_min": round(tok / wall * 60),
                 "retries": PROBE["retries"], "http_429": PROBE["rate429"],
                 "limiter_window": limiter.stats(t0)}
            print(f"  [mixed conc={c}] wall={wall:.1f}s ok={m['ok']}/{m['n']} "
                  f"dur avg/p90/max={m['dur_avg']}/{m['dur_p90']}/{m['dur_max']}s "
                  f"429={m['http_429']} 重试={m['retries']}", flush=True)
            summary.append(m)

    total_s = time.perf_counter() - t_all
    print(f"\n== 汇总（总耗时 {total_s:.0f}s，明细 {log_path}）==")
    hdr = f"{'bench':6} {'conc':>4} {'wall':>7} {'ok/fail':>8} {'p50':>6} {'p90':>6} {'max':>6} {'tok/min':>8} {'429':>4} {'retry':>5}"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        okfail = f"{s['ok']}/{s['fail']}"
        print(f"{s['bench']:6} {s['concurrency']:>4} {s['wall_s']:>7} "
              f"{okfail:>8} {str(s['dur_p50']):>6} {str(s['dur_p90']):>6} "
              f"{str(s['dur_max']):>6} {s['throughput_tok_min']:>8} {s['http_429']:>4} {s['retries']:>5}")
    (log_dir / f"concurrency_bench-{ts}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
