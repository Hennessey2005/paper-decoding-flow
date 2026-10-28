# -*- coding: utf-8 -*-
"""滑动窗口限流器（RPM + TPM 双限，60s 窗口，线程安全）。

turbo 模式的核心护栏：S3（并发 6）与 S4（并发 8）重叠执行时，
两个阶段的请求经过同一个 limiter，总请求数不会超供应商限额
（硅基流动 200 RPM / 2,000,000 TPM；默认软限留 10% 余量）。

用法（已在 scripts/concurrency_bench.py 实测验证：conc 12 重叠 0×429）：
    limiter = SlidingWindowLimiter(rpm=180, tpm=1_800_000)
    entry = limiter.acquire(est_tokens)     # 请求前阻塞式预留
    ...发请求...
    limiter.correct(entry, actual_tokens)   # 返回后按实际用量校准（多退少补）
"""
from __future__ import annotations

import threading
import time
from collections import deque


class TokenEntry:
    """TPM 窗口内的一条预留记录（以对象身份索引，防 id 复用歧义）。"""
    __slots__ = ("ts", "tokens")

    def __init__(self, ts: float, tokens: int):
        self.ts = ts
        self.tokens = tokens


class SlidingWindowLimiter:
    def __init__(self, rpm: int = 180, tpm: int = 1_800_000, window: float = 60.0):
        self.rpm, self.tpm, self.window = rpm, tpm, window
        self.created = time.monotonic()
        self._req: deque[float] = deque()
        self._tok: deque[TokenEntry] = deque()
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._req and self._req[0] < cutoff:
            self._req.popleft()
        while self._tok and self._tok[0].ts < cutoff:
            self._tok.popleft()

    def acquire(self, est_tokens: int) -> TokenEntry | None:
        """阻塞直到可发出一个预留 est_tokens 的请求；返回窗口内条目（供 correct 校准）。

        est_tokens 是「prompt 估算 + max_tokens 预留」；宁可高估，返回后校准收回余量。
        """
        est_tokens = max(1, int(est_tokens))
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict(now)
                wait_req = 0.0
                if len(self._req) >= self.rpm:
                    wait_req = self._req[0] + self.window - now
                wait_tok = 0.0
                tok_sum = sum(e.tokens for e in self._tok)
                if tok_sum + est_tokens > self.tpm:
                    s = tok_sum
                    for e in self._tok:
                        s -= e.tokens
                        if s + est_tokens <= self.tpm:
                            wait_tok = e.ts + self.window - now
                            break
                    else:  # 单条即超限：等整窗滑出
                        wait_tok = self.window
                wait = max(wait_req, wait_tok, 0.0)
                if wait <= 0:
                    self._req.append(now)
                    ent = TokenEntry(now, est_tokens)
                    self._tok.append(ent)
                    return ent
            time.sleep(min(wait, 2.0) + 0.01)

    def correct(self, entry: TokenEntry | None, actual_tokens: int) -> None:
        """按实际用量校准预留（条目已滑出窗口则忽略）。"""
        if entry is None:
            return
        with self._lock:
            entry.tokens = max(0, int(actual_tokens))

    def stats(self, since: float) -> dict:
        """since 参数保留兼容；护栏语义看 req_in_window（最近 60s 窗口内请求数，
        恒 <= rpm 软限），achieved_rpm 只是全程平均外推、短样本下会虚高。"""
        with self._lock:
            now = time.monotonic()
            self._evict(now)
            span = max(1e-6, now - self.created)
            return {"req_in_window": len(self._req),
                    "tok_in_window": sum(e.tokens for e in self._tok),
                    "avg_rpm": round(len(self._req) / span * 60, 1)}
