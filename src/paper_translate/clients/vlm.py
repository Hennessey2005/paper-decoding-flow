# -*- coding: utf-8 -*-
"""VLM / LLM 客户端（OpenAI 兼容协议：硅基流动、bigmodel 等）。

- 图片走 base64 data URL
- Qwen3 系列默认关闭 thinking（enable_thinking=False），避免长思考拖慢结构化抽取
- 指数退避重试；返回 usage 供成本统计
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx
from openai import OpenAI


def make_http_client(proxy: str | None, timeout: float) -> httpx.Client:
    """proxy: None/""=强制直连（默认，国内端点不受 VPN 开关影响）；
    "env"=跟随系统环境变量；其他值视为代理 URL（如 http://127.0.0.1:7897）。"""
    if proxy == "env":
        return httpx.Client(trust_env=True, timeout=timeout)
    if proxy:
        return httpx.Client(proxy=proxy, timeout=timeout)
    return httpx.Client(trust_env=False, timeout=timeout)


def _rate_limit_backoff(attempt: int, err: Exception) -> float:
    """429 限流退避：优先用服务端 Retry-After（封顶 60s），否则指数退避（基础 3s，429 用 5s，封顶 60s）。

    免费档模型（如智谱 GLM-4.6V/4.7-Flash）常被限流，需要更耐心地退避而非快速重试。
    """
    base = 3.0
    resp = getattr(err, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers is not None:
            ra = headers.get("Retry-After")
            if ra:
                try:
                    return min(60.0, float(ra))  # 服务端偶发天价 Retry-After，封顶防假死
                except (ValueError, TypeError):
                    pass
        if getattr(resp, "status_code", None) == 429:
            base = 5.0
    return min(60.0, base * (2 ** attempt))


def _warn_retry(attempt: int, max_retries: int, wait: float, err: Exception) -> None:
    """重试必须可见：静默等待是"长时间无输出"错觉的主要来源之一。"""
    print(f"  [retry] {type(err).__name__}: {str(err)[:90]}",
          f"-> wait {wait:.0f}s (attempt {attempt + 2}/{max_retries})", flush=True)


def _data_url(image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    suffix = image_path.suffix.lstrip(".").lower()
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{b64}"


class VLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0, max_retries: int = 3,
                 proxy: str | None = None, enable_thinking: bool = True,
                 local: bool = False):
        self.model = model
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.local = local  # Ollama 本地端点：走 options.num_ctx 控上下文，且请求间串行
        self._cli = OpenAI(api_key=api_key, base_url=base_url,
                           http_client=make_http_client(proxy, timeout))

    def chat_image(self, image_path: Path, system: str, user: str,
                   max_tokens: int = 2000, num_ctx: int | None = None) -> dict:
        """发送 单图+文本 请求。返回 {text, usage:{in,out}, duration_s, retries}。

        enable_thinking=False 时附加 extra_body={"enable_thinking": False}（关闭长思考，
        加速结构化抽取）；为 True 时不发送该字段，兼容不识别该参数的本地端点（如 Ollama）。
        local=True 时改用 Ollama 的 extra_body={"options": {...}} 传 num_ctx
        （实测 6GB 显存：table 需 4096，chart/image 用 2048 更快）。
        """
        content = [
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            {"type": "text", "text": user},
        ]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        extra: dict = {}
        if not self.enable_thinking:
            extra["enable_thinking"] = False
        if self.local:
            opts = {"temperature": 0.7}
            if num_ctx:
                opts["num_ctx"] = num_ctx
            extra["options"] = opts
        kw = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if extra:
            kw["extra_body"] = extra
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                kw = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
                if extra is not None:
                    kw["extra_body"] = extra
                r = self._cli.chat.completions.create(**kw)
                u = r.usage
                return {
                    "text": r.choices[0].message.content or "",
                    "usage": {"in": u.prompt_tokens, "out": u.completion_tokens},
                    "duration_s": round(time.perf_counter() - t0, 1),
                    "retries": attempt,
                }
            except Exception as e:  # 网络/限流 → 指数退避（尊重 Retry-After，打印等待）
                last_err = e
                wait = _rate_limit_backoff(attempt, e)
                _warn_retry(attempt, self.max_retries, wait, e)
                time.sleep(wait)
        raise RuntimeError(f"VLM 调用失败（{self.max_retries} 次重试后）: {last_err}")


def parse_model_json(text: str) -> dict:
    """从模型回复解析 JSON；容忍 ```json 围栏与尾随说明（json_repair 为 MinerU 自带依赖）。"""
    import json_repair

    obj = json_repair.loads(text)
    if isinstance(obj, dict):
        return obj
    # 兜底：截取第一个 { 到最后一个 }
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        obj = json_repair.loads(text[i:j + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"模型输出无法解析为 JSON 对象: {text[:120]}…")


def log_event(log_path: Path, event: dict) -> None:
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
