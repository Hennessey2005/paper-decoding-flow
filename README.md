# PDF:Paper Decoding Flow - An translation and parsing tool for academic papers

**PDF 学术论文 → 深度理解包**：一条命令产出「中文译文 + LaTeX 公式注册表 + 图表内容描述」。
面向中国在校学生的日常工具：一份解析，人读着不烧心，知识库嵌入也不烧心！

## 写在前面

关于任何bug和建议，欢迎给 ~~我~~ 我的AGENT 提issue。

## 功能

| 痛点 | 方案 | 产出 |
|---|---|---|
| 公式变乱码 / 被翻译毁掉 | 公式剥离为 LaTeX + 占位符保护 + 语法回验 | `formulas/` |
| 图表只是图片，无法检索 | 多模态模型生成中文描述 + 数据表 | `figures/` |
| 译文无法进知识库 | 语义块翻译 + 页码/坐标溯源层 | `译文.zh.md` + `segments.json` |
| 配环境劝退 | Windows `run.bat` 自举：自动装 Python/依赖/引导填 key，拖拽即用 | 首次双击后免维护 |

## 工作原理

```
PDF ─► S1 MinerU解析 ─► S2 公式剥离 ─► S3 图表理解 ─► S4 翻译 ─► S5 组装
        (CPU 本地)       (CPU 本地)     (VLM API)    (LLM API)   (本地)
```

版面/OCR/公式/表格识别全部 CPU 本地运行（MinerU basic 档）；仅图表理解与翻译调用 API；有本地 GPU 时可整体升级。

## 快速开始

### 已支持Windows（推荐：自动配置 + 拖拽）

```bash
git clone https://github.com/Hennessey2005/paper-decoding-flow.git
```

打开仓库目录，**把 PDF 拖到 `run.bat` 图标上**（支持一次拖多个；也可双击后把文件拖入窗口回车）。首次运行自动完成：缺 Python 则引导安装 → 建 venv 装依赖 → 记事本打开 `config\config.yaml` 引导填 key（硅基 key 对应 `default`；**强烈建议顺手填入智谱 key**，零成本解锁 `free` 模式），之后秒进。处理完成后自动打开结果目录。

### 手动 / 其他平台

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e .

copy config\config.example.yaml config\config.yaml  # 填入你的 API key（已 gitignore）
paper-translate paper.pdf
```

常用参数：

```bash
paper-translate paper.pdf --mode default|free|local|minimum  # 四种平级模式，见下表
paper-translate paper.pdf --cache-dir output/_cache/<hash>   # 复用解析缓存
paper-translate paper.pdf --open                             # 完成后打开输出目录
python -m paper_translate paper.pdf                          # 未安装命令时
```

默认适配硅基流动（OpenAI 兼容协议），模板已预填实测模型：图表理解 `Qwen/Qwen3-VL-32B-Instruct`、翻译 `deepseek-ai/DeepSeek-V4.1-Flash`，换平台只需改 `base_url`、`api_key` 和模型名（注意 VLM 需支持 `enable_thinking` 参数的密集系列多模态模型，见模板注释）。

<details>
<summary><b>配置 local 模式：需要本地 GPU 并安装 Ollama——克隆仓库后，在仓库所在目录打开你的 agent，将以下内容发送给它，一次性完成配置</b></summary>

````text
请在当前仓库完成 paper-translate 的 local 模式配置，按顺序执行，每步完成后再进入下一步：

1. 环境检查
   - 运行 nvidia-smi：确认存在 NVIDIA GPU 且可用显存 ≥ 4GB；没有则停止并告诉我本机不满足条件。
   - 检查 Ollama 是否已安装（ollama --version，或常见安装路径）。

2. 安装 Ollama（若未安装，Windows）
   - 优先：winget install -e --id Ollama.Ollama
   - 若下载缓慢（安装包走 GitHub CDN）：改下便携版 zip
     https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip
     解压后直接运行其中的 ollama.exe serve，无需安装；
     本机访问 GitHub 慢时可为 curl 加代理。

3. 启动服务
   - 在仓库外的模型目录启动：OLLAMA_MODELS=<模型目录> ollama serve（后台运行）
   - curl http://localhost:11434/api/version 应返回版本号。

4. 下载翻译模型（国内走 ModelScope，约 2.4GB；Q4_K_M 量化，实测显存增量约 2.8GB）
   - curl -L -o <模型目录>/gemma-3-4b-it-Q4_K_M.gguf
     "https://modelscope.cn/models/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"

5. 导入模型（名字必须是 gemma4b，与 config/config.yaml 的 local.llm_model 一致）
   - 优先：写一个 Modelfile（内容两行：FROM ./gemma-3-4b-it-Q4_K_M.gguf 和 PARAMETER num_ctx 4096），
     然后 ollama create gemma4b -f Modelfile
   - 若 ollama create 报 "invalid model name"（v0.40.10 已知 bug），改用 HTTP API 导入：
     a) 计算 gguf 的 sha256，POST http://localhost:11434/api/blobs/sha256:<digest>，body 为文件本体；
     b) POST http://localhost:11434/api/create，JSON body：
        {"model":"gemma4b","files":{"gemma-3-4b-it-Q4_K_M.gguf":"sha256:<完整digest>"},"stream":false}
     注意：本机若设置了 http_proxy/https_proxy 环境变量，大文件上传会被代理打断，需绕过代理直连 127.0.0.1。

6. 验证
   - GET http://localhost:11434/api/tags 应列出 gemma4b；
   - 用 OpenAI 兼容端点 http://localhost:11434/v1（api_key 任意占位）发一条中英翻译测试，
     确认输出正常，且 nvidia-smi 显存增量 ≤ 4GB。

7. （可选，解锁本地图表理解 S3，local 模式产物升级为全套）导入本地多模态模型：
   - 以 MiniCPM-V 2.6 为例，官方 GGUF 走 ModelScope（OpenBMB/MiniCPM-V-2_6-gguf），
     下载两个文件（缺一不可）：ggml-model-Q4_K_M.gguf（约 4.4GB）
     与 mmproj-model-f16.gguf（约 1GB，视觉投影）。
   - 按第 5 步的 blobs/create API 方式导入，/api/create 的 files 映射同时列出这两个文件
     （Ollama 按文件名识别 mmproj 自动挂视觉塔），模型名如 minicpmv26，
     并填入 config/config.yaml 的 local.vlm_model。
   - 可根据自己的显存大小选择更强的模型以提升性能
     （实测 RTX 4050 Laptop 6GB 用 MiniCPM-V 2.6 q4_K_M 可跑，属极限操作：
     ~15 tok/s，单图 10-40s，图表理解与翻译串行执行，跑完自动卸载）。

8. 汇报：全部通过后告诉我 local 模式已就绪——在 run.bat 中选 3) local，
   或 python -m paper_translate paper.pdf --mode local。
````

</details>

## 四种运行模式

<details>
<summary><b>点击展开：default / free / local / minimum 模式对比（自动配置、配置需求、成本、产物）</b></summary>

| 模式 | 名称 | 是否自动配置 | 配置需求 | 成本 | 产物 |
|---|---|---|---|---|---|
| `default`（推荐） | 硅基流动云端：Qwen3-VL 图表理解 + DeepSeek 翻译 | ✅ 首次引导填 key 即可用 | `api.api_key`（[硅基流动](https://siliconflow.cn)） | 按量计费，会议 < ¥1/篇，效果最好 | 全套：`译文.zh.md` / `译文.bilingual.md` / `formulas/` / `figures/`（含 VLM 中文描述与数据表）/ `segments.json` |
| `free` | 智谱免费云端：GLM-4.6V-Flash + GLM-4.7-Flash | ✅ 首次引导填 key 即可用 | `api.zhipu_api_key`（[open.bigmodel.cn](https://open.bigmodel.cn) 注册即得） | 0 元（免费档；偶发限流） | 与 `default` 相同的全套产物 |
| `local` | Ollama 本地模型：翻译 + 图表理解（可选），全离线 | ❌ 手动配置（agent轻松代劳） | 安装 [Ollama](https://ollama.com) + 导入翻译模型（推荐 `gemma3-4b-it` Q4_K_M）；可选导入多模态模型解锁图表理解 | 0 元（仅电费） | 全套（同 `default`；本地 VLM 描述数值类建议对照原图核验） |
| `minimum` | 纯解析（完全不调 API） | ✅ 零配置，开箱即用 | 无 | 0 元 | 解析产物：`译文.zh.md`（保留原文，未翻译）/ `formulas/` / `figures/`（裁剪图）/ `segments.json` |

</details>

所有模式共享 S1 本地解析（MinerU，按 PDF 内容哈希缓存，重跑零成本）。

## 输出结构

```
output/<论文标题>/
├── 译文.zh.md          # 中文译文（公式 LaTeX 内嵌、图表原位描述）
├── 译文.bilingual.md   # 中英对照
├── formulas/           # 公式注册表（编号/页码/LaTeX/回验状态）
├── figures/            # 图表理解（图注译文/VLM 描述/数据表）
├── segments.json       # 溯源层：译文段 ↔ PDF 页码/坐标
└── _cache/             # 缓存（断点续跑不重复计费）
```

输出目录自动以 **PDF 内的论文真实标题**命名（取自解析结果，纯本地；标题非法字符自动清洗、超 80 字符截断；同名论文以内容哈希后缀区分，同篇重跑复用目录），也可用 `-o` 指定。

> 📖 **阅读建议**：译文是含 LaTeX 公式的纯 Markdown，推荐用轻量阅读器 [mdview](https://github.com/mdview2026/mdview) 打开——双击 `.md` 即渲染阅读；社区版免费，**Pro 为付费**（约 ¥10 一次性买断、可离线），解锁 KaTeX 数学渲染与 Mermaid，公式密集的译文建议用 Pro。

## 效果展示（mdview 渲染 `译文.zh.md`）

公式处理：行内公式与展示公式均以 Markdown 数学语法保留原始 LaTeX，KaTeX 渲染，左侧目录可跳转：

![公式处理结果](formulas.png)

图表处理：图注译文原位引用 + 裁剪原图 + VLM 生成的中文内容描述：

![图表处理结果](pictures.png)

## 实测指标（Attention Is All You Need · arXiv 1706.03762v7 · 15 页双栏 · 纯 CPU）

冷启动全流程约 **13 分钟 / < ¥1**：S1 解析 223.5s（本地免费）+ S2–S5 约 530s；图表理解 9 项（5 表 + 4 图）约 12.7k tokens，翻译 103 单元约 40.7k tokens，API 合计约 53k tokens；显示公式语法回验 5/5，端到端自检 4/4 PASS；解析/翻译缓存命中后重跑 0 次 API 调用（约 7 min 纯本地重算）。

## 项目结构

```
run.bat                # Windows 拖拽入口（首跑自动配置环境）
src/paper_translate/   # cli / orchestrator / S2-S5 阶段模块 / title_naming / VLM+LLM 客户端
scripts/               # M1-M4 阶段回归测试 + 标题命名回归 + 模型选型对比
config/                # 配置模板 + 术语表（可积累）
```

## 相关链接

- [MinerU](https://github.com/opendatalab/MinerU)：本项目的本地文档解析引擎（版面 / OCR / 公式 / 表格识别）
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)：致敬其「直读 PDF 文本层」的轻量思路。并非所有论文都需要完整解析管线——若只需快速文本提取或轻量级翻译预处理，PyMuPDF 直读文本层是成本最低的替代路线
- [mdview](https://github.com/mdview2026/mdview)（[官网](https://www.mdview.top/)）：推荐的译文阅读器。双击即渲染的轻量 Markdown 查看器（约 2MB、跨平台），社区版永久免费，Pro 版付费买断（约 ¥10，可离线）增加 KaTeX 数学、Mermaid 与编辑模式；本项目产出含大量内嵌 LaTeX 公式，配合支持 KaTeX 的 Pro 版阅读体验最佳
- [硅基流动](https://siliconflow.cn)：VLM 图表理解与 LLM 翻译的 API 服务

## LICENSE

 MIT
