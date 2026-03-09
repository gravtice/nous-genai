# nous-genai

![CI](https://github.com/gravtice/nous-genai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-≥3.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

English README: `README.md`

一个接口实现所有模态模型的调用，四种使用方式：Skill MCP CLI SDK

## Features

- **多 Provider 支持**：OpenAI、Google (Gemini)、Anthropic (Claude)、阿里云 (DashScope/百炼)、火山引擎 (豆包)、Tuzi
- **多模态能力**：文本、图片、音频、视频的输入/输出与理解/生成
- **统一 API**：`Client.generate()` 一个接口覆盖所有能力
- **流式输出**：`generate_stream()` 支持实时响应
- **Tool Calling**：函数调用支持
- **JSON Schema 输出**：结构化输出支持
- **MCP Server**：Streamable HTTP 和 SSE 传输协议
- **安全设计**：SSRF 防护、DNS pinning、URL 下载限制、Bearer Token 认证

## 安装

```bash
# 从 PyPI 安装
pip install nous-genai

# 从源码安装（开发）
pip install -e .

# 或使用 uv（开发，推荐）
uv sync
```

## 配置（零参数）

SDK/CLI/MCP 启动时会自动加载环境文件，优先级（高 → 低）：

`.env.local > .env.production > .env.development > .env.test`

覆盖规则：进程环境变量优先于 `.env.*`（因为加载使用 `os.environ.setdefault()`）。

最小 `.env.local` 示例（只用 OpenAI）：

```bash
NOUS_GENAI_OPENAI_API_KEY=...
NOUS_GENAI_TIMEOUT_MS=120000
```

完整配置项见 `docs/CONFIGURATION.md`，也可以直接复制 `.env.example` 为 `.env.local` 后按需填写。

## 快速开始

### CLI（最快上手，统一 API，Agent 友好）

```bash
# 先看可用模型及能力（out=text/image/audio/video/embedding）
uv run genai model available --all

# 文本生成
uv run genai --model openai:gpt-4o-mini --prompt "你好"

# 图片理解（image -> text）
uv run genai --model openai:gpt-4o-mini --prompt "描述这张图" --image-path ./examples/demo_image.png

# 图片生成（text -> image file）
uv run genai --model openai:gpt-image-1 --prompt "白底、极简风格的红色立方体" --output-path ./out.png

# 语音转写（audio -> text）
uv run genai --model openai:whisper-1 --audio-path ./examples/demo_tts.mp3

# 语音合成（text -> audio file）
uv run genai --model openai:tts-1 --prompt "你好，这里是 nous genai" --output-path ./out.mp3

# 视频生成（text -> video，异步方式）
uv run genai --model openai:sora-2 --prompt "雨后水洼上一只纸船缓缓前进，电影感" --no-wait
# ...稍后
uv run genai --model openai:sora-2 --job-id "<job_id>" --output-path ./out.mp4 --timeout-ms 600000
```

### SDK：文本生成

```python
from nous.genai import Client, GenerateRequest, Message, OutputSpec, Part

client = Client()
resp = client.generate(
    GenerateRequest(
        model="openai:gpt-4o-mini",
        input=[Message(role="user", content=[Part.from_text("你好")])],
        output=OutputSpec(modalities=["text"]),
    )
)
print(resp.output[0].content[0].text)
```

### SDK：流式输出

```python
import sys
from nous.genai import Client, GenerateRequest, Message, OutputSpec, Part

client = Client()
req = GenerateRequest(
    model="openai:gpt-4o-mini",
    input=[Message(role="user", content=[Part.from_text("讲个冷笑话")])],
    output=OutputSpec(modalities=["text"]),
)
for ev in client.generate_stream(req):
    if ev.type == "output.text.delta":
        sys.stdout.write(str(ev.data.get("delta", "")))
        sys.stdout.flush()
print()
```

### SDK：图片理解

```python
from nous.genai import Client, GenerateRequest, Message, OutputSpec, Part, PartSourcePath
from nous.genai.types import detect_mime_type

path = "./cat.png"
mime = detect_mime_type(path) or "application/octet-stream"

client = Client()
resp = client.generate(
    GenerateRequest(
        model="openai:gpt-4o-mini",
        input=[
            Message(
                role="user",
                content=[
                    Part.from_text("描述这张图"),
                    Part(type="image", mime_type=mime, source=PartSourcePath(path=path)),
                ],
            )
        ],
        output=OutputSpec(modalities=["text"]),
    )
)
print(resp.output[0].content[0].text)
```

### SDK：列出可用模型

```python
from nous.genai import Client

client = Client()
print(client.list_all_available_models())
```

## 支持的 Provider

| Provider | 说明 |
|----------|------|
| `openai` | GPT-4、DALL-E、Whisper、TTS |
| `google` | Gemini、Imagen、Veo |
| `anthropic` | Claude |
| `aliyun` | DashScope / 百炼 (OpenAI-compatible + AIGC) |
| `volcengine` | 火山引擎 Ark / 豆包 (OpenAI-compatible) |
| `tuzi-web` / `tuzi-openai` / `tuzi-google` / `tuzi-anthropic` | Tuzi 多协议适配 |

## 二进制输出处理

图片/音频/视频的 `Part.source` 是一个"引用/载荷"联合类型：

- **输入**：支持 `bytes/path/base64/url/ref` 表达（MCP 模式下禁止 `bytes/path`）
- **输出**：返回 `url` / `base64` / `ref`（SDK 不自动下载或落盘）

需要写到文件时，参考 `examples/demo.py` 的 `_write_binary()`，通过 `Client.download_to_file()` 复用 SDK 内置的安全下载逻辑。

## CLI 与 MCP Server

```bash
# CLI
uv run genai --model openai:gpt-4o-mini --prompt "你好"
uv run genai model available --all

# Tuzi Chirp 音乐
uv run genai --model tuzi-web:chirp-v3-5 --prompt "lofi hiphop beat, 30s" --no-wait
# ...稍后
uv run genai --model tuzi-web:chirp-v3-5 --job-id "<job_id>" --output-path demo_suno.mp3 --timeout-ms 600000

# MCP Server
uv run genai-mcp-server                    # Streamable HTTP: /mcp, SSE: /sse
uv run genai-mcp-cli tools                 # CLI 测试
```

## 安全特性

- **SSRF 防护**：默认拒绝私网/loopback URL（`NOUS_GENAI_ALLOW_PRIVATE_URLS=1` 可放开）
- **DNS Pinning**：防止 DNS rebinding 攻击
- **下载限制**：单次下载上限 128MiB（`NOUS_GENAI_URL_DOWNLOAD_MAX_BYTES`）
- **Bearer Token 认证**：MCP Server 支持 Token 认证
- **Token Rules**：细粒度访问控制

## 测试

```bash
uv run pytest tests/ -v
```

## 文档

- [配置说明](docs/CONFIGURATION.md)
- [架构设计](docs/ARCHITECTURE_DESIGN.md)
- [MCP Server 安全审查](docs/MCP_SERVER_CLI_SECURITY_REVIEW.md)
- [贡献指南](CONTRIBUTING.md)
- [更新日志](CHANGELOG.md)

## License

[Apache-2.0](LICENSE)
