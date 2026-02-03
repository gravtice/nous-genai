---
name: nous-genai-skill
description: >
  Use nous-genai as an end user (not a contributor): run `genai` CLI for text/image/audio/video/embedding
  across providers (OpenAI/Gemini/Claude/DashScope/Doubao/Tuzi), save binary outputs, list available models, and start a
  local MCP server (Streamable HTTP/SSE) with auth. Use when setting up `.env.local` provider keys, choosing
  `{provider}:{model_id}`, or debugging common issues (auth/timeout/SSRF-download/MCP bearer-token rules).
  中文: 用 nous-genai 调用多家大模型 + 启动 MCP 服务 + 排错。
---

# nous-genai（用户版）

## Quick Start

**IMPORTANT:** Commands must run from this skill's base directory to load `.env.local` config.

```bash
# 1) Create `.env.local` in this skill directory
(cd "<SKILL_BASE_DIR>" && test -f .env.local || cp .env.example .env.local)

# 2) Edit `<SKILL_BASE_DIR>/.env.local` and set at least one provider key.
# Example (OpenAI):
#   NOUS_GENAI_OPENAI_API_KEY=...

# 3) Text
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai --model openai:gpt-4o-mini --prompt "Hello")

# 4) See what you can use (requires at least one provider key configured)
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai model available --all)
```

If `uvx` is unavailable, install once and use `genai` directly:

```bash
python -m pip install --upgrade nous-genai
(cd "<SKILL_BASE_DIR>" && genai --model openai:gpt-4o-mini --prompt "Hello")
```

## Configuration (Zero-parameter)

Put all config in `<SKILL_BASE_DIR>/.env.local` (copy from `<SKILL_BASE_DIR>/.env.example`).

Env file load priority (high → low):

- `.env.local > .env.production > .env.development > .env.test`

Process env vars override `.env.*` (SDK uses `os.environ.setdefault()`).

Minimal `.env.local` (OpenAI text only):

```bash
NOUS_GENAI_OPENAI_API_KEY=...
NOUS_GENAI_TIMEOUT_MS=120000
```

Notes:

- Do not commit `.env.local` (add it to `.gitignore` if needed).
- Provider keys also accept non-prefixed vars like `OPENAI_API_KEY`, but prefer `NOUS_GENAI_*` for clarity.

Common keys:

- OpenAI: `NOUS_GENAI_OPENAI_API_KEY` (or `OPENAI_API_KEY`)
- Google (Gemini): `NOUS_GENAI_GOOGLE_API_KEY` (or `GOOGLE_API_KEY`)
- Anthropic (Claude): `NOUS_GENAI_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`)
- Aliyun (DashScope/百炼): `NOUS_GENAI_ALIYUN_API_KEY` (or `ALIYUN_API_KEY`)
- Volcengine (Ark/豆包): `NOUS_GENAI_VOLCENGINE_API_KEY` (or `VOLCENGINE_API_KEY`)
- Tuzi: `NOUS_GENAI_TUZI_*_API_KEY`

## Model Format

Model string is `{provider}:{model_id}` (example: `openai:gpt-4o-mini`).

Use this to pick a model by output modality:

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai model available --all)
# Look for: out=text / out=image / out=audio / out=video / out=embedding
```

If you have not configured any keys yet, you can still view the SDK curated list:

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai model sdk)
```

## Common Scenarios

### Image understanding

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai --model openai:gpt-4o-mini --prompt "Describe this image" --image-path "/path/to/image.png")
```

### Image generation (save to file)

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai --model openai:gpt-image-1 --prompt "A red square, minimal" --output-path "/tmp/out.png")
```

### Speech-to-text (transcription)

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai --model openai:whisper-1 --audio-path "/path/to/audio.wav")
```

### Text-to-speech (save to file)

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai --model openai:tts-1 --prompt "你好" --output-path "/tmp/tts.mp3")
```

## Python SDK（集成到你的项目）

Install:

```bash
python -m pip install --upgrade nous-genai
```

Minimal example:

```python
from nous.genai import Client, GenerateRequest, Message, OutputSpec, Part

client = Client()
resp = client.generate(
    GenerateRequest(
        model="openai:gpt-4o-mini",
        input=[Message(role="user", content=[Part.from_text("Hello")])],
        output=OutputSpec(modalities=["text"]),
    )
)
print(resp.output[0].content[0].text)
```

Note: `Client()` loads `.env.*` from the current working directory; run your script in the directory that contains
`.env.local`, or export env vars in the process environment.

## MCP Server (给其它工具/LLM 调用)

Start server (Streamable HTTP: `/mcp`, SSE: `/sse`):

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai-mcp-server)
```

Recommended: set auth in `.env.local` before exposing the server:

```bash
# NOUS_GENAI_MCP_BEARER_TOKEN=sk-...
```

Debug with MCP CLI:

```bash
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai-mcp-cli env)
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai-mcp-cli tools)
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai-mcp-cli call --name list_providers)
(cd "<SKILL_BASE_DIR>" && uvx --from nous-genai genai-mcp-cli call --name generate --args '{"model":"openai:gpt-4o-mini","input":"Hello","output":{"modalities":["text"]}}')
```

## Troubleshooting

### Missing/invalid API key (401/403)
Set provider credentials in `<SKILL_BASE_DIR>/.env.local` (copy from `<SKILL_BASE_DIR>/.env.example`), then retry.

### File input errors (mime type)
If you see `cannot detect ... mime type`, verify the path exists and is a valid image/audio/video file.

### Timeout / long-running jobs
Increase `NOUS_GENAI_TIMEOUT_MS` in `.env.local` and retry.

### URL download blocked / SSRF protection
Binary outputs may be returned as URLs. Private/loopback URLs are rejected by default. Only if you understand the risk, set `NOUS_GENAI_ALLOW_PRIVATE_URLS=1`.

### MCP auth (401 Unauthorized)
Set `NOUS_GENAI_MCP_BEARER_TOKEN` (or `NOUS_GENAI_MCP_TOKEN_RULES`) in `.env.local`, and ensure `genai-mcp-cli` uses the same token.
