from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass, replace
from typing import Any, Iterator
from uuid import uuid4

from .._internal.errors import (
    GenAIError,
    invalid_request_error,
    not_supported_error,
    provider_error,
)
from .._internal.http import request_json
from ..types import (
    Capability,
    GenerateEvent,
    GenerateRequest,
    GenerateResponse,
    JobInfo,
    Message,
    Part,
    PartSourceUrl,
)
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter
from .openai import OpenAIAdapter

_ASYNCDATA_BASE_URL = "https://asyncdata.net"
_ASYNCDATA_PRO_BASE_URL = "https://pro.asyncdata.net"

_DEEPSEARCH_MODELS = frozenset(
    {
        "gemini-2.5-flash-deepsearch",
        "gemini-2.5-flash-deepsearch-async",
        "gemini-2.5-pro-deepsearch",
        "gemini-2.5-pro-deepsearch-async",
        "gemini-3-pro-deepsearch",
        "gemini-3-pro-deepsearch-async",
    }
)


def _is_deepsearch_model(model_id: str) -> bool:
    return model_id.lower().strip() in _DEEPSEARCH_MODELS


_MP4_URL_RE = re.compile(r"https?://[^\s\"'<>]+?\.mp4(?:\?[^\s\"'<>]*)?", re.IGNORECASE)
_AUDIO_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+?\.(?:mp3|wav|m4a|aac|flac|ogg|opus)(?:\?[^\s\"'<>]*)?",
    re.IGNORECASE,
)


def _extract_first_url(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    if m is None:
        return None
    return m.group(0)


def _closest_kling_duration(duration_sec: int | None) -> str:
    if duration_sec is None:
        return "5"
    try:
        sec = int(duration_sec)
    except Exception:
        return "5"
    return "5" if sec <= 5 else "10"


def _sora_api_model_and_prompt_suffix(model_id: str) -> tuple[str, str | None]:
    mid = model_id.strip()
    mid_l = mid.lower()
    if mid_l in {"sora-2", "sora-2-pro", "sora-2-character", "sora-2-pro-character"}:
        return mid, None
    if mid_l.startswith("sora-") and ":" in mid_l:
        parts = mid_l.split("-")
        ratio = parts[1] if len(parts) > 1 else ""
        res = parts[2] if len(parts) > 2 else ""
        dur = parts[3] if len(parts) > 3 else ""
        api_model = "sora-2-pro" if "720p" in res else "sora-2"
        suffix_parts: list[str] = []
        if ratio:
            suffix_parts.append(ratio)
        if res:
            suffix_parts.append(res)
        if dur:
            suffix_parts.append(dur if dur.endswith("s") else f"{dur}s")
        return api_model, " ".join(suffix_parts) if suffix_parts else None
    return mid, None


@dataclass(frozen=True, slots=True)
class TuziAdapter:
    """
    Tuzi exposes multiple protocols (OpenAI-compatible, Gemini v1beta, Anthropic /v1/messages)
    under a single API key. Route by model_id.

    For deepsearch models, uses the asyncdata.net async API.
    """

    openai: OpenAIAdapter | None
    gemini: GeminiAdapter | None
    anthropic: AnthropicAdapter | None
    proxy_url: str | None = None

    def capabilities(self, model_id: str) -> Capability:
        mid_l = model_id.lower().strip()
        if mid_l.startswith(("suno-", "suno_")):
            raise invalid_request_error(
                "suno model ids are not supported; use chirp-* (e.g. chirp-v3-5)"
            )
        if mid_l in {"kling_image", "seededit"}:
            return Capability(
                input_modalities={"text", "image"},
                output_modalities={"image"},
                supports_stream=False,
                supports_job=(mid_l == "kling_image"),
                supports_tools=False,
                supports_json_schema=False,
            )
        if mid_l.startswith("chirp-") and mid_l != "chirp-v3":
            return Capability(
                input_modalities={"text"},
                output_modalities={"audio"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        if _is_deepsearch_model(model_id):
            return Capability(
                input_modalities={"text"},
                output_modalities={"text"},
                supports_stream=False,
                supports_job=True,
                supports_tools=False,
                supports_json_schema=False,
            )
        return self._route(model_id).capabilities(model_id)

    def generate(
        self, request: GenerateRequest, *, stream: bool
    ) -> GenerateResponse | Iterator[GenerateEvent]:
        model_id = request.model_id()
        mid_l = model_id.lower().strip()
        modalities = set(request.output.modalities)

        if mid_l.startswith(("suno-", "suno_")):
            raise invalid_request_error(
                "suno model ids are not supported; use chirp-* (e.g. chirp-v3-5)"
            )

        if modalities == {"video"} and mid_l.startswith("pika-"):
            raise not_supported_error(
                "tuzi pika endpoints are not available on api.tu-zi.com (returns HTML)"
            )

        if modalities == {"video"} and "seedance" in mid_l:
            raise not_supported_error(
                "doubao seedance video is not supported on tuzi-web (upstream returns multipart: NextPart: EOF)"
            )

        if modalities == {"video"} and mid_l.startswith("kling"):
            if stream:
                raise invalid_request_error(
                    "kling video generation does not support streaming"
                )
            return self._kling_text2video(request, model_id=model_id)

        if modalities == {"video"} and mid_l.startswith("sora-"):
            if stream:
                raise invalid_request_error(
                    "sora video generation does not support streaming"
                )
            return self._async_chat_video(request, model_id=model_id)

        if modalities == {"video"} and mid_l.startswith("runway-"):
            raise not_supported_error(
                "tuzi runway endpoints are not available on api.tu-zi.com (returns HTML)"
            )

        if modalities == {"image"} and mid_l in {"kling_image", "seededit"}:
            if stream:
                raise invalid_request_error(
                    f"{mid_l} image generation does not support streaming"
                )
            if mid_l == "kling_image" and self._has_image_input(request):
                return self._route(model_id).generate(request, stream=False)
            if mid_l == "kling_image":
                return self._kling_text2image(request, model_id=model_id)
            return self._seededit(request, model_id=model_id)

        if modalities == {"audio"} and mid_l.startswith("chirp-") and mid_l != "chirp-v3":
            if stream:
                raise invalid_request_error(
                    "chirp music generation does not support streaming"
                )
            return self._suno_music(request, model_id=model_id)

        if _is_deepsearch_model(model_id):
            if stream:
                raise invalid_request_error(
                    "deepsearch models do not support streaming; use stream=False"
                )
            return self._deepsearch(request, model_id=model_id)
        return self._route(model_id).generate(request, stream=stream)

    def _has_image_input(self, request: GenerateRequest) -> bool:
        for msg in request.input:
            for part in msg.content:
                if part.type == "image":
                    return True
        return False

    def _base_host(self) -> str:
        if self.gemini is not None and self.gemini.base_url:
            return self.gemini.base_url.rstrip("/")
        if self.anthropic is not None and self.anthropic.base_url:
            return self.anthropic.base_url.rstrip("/")
        if self.openai is not None and self.openai.base_url:
            base = self.openai.base_url.rstrip("/")
            if base.endswith("/v1"):
                return base[:-3]
            return base
        raise invalid_request_error("tuzi base url not configured")

    def _bearer_headers(self) -> dict[str, str]:
        if self.openai is not None and self.openai.api_key:
            return {"Authorization": f"Bearer {self.openai.api_key}"}
        if self.gemini is not None and self.gemini.api_key:
            return {"Authorization": f"Bearer {self.gemini.api_key}"}
        if self.anthropic is not None and self.anthropic.api_key:
            return {"Authorization": f"Bearer {self.anthropic.api_key}"}
        raise invalid_request_error("tuzi api key not configured")

    def _single_text_prompt(self, request: GenerateRequest) -> str:
        texts: list[str] = []
        for msg in request.input:
            for part in msg.content:
                if part.type != "text":
                    continue
                t = part.require_text().strip()
                if t:
                    texts.append(t)
        if len(texts) != 1:
            raise invalid_request_error("this operation requires exactly one text part")
        return texts[0]

    def _text_prompt_or_none(self, request: GenerateRequest) -> str | None:
        chunks: list[str] = []
        for msg in request.input:
            for part in msg.content:
                if part.type != "text":
                    continue
                t = part.require_text().strip()
                if t:
                    chunks.append(t)
        if not chunks:
            return None
        return "\n".join(chunks).strip()

    def _seededit(self, request: GenerateRequest, *, model_id: str) -> GenerateResponse:
        if self.openai is None:
            raise invalid_request_error("tuzi openai adapter not configured")
        if not self._has_image_input(request):
            raise invalid_request_error("seededit requires image input")
        req = replace(request, model="tuzi-web:api-images-seededit")
        resp = self.openai.generate(req, stream=False)
        assert isinstance(resp, GenerateResponse)
        return replace(resp, model=f"tuzi-web:{model_id}")

    def _kling_text2image(
        self, request: GenerateRequest, *, model_id: str
    ) -> GenerateResponse:
        prompt = self._single_text_prompt(request)
        host = self._base_host()
        body: dict[str, object] = {
            "prompt": prompt,
            "negative_prompt": "",
            "aspect_ratio": "1:1",
            "callback_url": "",
        }
        opts = request.provider_options.get("tuzi-web")
        if isinstance(opts, dict):
            for k, v in opts.items():
                if k in body:
                    raise invalid_request_error(f"provider_options cannot override {k}")
                body[k] = v

        obj = request_json(
            method="POST",
            url=f"{host}/kling/v1/images/text2image",
            headers=self._bearer_headers(),
            json_body=body,
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("kling submit missing task_id")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        poll_url = f"{host}/kling/v1/images/text2image/{task_id}"
        budget_ms = (
            120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        )
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            obj = request_json(
                method="GET",
                url=poll_url,
                headers=self._bearer_headers(),
                json_body=None,
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )
            data = obj.get("data")
            if not isinstance(data, dict):
                time.sleep(1.0)
                continue
            status = data.get("task_status")
            if status == "failed":
                raise provider_error(
                    f"kling task failed: {data.get('task_status_msg')}"
                )
            if status == "succeed":
                task_result = data.get("task_result")
                if isinstance(task_result, dict):
                    images = task_result.get("images")
                    if isinstance(images, list) and images:
                        first = images[0]
                        if isinstance(first, str) and first:
                            part = Part(type="image", source=PartSourceUrl(url=first))
                            return GenerateResponse(
                                id=f"sdk_{uuid4().hex}",
                                provider="tuzi-web",
                                model=f"tuzi-web:{model_id}",
                                status="completed",
                                output=[Message(role="assistant", content=[part])],
                            )
                        if isinstance(first, dict):
                            u = first.get("url")
                            if isinstance(u, str) and u:
                                part = Part(type="image", source=PartSourceUrl(url=u))
                                return GenerateResponse(
                                    id=f"sdk_{uuid4().hex}",
                                    provider="tuzi-web",
                                    model=f"tuzi-web:{model_id}",
                                    status="completed",
                                    output=[Message(role="assistant", content=[part])],
                                )
                raise provider_error("kling task succeeded but missing image url")
            time.sleep(min(1.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(job_id=task_id, poll_after_ms=1_000),
        )

    def _async_chat_video(
        self, request: GenerateRequest, *, model_id: str
    ) -> GenerateResponse:
        if self.openai is None:
            raise invalid_request_error(
                "NOUS_GENAI_TUZI_OPENAI_API_KEY required for async chat video models"
            )

        api_model, suffix = _sora_api_model_and_prompt_suffix(model_id)
        messages = []
        for msg in request.input:
            role = msg.role if msg.role in {"system", "assistant"} else "user"
            text = "".join(
                p.require_text() for p in msg.content if p.type == "text"
            ).strip()
            if not text:
                continue
            if suffix and role == "user":
                text = f"{text} {suffix}".strip()
                suffix = None
            messages.append({"role": role, "content": text})
        if not messages:
            raise invalid_request_error(
                "video generation requires at least one text message"
            )

        original_url = f"{self.openai.base_url}/chat/completions"
        submit_url = f"{_ASYNCDATA_BASE_URL}/tran/{original_url}"
        obj = request_json(
            method="POST",
            url=submit_url,
            headers=self._bearer_headers(),
            json_body={"model": api_model, "messages": messages},
            timeout_ms=max(request.params.timeout_ms or 120_000, 120_000),
            proxy_url=self.proxy_url,
        )

        task_id = obj.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("async video submit missing id")
        source_url = obj.get("source_url")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        content = self._poll_asyncdata_content(
            task_id=task_id, source_url=source_url, timeout_ms=request.params.timeout_ms
        )
        if content is None:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        mp4 = _extract_first_url(_MP4_URL_RE, content)
        if not mp4:
            raise provider_error("async video completed but no mp4 url found")
        part = Part(type="video", mime_type="video/mp4", source=PartSourceUrl(url=mp4))
        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[part])],
        )

    def _poll_asyncdata_content(
        self, *, task_id: str, source_url: object, timeout_ms: int | None
    ) -> str | None:
        poll_urls: list[str] = []
        if isinstance(source_url, str) and source_url.strip():
            poll_urls.append(source_url.strip())
        poll_urls.extend(
            [
                f"{_ASYNCDATA_BASE_URL}/source/{task_id}",
                f"{_ASYNCDATA_PRO_BASE_URL}/source/{task_id}",
            ]
        )

        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                return None
            for url in poll_urls:
                obj = request_json(
                    method="GET",
                    url=url,
                    headers=None,
                    json_body=None,
                    timeout_ms=min(30_000, remaining_ms),
                    proxy_url=self.proxy_url,
                )
                content = obj.get("content")
                if isinstance(content, str) and content:
                    return content
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

    def _kling_text2video(
        self, request: GenerateRequest, *, model_id: str
    ) -> GenerateResponse:
        prompt = self._single_text_prompt(request)
        host = self._base_host()
        video = request.output.video
        body: dict[str, object] = {
            "prompt": prompt,
            "negative_prompt": "",
            "aspect_ratio": (
                video.aspect_ratio if video and video.aspect_ratio else "16:9"
            ),
            "duration": _closest_kling_duration(video.duration_sec if video else None),
            "callback_url": "",
        }
        obj = request_json(
            method="POST",
            url=f"{host}/kling/v1/videos/text2video",
            headers=self._bearer_headers(),
            json_body=body,
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise provider_error("kling submit missing task_id")

        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=1_000),
            )

        poll_url = f"{host}/kling/v1/videos/text2video/{task_id}"
        budget_ms = (
            120_000 if request.params.timeout_ms is None else request.params.timeout_ms
        )
        deadline = time.time() + max(1, budget_ms) / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            obj = request_json(
                method="GET",
                url=poll_url,
                headers=self._bearer_headers(),
                json_body=None,
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )
            data = obj.get("data")
            if not isinstance(data, dict):
                time.sleep(1.0)
                continue
            status = data.get("task_status")
            if status == "failed":
                raise provider_error(
                    f"kling task failed: {data.get('task_status_msg')}"
                )
            if status == "succeed":
                task_result = data.get("task_result")
                if isinstance(task_result, dict):
                    videos = task_result.get("videos")
                    if isinstance(videos, list) and videos:
                        first = videos[0]
                        if isinstance(first, dict):
                            u = first.get("url")
                            if isinstance(u, str) and u:
                                part = Part(
                                    type="video",
                                    mime_type="video/mp4",
                                    source=PartSourceUrl(url=u),
                                )
                                return GenerateResponse(
                                    id=f"sdk_{uuid4().hex}",
                                    provider="tuzi-web",
                                    model=f"tuzi-web:{model_id}",
                                    status="completed",
                                    output=[Message(role="assistant", content=[part])],
                                )
                raise provider_error("kling task succeeded but missing video url")
            time.sleep(min(1.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(job_id=task_id, poll_after_ms=1_000),
        )

    def _suno_music(
        self, request: GenerateRequest, *, model_id: str
    ) -> GenerateResponse:
        prompt = self._single_text_prompt(request)
        host = self._base_host()
        mid_l = model_id.lower().strip()
        if not (mid_l.startswith("chirp-") and mid_l != "chirp-v3"):
            raise invalid_request_error(
                f"unsupported music model_id: {model_id} (use chirp-*, e.g. chirp-v3-5)"
            )
        body: dict[str, object] = {"prompt": prompt, "mv": model_id}
        opts = request.provider_options.get("tuzi-web")
        if isinstance(opts, dict):
            for k, v in opts.items():
                if k in body:
                    raise invalid_request_error(f"provider_options cannot override {k}")
                if k == "tags" and isinstance(v, list) and all(
                    isinstance(x, str) and x.strip() for x in v
                ):
                    body["tags"] = ",".join([x.strip() for x in v])
                    continue
                body[k] = v
        obj = request_json(
            method="POST",
            url=f"{host}/suno/submit/music",
            headers=self._bearer_headers(),
            json_body=body,
            timeout_ms=max(request.params.timeout_ms or 60_000, 60_000),
            proxy_url=self.proxy_url,
        )

        clip_id: str | None = None
        audio_url: str | None = None
        data = obj.get("data")
        sources: list[dict[str, object]] = [obj]
        if isinstance(data, dict):
            sources.append(data)
        for src in sources:
            clips = src.get("clips")
            if not isinstance(clips, list):
                continue
            for clip in clips:
                if not isinstance(clip, dict):
                    continue
                if clip_id is None:
                    cid = clip.get("id")
                    if isinstance(cid, str) and cid.strip():
                        clip_id = cid.strip()
                if audio_url is None:
                    au = clip.get("audio_url")
                    if isinstance(au, str) and au.strip():
                        audio_url = au.strip()
                if clip_id is not None and audio_url is not None:
                    break
            if clip_id is not None:
                break

        if request.wait and audio_url:
            part = Part(
                type="audio",
                mime_type="audio/mpeg",
                source=PartSourceUrl(url=audio_url),
            )
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="completed",
                output=[Message(role="assistant", content=[part])],
            )

        task_id: str | None = None
        if isinstance(data, str) and data.strip():
            task_id = data.strip()
        for src in sources:
            if task_id is not None:
                break
            for key in ("task_id", "id"):
                v = src.get(key)
                if isinstance(v, str) and v.strip():
                    task_id = v.strip()
                    break
        if task_id is not None:
            return self._suno_wait_fetch_audio(
                task_id=task_id,
                model_id=model_id,
                timeout_ms=request.params.timeout_ms,
                wait=request.wait,
            )

        if clip_id is not None:
            return self._suno_wait_feed_audio(
                clip_id=clip_id,
                model_id=model_id,
                timeout_ms=request.params.timeout_ms,
                wait=request.wait,
            )

        keys = ",".join(sorted([k for k in obj.keys() if isinstance(k, str)])) or "<none>"
        raise provider_error(f"suno music submit missing clip id (keys={keys})")

    def _suno_feed(
        self, *, host: str, ids: str, timeout_ms: int | None
    ) -> dict[str, object]:
        qs = urllib.parse.urlencode({"ids": ids})
        obj = request_json(
            method="GET",
            url=f"{host}/suno/feed?{qs}",
            headers=self._bearer_headers(),
            json_body=None,
            timeout_ms=timeout_ms,
            proxy_url=self.proxy_url,
        )
        if not isinstance(obj, dict):
            raise provider_error("suno feed invalid response", retryable=True)
        return obj

    def _suno_wait_feed_audio(
        self, *, clip_id: str, model_id: str, timeout_ms: int | None, wait: bool
    ) -> GenerateResponse:
        if not wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=clip_id, poll_after_ms=2_000),
            )

        host = self._base_host()
        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        last_status: str | None = None
        last_detail: str | None = None
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break

            data = self._suno_feed(
                host=host, ids=clip_id, timeout_ms=min(30_000, remaining_ms)
            )
            clips = data.get("clips")
            clip: dict[str, object] | None = None
            if isinstance(clips, list):
                for c in clips:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("id")
                    if isinstance(cid, str) and cid.strip() == clip_id:
                        clip = c
                        break
                if clip is None and clips and isinstance(clips[0], dict):
                    clip = clips[0]

            if clip is not None:
                raw_status = clip.get("status")
                status = raw_status.strip().upper() if isinstance(raw_status, str) else ""
                if status:
                    last_status = status
                elif raw_status is not None:
                    last_status = str(raw_status)

                au = clip.get("audio_url")
                if isinstance(au, str) and au.strip():
                    part = Part(
                        type="audio",
                        mime_type="audio/mpeg",
                        source=PartSourceUrl(url=au.strip()),
                    )
                    return GenerateResponse(
                        id=f"sdk_{uuid4().hex}",
                        provider="tuzi-web",
                        model=f"tuzi-web:{model_id}",
                        status="completed",
                        output=[Message(role="assistant", content=[part])],
                    )

                if status in {"FAIL", "FAILED", "ERROR"}:
                    meta = clip.get("metadata")
                    if isinstance(meta, dict):
                        err = meta.get("error_message")
                        if isinstance(err, str) and err:
                            last_detail = err
                    raise provider_error(f"suno feed task failed: {last_detail or ''}".strip())

                meta = clip.get("metadata")
                if isinstance(meta, dict):
                    err = meta.get("error_message")
                    if isinstance(err, str) and err:
                        last_detail = err

            time.sleep(min(2.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(
                job_id=clip_id,
                poll_after_ms=2_000,
                last_status=last_status,
                last_detail=last_detail,
            ),
        )

    def _suno_fetch(
        self, *, host: str, task_id: str, timeout_ms: int | None
    ) -> dict[str, object]:
        obj = request_json(
            method="GET",
            url=f"{host}/suno/fetch/{task_id}",
            headers=self._bearer_headers(),
            json_body=None,
            timeout_ms=timeout_ms,
            proxy_url=self.proxy_url,
        )
        data = obj.get("data")
        if not isinstance(data, dict):
            raise provider_error("suno fetch missing data")
        return data

    def _suno_wait_fetch_audio(
        self, *, task_id: str, model_id: str, timeout_ms: int | None, wait: bool
    ) -> GenerateResponse:
        if not wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )
        host = self._base_host()
        budget_ms = 120_000 if timeout_ms is None else timeout_ms
        deadline = time.time() + max(1, budget_ms) / 1000.0
        last_status: str | None = None
        last_detail: str | None = None
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break
            data = self._suno_fetch(
                host=host, task_id=task_id, timeout_ms=min(30_000, remaining_ms)
            )
            raw_status = data.get("status")
            status = raw_status.strip().upper() if isinstance(raw_status, str) else ""
            if status:
                last_status = status
            elif raw_status is not None:
                last_status = str(raw_status)
            fail_reason = data.get("fail_reason")
            if isinstance(fail_reason, str) and fail_reason:
                last_detail = fail_reason

            def _collect_urls(obj: object) -> list[str]:
                urls: list[str] = []
                if isinstance(obj, dict):
                    u = obj.get("audio_url")
                    if isinstance(u, str) and u.strip():
                        urls.append(u.strip())
                    clips = obj.get("clips")
                    if isinstance(clips, list):
                        for clip in clips:
                            urls.extend(_collect_urls(clip))
                elif isinstance(obj, list):
                    for item in obj:
                        urls.extend(_collect_urls(item))
                return urls

            inner = data.get("data")
            urls = _collect_urls(data) + _collect_urls(inner)
            if not urls:
                blob = json.dumps(inner if inner is not None else data, ensure_ascii=False)
                u = _extract_first_url(_AUDIO_URL_RE, blob)
                if u:
                    urls.append(u)
            if urls:
                part = Part(
                    type="audio",
                    mime_type="audio/mpeg",
                    source=PartSourceUrl(url=urls[0]),
                )
                return GenerateResponse(
                    id=f"sdk_{uuid4().hex}",
                    provider="tuzi-web",
                    model=f"tuzi-web:{model_id}",
                    status="completed",
                    output=[Message(role="assistant", content=[part])],
                )

            if status in {"SUCCESS", "SUCCEEDED", "COMPLETE", "COMPLETED", "DONE"}:
                raise provider_error("suno music succeeded but missing audio url")
            if status in {"FAIL", "FAILED", "ERROR"}:
                raise provider_error(f"suno task failed: {data.get('fail_reason')}")
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="running",
            job=JobInfo(
                job_id=task_id,
                poll_after_ms=2_000,
                last_status=last_status,
                last_detail=last_detail,
            ),
        )

    def _deepsearch(
        self, request: GenerateRequest, *, model_id: str
    ) -> GenerateResponse:
        """Handle deepsearch models via asyncdata.net async API."""
        if self.openai is None:
            raise invalid_request_error(
                "NOUS_GENAI_TUZI_OPENAI_API_KEY required for deepsearch models"
            )

        # Build chat completions body
        messages = []
        for msg in request.input:
            role = msg.role
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "assistant"
            else:
                role = "user"
            text = "".join(p.require_text() for p in msg.content if p.type == "text")
            if text:
                messages.append({"role": role, "content": text})

        if not messages:
            raise invalid_request_error("deepsearch requires at least one message")

        # asyncdata.net requires -async suffix for deepsearch models
        api_model_id = model_id if model_id.endswith("-async") else f"{model_id}-async"
        body: dict[str, Any] = {"model": api_model_id, "messages": messages}
        if request.params.temperature is not None:
            body["temperature"] = request.params.temperature

        # Submit async task
        # Note: URL is NOT encoded per official API docs
        original_url = f"{self.openai.base_url}/chat/completions"
        submit_url = f"{_ASYNCDATA_BASE_URL}/tran/{original_url}"

        # asyncdata.net may take a long time to return task_id; retry on transient errors
        submit_timeout_ms = max(request.params.timeout_ms or 300_000, 300_000)
        last_error: str | None = None
        for attempt in range(3):
            obj = request_json(
                method="POST",
                url=submit_url,
                headers={"Authorization": f"Bearer {self.openai.api_key}"},
                json_body=body,
                timeout_ms=submit_timeout_ms,
                proxy_url=self.proxy_url,
            )
            task_id = obj.get("id")
            if isinstance(task_id, str) and task_id:
                break
            last_error = obj.get("error", "missing task id")
            time.sleep(1.0)  # Brief delay before retry
        else:
            raise provider_error(f"deepsearch submit failed: {last_error}")

        # Non-blocking mode
        if not request.wait:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        # Blocking mode: poll until complete
        budget_ms = request.params.timeout_ms or 300_000  # 5 min default for deepsearch
        deadline = time.time() + max(1, budget_ms) / 1000.0
        content = self._poll_deepsearch(task_id=task_id, deadline=deadline)

        if content is None:
            return GenerateResponse(
                id=f"sdk_{uuid4().hex}",
                provider="tuzi-web",
                model=f"tuzi-web:{model_id}",
                status="running",
                job=JobInfo(job_id=task_id, poll_after_ms=2_000),
            )

        return GenerateResponse(
            id=f"sdk_{uuid4().hex}",
            provider="tuzi-web",
            model=f"tuzi-web:{model_id}",
            status="completed",
            output=[Message(role="assistant", content=[Part.from_text(content)])],
        )

    def _poll_deepsearch(self, *, task_id: str, deadline: float) -> str | None:
        """Poll asyncdata.net until task completes or deadline reached."""
        poll_url = f"{_ASYNCDATA_BASE_URL}/source/{task_id}"
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                return None

            obj = request_json(
                method="GET",
                url=poll_url,
                headers=None,
                json_body=None,
                timeout_ms=min(30_000, remaining_ms),
                proxy_url=self.proxy_url,
            )

            content = obj.get("content")
            if isinstance(content, str) and content:
                return content

            # Still processing, wait before next poll
            time.sleep(min(2.0, max(0.0, deadline - time.time())))

    def list_models(self, *, timeout_ms: int | None = None) -> list[str]:
        """
        Fetch remote model ids by querying each underlying protocol adapter (when configured).
        """
        out: set[str] = set()
        if self.openai is not None:
            try:
                openai_models = self.openai.list_models(timeout_ms=timeout_ms)
            except GenAIError:
                openai_models = []
            if openai_models:
                return openai_models
        if self.gemini is not None:
            try:
                out.update(self.gemini.list_models(timeout_ms=timeout_ms))
            except GenAIError:
                pass
        if self.anthropic is not None:
            try:
                out.update(self.anthropic.list_models(timeout_ms=timeout_ms))
            except GenAIError:
                pass
        return sorted(out)

    def _route(self, model_id: str):
        mid = model_id.strip()
        if not mid:
            raise invalid_request_error("model_id must not be empty")
        mid_l = mid.lower()

        if mid_l.startswith("claude-"):
            if self.anthropic is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_ANTHROPIC_API_KEY/TUZI_ANTHROPIC_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.anthropic

        if mid_l.startswith(("models/", "gemini-", "gemma-", "veo-")):
            if self.gemini is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_GOOGLE_API_KEY/TUZI_GOOGLE_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.gemini
        if mid_l.startswith("veo2"):
            if self.gemini is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_GOOGLE_API_KEY/TUZI_GOOGLE_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.gemini

        if mid_l in {
            "text-embedding-004",
            "embedding-001",
            "embedding-gecko-001",
            "gemini-embedding-001",
            "gemini-embedding-exp-03-07",
        }:
            if self.gemini is None:
                raise invalid_request_error(
                    "NOUS_GENAI_TUZI_GOOGLE_API_KEY/TUZI_GOOGLE_API_KEY "
                    "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
                )
            return self.gemini

        if self.openai is None:
            raise invalid_request_error(
                "NOUS_GENAI_TUZI_OPENAI_API_KEY/TUZI_OPENAI_API_KEY "
                "(or NOUS_GENAI_TUZI_WEB_API_KEY/TUZI_WEB_API_KEY) not configured"
            )
        return self.openai
