"""
OpenAI-compatible proxy using Claude Max subscription.

Architecture:
  Startup: spawns ONE claude --print from clean temp dir Ã¢â€ â€™ captures auth headers + minimal body template
  Per-request: direct aiohttp call to api.anthropic.com via persistent session

Optimizations:
  - Clean temp dir capture: ~350 chars system-reminder vs ~16K (no CLAUDE.md bloat)
  - Persistent aiohttp session: reuses TCP+TLS connection (saves ~200-500ms/request)
  - Direct API calls: no subprocess per request

Concurrency fix:
  - Single-flight auth refresh: if auth expires and many requests arrive at once,
    exactly ONE capture_auth() runs; others wait for it, then proceed.
"""

import asyncio
import json
import time
import uuid
import logging
import shutil
import os
import copy
import codecs
import re
from stream_guard import guard_sse
import tempfile
from contextlib import asynccontextmanager, suppress
from itertools import count
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from aiohttp import web
import aiohttp
from chatgpt_provider import ChatGPTProvider, complete as complete_chatgpt, protect
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
if __name__ == "__main__":
    # Mirror the console to logs/proxy.log so failures can be read after the fact.
    # Only metadata is logged here: never prompts, replies, keys or auth headers.
    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(_log_dir, exist_ok=True)
    _file_handler = RotatingFileHandler(os.path.join(_log_dir, "proxy.log"),
                                        maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(_file_handler)
logger = logging.getLogger("proxy")

DEFAULT_MODEL = "claude-sonnet-4-6"
INTIMATE_SCENE_FALLBACKS = ("deepseek/deepseek-v3.2", "google/gemma-4-31b-it")
_intimate_scene_counter = count()
OPENROUTER_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)?")
CLAUDE_OR_GPT_MODEL = re.compile(r"(?:^|[._-])(?:claude|chatgpt|gpt)(?=$|[^a-z])")
CLAUDE_OR_GPT_PROVIDERS = {"anthropic", "openai", "chatgpt", "azure", "azure-openai"}
# Emitted by the OStimNet active-scene template branches, never by encounter ledgers.
ACTIVE_SCENE_TAG = "[[isekai-route:ostim-active-v1]]"
ACTIVE_SCENE_TAG_LINE = re.compile(rf"^[ \t]*{re.escape(ACTIVE_SCENE_TAG)}[ \t]*(?:\r?\n|$)", re.MULTILINE)
INTERCEPTOR_PORT = 9999

CLAUDE_PATH = shutil.which("claude")
logger.info(f"Using Claude CLI: {CLAUDE_PATH}")

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
PROTECTED_CONFIG_FILE = CONFIG_FILE + ".protected"
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))
chatgpt = ChatGPTProvider(os.path.join(os.path.dirname(CONFIG_FILE), "chatgpt-accounts.protected"))


def _load_config() -> dict:
    """Load persisted config from disk, return empty dict on missing/corrupt file."""
    try:
        if os.path.exists(PROTECTED_CONFIG_FILE):
            with open(PROTECTED_CONFIG_FILE, "rb") as f:
                return json.loads(protect(f.read(), decrypt=True))
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(data: dict) -> None:
    """Persist config dict to disk."""
    descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(CONFIG_FILE))
    try:
        with os.fdopen(descriptor, "wb") as f:
            f.write(protect(json.dumps(data).encode()))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, PROTECTED_CONFIG_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# --- Auth cache + persistent session (with single-flight refresh) ---

class AuthCache:
    def __init__(self):
        self.headers: Optional[dict] = None
        self.body_template: Optional[dict] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Single-flight refresh controls
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def is_ready(self) -> bool:
        return self.headers is not None and self.body_template is not None

    def invalidate(self) -> None:
        self.headers = None
        self.body_template = None

    async def ensure_ready(self, *, force: bool = False, timeout: float = 60.0) -> None:
        """
        If auth is ready -> return.
        If a refresh is already running -> wait for it.
        Else -> start ONE capture_auth() and everyone waits for it.
        """
        async with self._refresh_lock:
            if self.is_ready and not force:
                return

            # If a refresh is already running, reuse it.
            if self._refresh_task and not self._refresh_task.done():
                task = self._refresh_task
            else:
                # Start a new refresh
                self.invalidate()
                self._refresh_task = asyncio.create_task(capture_auth())
                task = self._refresh_task

        # Await outside the lock so other requests can queue behind the same task.
        try:
            await asyncio.wait_for(task, timeout=timeout)
        finally:
            if task.done():
                async with self._refresh_lock:
                    if self._refresh_task is task:
                        self._refresh_task = None

        if not self.is_ready:
            raise RuntimeError("Auth refresh finished but headers/template were not captured")

auth = AuthCache()


# --- OpenRouter + Round-Robin state ---
_cfg = _load_config()
openrouter_api_key: Optional[str] = _cfg.get("openrouter_api_key") or None
if openrouter_api_key:
    logger.info("OpenRouter API key loaded from config.json")
_round_robin_counter: int = 0


def parse_model_list(model_field: str) -> list[str]:
    """Parse comma-separated model list from request, trimming whitespace."""
    return [m.strip() for m in model_field.split(",") if m.strip()]


def pick_model_round_robin(models: list[str]) -> str:
    """Pick next model from list using round-robin."""
    global _round_robin_counter
    model = models[_round_robin_counter % len(models)]
    _round_robin_counter += 1
    return model


def is_openrouter_model(model: str) -> bool:
    """OpenRouter models use 'provider/model' format (contain '/')."""
    return "/" in model


def pick_scene_model(models: list[str], requested_fallbacks: object) -> str:
    if isinstance(requested_fallbacks, list):
        models = models + [item.strip() for item in requested_fallbacks if isinstance(item, str)]
    preferred = []
    last_resort = []
    for model in dict.fromkeys(models):
        model_id = model.casefold()
        if model_id.startswith("claude-") and "/" not in model_id:
            last_resort.append(model)
        elif OPENROUTER_MODEL_ID.fullmatch(model_id):
            provider, name = model_id.split("/", 1)
            if provider == "openrouter" or name.partition(":")[0] in {"auto", "router"}:
                continue
            if provider in CLAUDE_OR_GPT_PROVIDERS or CLAUDE_OR_GPT_MODEL.search(name):
                last_resort.append(model)
            else:
                preferred.append(model)

    eligible = preferred or last_resort or INTIMATE_SCENE_FALLBACKS
    return eligible[next(_intimate_scene_counter) % len(eligible)] if len(eligible) > 1 else eligible[0]


# --- MITM Interceptor (startup only) ---

async def interceptor_handler(request):
    body = await request.read()
    headers = dict(request.headers)
    headers.pop("Host", None)
    headers.pop("host", None)

    try:
        parsed = json.loads(body)
    except Exception:
        parsed = {}

    model = parsed.get("model", "")
    real_url = f"https://api.anthropic.com{request.path_qs}"

    # Skip haiku warmup and token counting
    if "haiku" in model or "count_tokens" in request.path:
        async with aiohttp.ClientSession() as session:
            async with session.post(real_url, data=body, headers=headers) as resp:
                resp_body = await resp.read()
                return web.Response(
                    body=resp_body,
                    status=resp.status,
                    headers={"Content-Type": resp.headers.get("Content-Type", "application/json")},
                )

    # Capture auth headers and body template (skip preflight requests missing real payload)
    if not auth.is_ready and "system" in parsed and "messages" in parsed:
        # Build template locally, then assign both fields "atomically"
        captured_headers = dict(headers)

        # Strip tool definitions (60KB dead weight) and extended thinking
        parsed.pop("tools", None)
        parsed.pop("thinking", None)
        parsed.pop("context_management", None)

        captured_template = parsed

        auth.headers = captured_headers
        auth.body_template = captured_template

        template_size = len(json.dumps(captured_template))
        logger.info(f"Captured {len(auth.headers)} headers + template ({template_size:,} bytes, tools stripped)")

    # Forward to real API
    async with aiohttp.ClientSession() as session:
        async with session.post(real_url, data=body, headers=headers) as resp:
            resp_body = await resp.read()
            return web.Response(
                body=resp_body,
                status=resp.status,
                headers={"Content-Type": resp.headers.get("Content-Type", "text/event-stream")},
            )


async def capture_auth():
    # Use clean temp dir to minimize system-reminder bloat (no CLAUDE.md, no skills)
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{INTERCEPTOR_PORT}"

        logger.info("Capturing auth from claude --print...")
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_PATH,
            "--print",
            "--output-format",
            "text",
            "--model",
            DEFAULT_MODEL,
            "--no-session-persistence",
            "--system-prompt",
            "Say ok",
            "ok",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=tmpdir,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if proc.returncode is None:
                proc.kill()
            with suppress(Exception):
                await proc.wait()
            raise

        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="replace")[:800]
            out = (stdout or b"").decode("utf-8", errors="replace")[:800]
            logger.error(f"claude --print failed rc={proc.returncode}\nSTDERR:\n{err}\nSTDOUT:\n{out}")
            raise RuntimeError("claude --print failed while capturing auth")


async def start_interceptor():
    """Start MITM interceptor and capture auth from a clean temp dir."""
    iapp = web.Application()
    iapp.router.add_route("*", "/{path_info:.*}", interceptor_handler)

    runner = web.AppRunner(iapp)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", INTERCEPTOR_PORT)
    await site.start()
    logger.info(f"Interceptor on port {INTERCEPTOR_PORT}")

    return runner


# --- Direct API call ---

def _build_api_body(system_prompt: Optional[str], messages: list, model: str) -> dict:
    """Build Anthropic API request body from template."""
    body = copy.deepcopy(auth.body_template)

    # 1. Replace system prompt (keep billing block 0)
    billing = body["system"][0]
    body["system"] = [billing]
    if system_prompt:
        body["system"].extend(claude_text_blocks(system_prompt))

    # 2. Build full conversation, preserving template auth blocks in first user msg
    auth_blocks = []
    template_first = body["messages"][0] if body["messages"] else {}
    if isinstance(template_first.get("content"), list):
        for block in template_first["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                if "<system-reminder>" in block.get("text", ""):
                    auth_blocks.append(block)

    new_messages = []
    for i, m in enumerate(messages):
        if i == 0 and m["role"] == "user":
            content = auth_blocks + claude_text_blocks(m["content"])
            new_messages.append({"role": "user", "content": content})
        else:
            new_messages.append({"role": m["role"], "content": claude_text_blocks(m["content"])})
    body["messages"] = new_messages

    # Keep the earliest cache breakpoints and leave all prompt text intact.
    # Automatic caching also consumes one of Anthropic's four slots.
    remaining_cache_slots = 4 - int("cache_control" in body)
    cache_blocks = body.get("tools", []) + body["system"]
    for message in new_messages:
        cache_blocks.extend(message["content"])
    for block in cache_blocks:
        if "cache_control" not in block:
            continue
        if remaining_cache_slots:
            remaining_cache_slots -= 1
        else:
            del block["cache_control"]

    # 3. Model, streaming, and disable extended thinking
    body["model"] = model
    body["stream"] = True
    body.pop("thinking", None)
    return body


async def call_api_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int) -> str:
    """Direct API call, collects full response (non-streaming to caller)."""
    body = _build_api_body(system_prompt, messages, model)
    headers = dict(auth.headers)
    body_bytes = json.dumps(body).encode("utf-8")
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs)")
    start = time.time()

    session = auth.session or aiohttp.ClientSession()
    owns_session = auth.session is None
    try:
        async with session.post(
            "https://api.anthropic.com/v1/messages?beta=true",
            data=body_bytes,
            headers=headers,
        ) as resp:
            elapsed = time.time() - start

            if resp.status != 200:
                error_body = await resp.read()
                error_text = error_body.decode("utf-8", errors="replace")
                logger.error(f"[{request_id}] API {resp.status}: {error_text[:300]}")
                if resp.status in (401, 403) or "credential" in error_text.lower():
                    logger.warning("Auth expired/invalid (direct)")
                    auth.invalidate()
                raise HTTPException(status_code=resp.status, detail=error_text[:200])

            resp_body = await resp.read()
            text_parts = []
            content_type = resp.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                for line in resp_body.decode("utf-8", errors="replace").split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text_parts.append(delta.get("text", ""))
                        except json.JSONDecodeError:
                            pass
            else:
                try:
                    data = json.loads(resp_body)
                    for block in data.get("content", []):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                except json.JSONDecodeError:
                    pass

            response_text = "".join(text_parts)
            logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s)")
            return response_text
    finally:
        if owns_session:
            await session.close()


async def call_api_streaming_with_retry(system_prompt: Optional[str], messages: list, model: str, max_tokens: int):
    """
    Direct API call, yields OpenAI-format SSE chunks as they arrive.
    Retries ONCE on 401/403 by forcing a single-flight auth refresh before yielding any chunks.
    """
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    for attempt in range(2):
        await auth.ensure_ready(timeout=60)

        body = _build_api_body(system_prompt, messages, model)
        headers = dict(auth.headers)
        body_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Length"] = str(len(body_bytes))

        request_id = uuid.uuid4().hex[:8]
        logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs, stream, attempt={attempt+1})")
        start = time.time()
        total_chars = 0

        session = auth.session or aiohttp.ClientSession()
        owns_session = auth.session is None
        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages?beta=true",
                data=body_bytes,
                headers=headers,
            ) as resp:
                # If auth invalid, refresh and retry BEFORE yielding anything.
                if resp.status in (401, 403) and attempt == 0:
                    error_text = (await resp.read()).decode("utf-8", errors="replace")[:300]
                    logger.warning(f"[{request_id}] Auth invalid during stream start: {error_text}")
                    auth.invalidate()
                    await auth.ensure_ready(force=True, timeout=60)
                    continue

                if resp.status != 200:
                    error_body = await resp.read()
                    error_text = error_body.decode("utf-8", errors="replace")
                    logger.error(f"[{request_id}] API {resp.status}: {error_text[:300]}")
                    if resp.status in (401, 403) or "credential" in error_text.lower():
                        logger.warning("Auth expired/invalid (stream)")
                        auth.invalidate()

                    err_chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"[API Error {resp.status}]"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Stream Claude SSE -> OpenAI SSE
                role_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

                buffer = ""
                upstream_completed = False
                # Diagnostics for replies that arrive without text (SkyrimNet: "Empty completion").
                anthropic_id = resp.headers.get("request-id", "?")
                stop_reason = None
                stream_error = None
                block_types = []
                last_event = None
                try:
                    async for raw_chunk in resp.content.iter_any():
                        buffer += raw_chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            event_type = event.get("type")
                            last_event = event_type
                            if event_type == "message_stop":
                                upstream_completed = True
                            elif event_type == "message_delta":
                                stop_reason = event.get("delta", {}).get("stop_reason") or stop_reason
                            elif event_type == "content_block_start":
                                block_types.append(event.get("content_block", {}).get("type"))
                            elif event_type == "error":
                                error = event.get("error", {})
                                stream_error = f"{error.get('type')}: {str(error.get('message'))[:200]}"
                                logger.warning(f"[{request_id}] Claude stream error from Anthropic "
                                               f"(request-id {anthropic_id}): {stream_error}")
                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        total_chars += len(text)
                                        oai_chunk = {
                                            "id": cmpl_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                        }
                                        yield f"data: {json.dumps(oai_chunk)}\n\n"
                except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                    logger.warning(f"[{request_id}] Claude stream connection failed after {time.time() - start:.1f}s "
                                   f"({total_chars} chars, request-id {anthropic_id}): {type(error).__name__}: {error}")
                    raise

                elapsed = time.time() - start
                if not upstream_completed:
                    logger.warning(f"[{request_id}] Claude stream ended before message_stop after {elapsed:.1f}s "
                                   f"({total_chars} chars, last event {last_event}, error {stream_error or 'none'}, "
                                   f"request-id {anthropic_id})")
                    raise RuntimeError("Claude stream ended before message_stop")
                if not total_chars:
                    logger.warning(f"[{request_id}] Claude replied with no text after {elapsed:.1f}s "
                                   f"(stop_reason {stop_reason}, blocks {block_types}, request-id {anthropic_id})")

                stop_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                logger.info(f"[{request_id}] <- {total_chars} chars ({elapsed:.1f}s, streamed, stop {stop_reason})")
                return
        finally:
            if owns_session:
                await session.close()

    # If we somehow got here, retry didn't help
    err_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": "[Auth refresh failed]"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(err_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    return


# --- OpenRouter API calls ---

async def call_openrouter_direct(openrouter_api_key: str, system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to OpenRouter (OpenAI-compatible), collect full response."""

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append(copy.deepcopy(m))

    payload = {"model": model, "messages": oai_messages, "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})
    for field in ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call", "reasoning_effort"):
        payload.pop(field, None)
    payload["reasoning"] = {"effort": "none", "exclude": True}
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs)")
    start = time.time()

    session = auth.session or aiohttp.ClientSession()
    owns_session = auth.session is None
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            elapsed = time.time() - start
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenRouter {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, OpenRouter)")
            return text
    finally:
        if owns_session:
            await session.close()


async def call_openrouter_streaming(openrouter_api_key: str, system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to OpenRouter with streaming, passthrough SSE directly."""

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append(copy.deepcopy(m))

    payload = {"model": model, "messages": oai_messages, "max_tokens": max_tokens, "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})
    for field in ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call", "reasoning_effort"):
        payload.pop(field, None)
    payload["reasoning"] = {"effort": "none", "exclude": True}
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs, stream)")
    start = time.time()

    session = auth.session or aiohttp.ClientSession()
    owns_session = auth.session is None
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenRouter {resp.status}: {error_body[:300]}")
                err_chunk = {"error": {"message": f"OpenRouter request failed ({resp.status})", "type": "upstream_error"}}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                return

            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")()
            async for raw_chunk in resp.content.iter_any():
                buffer += decoder.decode(raw_chunk)
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    event = event.strip()
                    if event:
                        yield event + "\n\n"

            if buffer.strip():
                yield buffer.strip() + "\n\n"

            elapsed = time.time() - start
            logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, OpenRouter)")
    finally:
        if owns_session:
            await session.close()


# --- OpenAI-compatible API ---

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]
    model_config = ConfigDict(extra="allow")


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


def consume_active_scene_tag(messages: list[ChatMessage]) -> bool:
    active = False
    for message in messages:
        if message.role not in ("system", "developer", "user"):
            continue
        if isinstance(message.content, str):
            if ACTIVE_SCENE_TAG in message.content:
                message.content, removed = ACTIVE_SCENE_TAG_LINE.subn("", message.content)
                active |= bool(removed)
            continue
        for block in message.content:
            text = block.get("text")
            if block.get("type") == "text" and isinstance(text, str) and ACTIVE_SCENE_TAG in text:
                block["text"], removed = ACTIVE_SCENE_TAG_LINE.subn("", text)
                active |= bool(removed)
    return active


@asynccontextmanager
async def lifespan(app):
    runner = await start_interceptor() if CLAUDE_PATH else None

    # Persistent outgoing HTTP session
    auth.session = aiohttp.ClientSession()

    async def warm_claude():
        try:
            await auth.ensure_ready(force=True, timeout=60)
        except Exception:
            logger.warning("Claude sign-in unavailable. ChatGPT and OpenRouter remain available.")

    warm_task = asyncio.create_task(warm_claude()) if CLAUDE_PATH else None

    try:
        yield
    finally:
        if warm_task:
            warm_task.cancel()
            with suppress(asyncio.CancelledError):
                await warm_task
        if auth.session:
            await auth.session.close()
            auth.session = None
        if runner:
            await runner.cleanup()


app = FastAPI(title="Claude SkyrimNet Proxy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def claude_text_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if any(block.get("type") != "text" or not isinstance(block.get("text"), str) for block in content):
        raise HTTPException(400, "The Claude route currently supports text blocks only")
    return copy.deepcopy(content)


def completion_result(text, model, prompt):
    prompt_tokens = len(prompt) // 4
    completion_tokens = len(text) // 4
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text, "name": None},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    if not any(message.role == "user" for message in req.messages):
        raise HTTPException(400, "No user message provided")
    models = parse_model_list(req.model or DEFAULT_MODEL)
    intimate_scene = consume_active_scene_tag(req.messages)
    if intimate_scene:
        model = pick_scene_model(models, (req.model_extra or {}).get("models"))
        logger.info("OStimNet scene route -> %s", model)
    else:
        if not models:
            raise HTTPException(400, "A model is required")
        model = pick_model_round_robin(models) if len(models) > 1 else models[0]
    if model == "claude-sonnet-4-5-20250929":
        logger.info("Mapping legacy Sonnet 4.5 request to %s", DEFAULT_MODEL)
        model = DEFAULT_MODEL
    if model.startswith("chatgpt/"):
        try:
            return await complete_chatgpt(chatgpt, auth.session, req, model)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise HTTPException(502, "ChatGPT connection interrupted")

    max_tokens = req.max_tokens or 4096
    stream_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if is_openrouter_model(model):
        key = openrouter_api_key or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not key:
            raise HTTPException(401, "Configure your OpenRouter API key in the dashboard")
        messages = [message.model_dump(exclude_none=True) for message in req.messages]
        extra_params = {key: value for key, value in (req.model_extra or {}).items() if value is not None}
        if intimate_scene:
            # Keep OpenRouter on the pinned model even if the caller supplies routing options.
            for field in ("models", "route", "preset", "plugins"):
                extra_params.pop(field, None)
        if req.temperature is not None:
            extra_params["temperature"] = req.temperature
        if req.stream:
            return StreamingResponse(
                guard_sse(call_openrouter_streaming(key, None, messages, model, max_tokens, **extra_params)),
                media_type="text/event-stream", headers=stream_headers,
            )
        try:
            text = await call_openrouter_direct(key, None, messages, model, max_tokens, **extra_params)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            logger.warning("OpenRouter request interrupted (%s)", type(error).__name__)
            raise HTTPException(502, "OpenRouter connection interrupted; retry the request") from None
        return completion_result(text, model, json.dumps(messages, ensure_ascii=False))

    if not CLAUDE_PATH:
        raise HTTPException(503, "Claude CLI is not installed. Select a ChatGPT or OpenRouter model.")
    system_blocks = []
    messages = []
    for message in req.messages:
        blocks = claude_text_blocks(message.content)
        if message.role in ("system", "developer"):
            system_blocks.extend(blocks)
        elif message.role in ("user", "assistant"):
            messages.append({"role": message.role, "content": blocks})
        else:
            raise HTTPException(400, "Unsupported message role for Claude")
    if messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "Continue."}]})
    merged = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"].extend(message["content"])
        else:
            merged.append(message)
    try:
        await auth.ensure_ready(timeout=60)
    except Exception:
        raise HTTPException(503, "Claude authentication is not ready")
    if req.stream:
        return StreamingResponse(
            guard_sse(call_api_streaming_with_retry(system_blocks, merged, model, max_tokens)),
            media_type="text/event-stream", headers=stream_headers,
        )
    text = None
    for attempt in range(2):
        try:
            text = await call_api_direct(system_blocks, merged, model, max_tokens)
            break
        except HTTPException as error:
            if error.status_code not in (401, 403) or attempt:
                raise
            auth.invalidate()
            await auth.ensure_ready(force=True, timeout=60)
    if not text:
        raise HTTPException(500, "Empty response")
    return completion_result(text, model, json.dumps(req.model_dump()["messages"], ensure_ascii=False))

@app.post("/config/openrouter-key")
async def set_openrouter_key(request: Request):
    require_local_setup(request)
    global openrouter_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        openrouter_api_key = None
        cfg.pop("openrouter_api_key", None)
        _save_config(cfg)
        logger.info("OpenRouter API key cleared")
        return {"status": "cleared"}
    openrouter_api_key = key
    cfg["openrouter_api_key"] = key
    _save_config(cfg)
    logger.info("OpenRouter API key configured and saved to config.json")
    return {"status": "saved"}


@app.get("/v1/models")
async def list_models():
    data = [
        {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
    ]
    if openrouter_api_key:
        data.append({"id": "openrouter/*", "object": "model", "owned_by": "openrouter"})
    if chatgpt.state["active"]:
        try:
            data.extend(await chatgpt.models(auth.session))
        except (HTTPException, aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("ChatGPT model list unavailable. Reconnect from the dashboard.")
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if auth.is_ready or openrouter_api_key or chatgpt.state["active"] else "needs_sign_in",
        "claude_path": CLAUDE_PATH,
        "auth_cached": auth.is_ready,
        "openrouter_configured": openrouter_api_key is not None,
        "chatgpt": chatgpt.status(),
    }


def require_local_setup(request):
    origin = request.headers.get("origin")
    allowed = {f"http://127.0.0.1:{PROXY_PORT}", f"http://localhost:{PROXY_PORT}"}
    if request.headers.get("x-proxy-setup") != "1" or (origin and origin not in allowed):
        raise HTTPException(403, "Open setup from the local proxy dashboard")


@app.post("/chatgpt/sign-in")
async def chatgpt_sign_in(request: Request):
    require_local_setup(request)
    body = await request.json()
    url = await chatgpt.authorize(f"http://127.0.0.1:{PROXY_PORT}/auth/callback", body.get("account_id"))
    return {"url": url}


@app.get("/auth/callback", response_class=HTMLResponse)
async def chatgpt_callback(request: Request):
    try:
        enabled, first = await chatgpt.callback(auth.session, request.query_params)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        raise HTTPException(502, "Sign-in connection interrupted. Start again.")
    if not enabled:
        return HTMLResponse('<h1>ChatGPT plan usage is disabled</h1><p>Sign in again and enable plan usage to send NPC requests.</p><a href="/chatgpt">Back to setup</a>', headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
    if first:
        return HTMLResponse('<dialog open><h1>YouÃ¢â‚¬â„¢re using your ChatGPT plan</h1><p>Eligible NPC requests in this proxy use your ChatGPT plan. Manage usage in ChatGPT settings.</p><a href="/chatgpt">Got it</a></dialog>', headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
    return HTMLResponse('<p>ChatGPT connected.</p><a href="/chatgpt">Continue</a>', headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@app.get("/chatgpt/status")
async def chatgpt_status():
    return chatgpt.status()


@app.post("/chatgpt/select")
async def chatgpt_select(request: Request):
    require_local_setup(request)
    await chatgpt.select((await request.json()).get("account_id"))
    return chatgpt.status()


@app.get("/chatgpt/models")
async def chatgpt_models():
    try:
        return {"data": await chatgpt.models(auth.session)}
    except (aiohttp.ClientError, asyncio.TimeoutError):
        raise HTTPException(502, "Could not reach the ChatGPT model catalog")


@app.get("/chatgpt", response_class=HTMLResponse)
async def chatgpt_setup():
    return HTMLResponse(CHATGPT_PAGE, headers={"Cache-Control": "no-store"})


with open(os.path.join(os.path.dirname(__file__), "chatgpt.html"), encoding="utf-8") as page:
    CHATGPT_PAGE = page.read()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    status = "Ready" if auth.is_ready else "Claude not connected"
    status_color = "#4ade80" if auth.is_ready else "#facc15"
    template_size = len(json.dumps(auth.body_template)) if auth.body_template else 0

    or_status = "Configured (saved)" if openrouter_api_key else "Not set"
    or_color = "#4ade80" if openrouter_api_key else "#64748b"

    models = [
        ("claude-opus-4-6", "Opus 4.6", "Most capable, slowest"),
        ("claude-sonnet-4-6", "Sonnet 4.6", "Best balance (default)"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", "Fastest, least capable"),
        ("provider/model", "OpenRouter", "Any model via OpenRouter (requires key)"),
    ]
    model_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in models
    )

    return f"""<!DOCTYPE html>
<html><head><title>Claude SkyrimNet Proxy</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:700px; margin:40px auto; padding:0 20px }}
  h1 {{ color:#f8fafc; font-size:1.5rem; margin-bottom:4px }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:30px }}
  .status {{ display:inline-block; padding:4px 12px; border-radius:12px; font-size:0.85rem; font-weight:600;
             background:{status_color}20; color:{status_color}; border:1px solid {status_color}40 }}
  .card {{ background:#1e293b; border-radius:8px; padding:20px; margin:16px 0; border:1px solid #334155 }}
  table {{ width:100%; border-collapse:collapse }}
  th {{ text-align:left; color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; padding:8px 12px; border-bottom:1px solid #334155 }}
  td {{ padding:8px 12px; border-bottom:1px solid #1e293b }}
  .label {{ color:#94a3b8; font-size:0.85rem }}
  .value {{ color:#f1f5f9; font-family:monospace; font-size:0.85rem }}
  .endpoint {{ background:#0f172a; padding:10px 14px; border-radius:6px; font-family:monospace; font-size:0.85rem; color:#67e8f9; margin:8px 0; border:1px solid #334155 }}
  #testArea {{ margin-top:16px }}
  textarea {{ width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:10px; font-family:monospace; font-size:0.85rem; resize:vertical; box-sizing:border-box }}
  button {{ background:#3b82f6; color:white; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin-top:8px }}
  button:hover {{ background:#2563eb }}
  button:disabled {{ background:#475569; cursor:wait }}
  #response {{ margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155; font-size:0.9rem; min-height:40px; white-space:pre-wrap }}
  .timing {{ color:#4ade80; font-size:0.8rem; margin-top:6px }}
</style></head>
<body>
  <h1>Claude SkyrimNet Proxy</h1>
  <div class="subtitle">Claude, ChatGPT subscriptions and OpenRouter for SkyrimNet</div>
  <p><a href="/chatgpt" style="color:#93c5fd">ChatGPT account, models and streaming test</a></p>
  <span class="status">{status}</span>

  <div class="card">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
      <div><span class="label">Endpoint</span><div class="endpoint">http://127.0.0.1:8000/v1/chat/completions</div></div>
      <div><span class="label">API Key</span><div class="endpoint">not required</div></div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:12px">
      <div><span class="label">Template</span><br><span class="value">{template_size:,} bytes</span></div>
      <div><span class="label">Default Model</span><br><span class="value">{DEFAULT_MODEL.split("-")[1].title()} {DEFAULT_MODEL.split("-")[2]}</span></div>
      <div><span class="label">Claude CLI</span><br><span class="value">{os.path.basename(CLAUDE_PATH) if CLAUDE_PATH else 'Not installed'}</span></div>
    </div>
  </div>

  <div class="card">
    <h3 style="margin:0 0 12px; font-size:1rem; color:#f1f5f9">Supported Models</h3>
    <table><thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
    <tbody>{model_rows}</tbody></table>
  </div>

  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">OpenRouter API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="orKey" placeholder="Paste your API key here (sk-or-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOrKey()" style="margin-top:0">Save</button>
      <span id="orStatus" style="color:{or_color}; font-size:0.85rem; font-weight:600">{or_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Use <code style="color:#67e8f9">provider/model</code> IDs (e.g. <code style="color:#67e8f9">openai/gpt-4o</code>) to route through OpenRouter.
      Comma-separate models for round-robin rotation.
    </div>
  </div>

  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Quick Test</h3>
    <textarea id="sysPrompt" rows="2" placeholder="System prompt (e.g. You are Lydia, a Nord housecarl.)">You are Lydia, a Nord housecarl sworn to protect the Dragonborn. Stay in character. One sentence only.</textarea>
    <textarea id="userMsg" rows="1" placeholder="User message" style="margin-top:6px">What do you think of dragons?</textarea>
    <button onclick="testChat()" id="testBtn">Send</button>
    <div id="response" style="display:none"></div>
    <div id="timing" class="timing"></div>
  </div>

<script>
async function saveOrKey() {{
  const key = document.getElementById('orKey').value.trim();
  const status = document.getElementById('orStatus');
  try {{
    await fetch('/config/openrouter-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json', 'X-Proxy-Setup': '1'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('orKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function testChat() {{
  const btn = document.getElementById('testBtn');
  const resp = document.getElementById('response');
  const timing = document.getElementById('timing');
  btn.disabled = true; btn.textContent = 'Waiting...';
  resp.style.display = 'block'; resp.textContent = '...';
  timing.textContent = '';
  const start = Date.now();
  try {{
    const r = await fetch('/v1/chat/completions', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        model: 'claude-sonnet-4-6',
        messages: [
          {{role: 'system', content: document.getElementById('sysPrompt').value}},
          {{role: 'user', content: document.getElementById('userMsg').value}}
        ]
      }})
    }});
    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    if (data.choices) {{
      resp.textContent = data.choices[0].message.content;
      timing.textContent = elapsed + 's';
    }} else {{
      resp.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    resp.textContent = 'Error: ' + e.message;
  }}
  btn.disabled = false; btn.textContent = 'Send';
}}
</script>
</body></html>"""


if __name__ == "__main__":
    # OAuth codes arrive in the callback URL, so suppress URL access logging.
    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT, access_log=False)
