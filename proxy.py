"""
OpenAI-compatible proxy using Claude Max subscription.

Architecture:
  Startup: spawns ONE claude --print from clean temp dir → captures auth headers + minimal body template
  Per-request: direct aiohttp call to api.anthropic.com via persistent session

Optimizations:
  - Clean temp dir capture: ~350 chars system-reminder vs ~16K (no CLAUDE.md bloat)
  - Persistent aiohttp session: reuses TCP+TLS connection (saves ~200-500ms/request)
  - Direct API calls: no subprocess per request
"""

import asyncio
import json
import time
import uuid
import logging
import shutil
import os
import copy
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from aiohttp import web
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy")

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
INTERCEPTOR_PORT = 9999

CLAUDE_PATH = shutil.which("claude")
if not CLAUDE_PATH:
    raise RuntimeError("claude CLI not found on PATH")
logger.info(f"Using Claude CLI: {CLAUDE_PATH}")


# --- Auth cache + persistent session ---

class AuthCache:
    def __init__(self):
        self.headers: Optional[dict] = None
        self.body_template: Optional[dict] = None
        self.session: Optional[aiohttp.ClientSession] = None

    @property
    def is_ready(self):
        return self.headers is not None and self.body_template is not None

auth = AuthCache()


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
                return web.Response(body=resp_body, status=resp.status,
                    headers={"Content-Type": resp.headers.get("Content-Type", "application/json")})

    # Capture auth headers and body template
    if not auth.is_ready:
        auth.headers = dict(headers)
        # Strip tool definitions (60KB dead weight) and extended thinking
        parsed.pop("tools", None)
        parsed.pop("thinking", None)
        auth.body_template = parsed
        template_size = len(json.dumps(parsed))
        logger.info(f"Captured {len(auth.headers)} headers + template ({template_size:,} bytes, tools stripped)")

    # Forward to real API
    async with aiohttp.ClientSession() as session:
        async with session.post(real_url, data=body, headers=headers) as resp:
            resp_body = await resp.read()
            return web.Response(body=resp_body, status=resp.status,
                headers={"Content-Type": resp.headers.get("Content-Type", "text/event-stream")})


async def start_interceptor():
    """Start MITM interceptor and capture auth from a clean temp dir."""
    iapp = web.Application()
    iapp.router.add_route("*", "/{path_info:.*}", interceptor_handler)

    runner = web.AppRunner(iapp)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", INTERCEPTOR_PORT)
    await site.start()
    logger.info(f"Interceptor on port {INTERCEPTOR_PORT}")

    # Use clean temp dir to minimize system-reminder bloat (no CLAUDE.md, no skills)
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{INTERCEPTOR_PORT}"

        logger.info("Warming up: capturing auth from claude --print...")
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_PATH, "--print",
            "--output-format", "text",
            "--model", DEFAULT_MODEL,
            "--no-session-persistence",
            "--system-prompt", "Say ok",
            "ok",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=tmpdir,
        )
        await asyncio.wait_for(proc.communicate(), timeout=60)

    if auth.is_ready:
        # Create persistent session for all future API calls
        auth.session = aiohttp.ClientSession()
        logger.info("Auth captured — direct API mode active (persistent session)")
    else:
        logger.error("Failed to capture auth headers!")

    return runner


# --- Direct API call ---

def _build_api_body(system_prompt: Optional[str], messages: list, model: str) -> dict:
    """Build Anthropic API request body from template."""
    body = copy.deepcopy(auth.body_template)

    # 1. Replace system prompt (keep billing block 0)
    billing = body["system"][0]
    body["system"] = [billing]
    if system_prompt:
        body["system"].append({"type": "text", "text": system_prompt})

    # 2. Build full conversation, preserving template auth blocks in first user msg
    # The template's first user message contains system-reminder content blocks
    # that authenticate this as a Claude Code request — we must keep them.
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
            # First user message: prepend auth blocks from template
            content = auth_blocks + [{"type": "text", "text": m["content"]}]
            new_messages.append({"role": "user", "content": content})
        else:
            new_messages.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    body["messages"] = new_messages

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
                    logger.warning("Auth expired -- restart proxy to re-auth")
                    auth.headers = None
                    auth.body_template = None
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
        if not auth.session:
            await session.close()


async def call_api_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int):
    """Direct API call, yields OpenAI-format SSE chunks as they arrive."""

    body = _build_api_body(system_prompt, messages, model)
    headers = dict(auth.headers)
    body_bytes = json.dumps(body).encode("utf-8")
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs, stream)")
    start = time.time()
    total_chars = 0

    session = auth.session or aiohttp.ClientSession()
    owns_session = not auth.session
    try:
        async with session.post(
            "https://api.anthropic.com/v1/messages?beta=true",
            data=body_bytes,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.read()
                error_text = error_body.decode("utf-8", errors="replace")
                logger.error(f"[{request_id}] API {resp.status}: {error_text[:300]}")
                if resp.status in (401, 403) or "credential" in error_text.lower():
                    logger.warning("Auth expired -- restart proxy to re-auth")
                    auth.headers = None
                    auth.body_template = None
                # Yield an error chunk so client sees the failure
                err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                             "choices": [{"index": 0, "delta": {"content": f"[API Error {resp.status}]"}, "finish_reason": None}]}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Stream Claude SSE -> OpenAI SSE
            # Send initial chunk with role
            role_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                          "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
            yield f"data: {json.dumps(role_chunk)}\n\n"

            buffer = ""
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

                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                total_chars += len(text)
                                oai_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                             "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                                yield f"data: {json.dumps(oai_chunk)}\n\n"

            # Final chunk with finish_reason
            stop_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                          "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            elapsed = time.time() - start
            logger.info(f"[{request_id}] <- {total_chars} chars ({elapsed:.1f}s, streamed)")
    finally:
        if owns_session:
            await session.close()


# --- OpenAI-compatible API ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None
    stream: Optional[bool] = False


@asynccontextmanager
async def lifespan(app):
    runner = await start_interceptor()
    yield
    if auth.session:
        await auth.session.close()
    await runner.cleanup()

app = FastAPI(title="Claude SkyrimNet Proxy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if not auth.is_ready:
        raise HTTPException(status_code=503, detail="Auth not ready -- warming up")

    model = req.model or DEFAULT_MODEL

    system_prompt = None
    anthropic_messages = []
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role in ("user", "assistant"):
            anthropic_messages.append({"role": msg.role, "content": msg.content})

    if not anthropic_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    if anthropic_messages[0]["role"] != "user":
        anthropic_messages.insert(0, {"role": "user", "content": "Continue."})

    # Merge consecutive same-role messages
    merged = []
    for msg in anthropic_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    max_tokens = req.max_tokens or 4096

    # Streaming response (SSE)
    if req.stream:
        return StreamingResponse(
            call_api_streaming(system_prompt, merged, model, max_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming response (JSON)
    response = await call_api_direct(system_prompt, merged, model, max_tokens)

    if not response:
        raise HTTPException(status_code=500, detail="Empty response")

    prompt_text = (system_prompt or "") + " ".join(m["content"] for m in merged)
    prompt_tokens = len(prompt_text) // 4
    completion_tokens = len(response) // 4

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response, "name": None},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": None,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-3-7-sonnet-20250219", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if auth.is_ready else "warming_up",
        "claude_path": CLAUDE_PATH,
        "auth_cached": auth.is_ready,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    status = "Ready" if auth.is_ready else "Warming up..."
    status_color = "#4ade80" if auth.is_ready else "#facc15"
    template_size = len(json.dumps(auth.body_template)) if auth.body_template else 0

    models = [
        ("claude-opus-4-6", "Opus 4.6", "Most capable, slowest"),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", "Best balance (default)"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", "Fastest, least capable"),
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
  <div class="subtitle">OpenAI-compatible proxy using Claude Max subscription</div>
  <span class="status">{status}</span>

  <div class="card">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
      <div><span class="label">Endpoint</span><div class="endpoint">http://127.0.0.1:8000/v1/chat/completions</div></div>
      <div><span class="label">API Key</span><div class="endpoint">not required</div></div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:12px">
      <div><span class="label">Template</span><br><span class="value">{template_size:,} bytes</span></div>
      <div><span class="label">Default Model</span><br><span class="value">{DEFAULT_MODEL.split("-")[1].title()} {DEFAULT_MODEL.split("-")[2]}</span></div>
      <div><span class="label">Claude CLI</span><br><span class="value">{os.path.basename(CLAUDE_PATH)}</span></div>
    </div>
  </div>

  <div class="card">
    <h3 style="margin:0 0 12px; font-size:1rem; color:#f1f5f9">Supported Models</h3>
    <table><thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
    <tbody>{model_rows}</tbody></table>
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
        model: 'claude-sonnet-4-5-20250929',
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
