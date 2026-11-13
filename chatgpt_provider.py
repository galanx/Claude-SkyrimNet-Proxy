"""ChatGPT plan authorization and Responses-to-Chat-Completions translation.

Uses the public Sign in with ChatGPT flow for local open-source apps.
Credentials are DPAPI protected on Windows and owner-readable on Unix.
"""

import asyncio
import base64
import codecs
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlencode
import uuid

import aiohttp
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import jwt

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"


def protect(data, decrypt=False):
    if os.name != "nt":
        return data
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    result = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    operation = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    operation.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(Blob)]
    operation.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        kernel.LocalFree(result.data)


class ChatGPTProvider:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = asyncio.Lock()
        self.pending = {}
        self.state = {"host_id": f"urn:uuid:{uuid.uuid4()}", "accounts": {}, "active": None}
        if self.path.exists():
            self.state = json.loads(protect(self.path.read_bytes(), decrypt=True))
        self.jwks = jwt.PyJWKClient(f"{ISSUER}/.well-known/jwks.json")

    def save(self, state):
        encoded = protect(json.dumps(state).encode())
        descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".chatgpt-")
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.state = state

    def status(self):
        accounts = [{"id": key, "label": record.get("email") or record["subject"],
                     "plan_enabled": PLAN_SCOPE in record.get("scopes", [])}
                    for key, record in self.state["accounts"].items()]
        return {"active": self.state["active"], "accounts": accounts}

    async def authorize(self, redirect_uri, account_id=None):
        async with self.lock:
            record = self.state["accounts"].get(account_id) if account_id else None
            if account_id and not record:
                raise HTTPException(400, "Unknown ChatGPT account")
            self.save(self.state)
            state = secrets.token_urlsafe(32)
            nonce = secrets.token_urlsafe(32)
            verifier = secrets.token_urlsafe(48)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
            self.pending = {key: value for key, value in self.pending.items()
                            if value["expires"] > time.time()}
            if len(self.pending) >= 8:
                raise HTTPException(429, "Finish an existing sign-in first")
            self.pending[state] = {"nonce": nonce, "verifier": verifier,
                                   "redirect_uri": redirect_uri, "account_id": account_id,
                                   "expires": time.time() + 600}
            params = {"client_id": account_id or "dynamic_agent_client",
                      "ext_agent_host_id": self.state["host_id"], "response_type": "code",
                      "redirect_uri": redirect_uri, "scope": SCOPES, "resource": RESOURCE,
                      "state": state, "nonce": nonce, "code_challenge_method": "S256",
                      "code_challenge": challenge}
            if record:
                if record.get("id_token"):
                    params["id_token_hint"] = record["id_token"]
            else:
                params["agent_name_hint"] = "Claude SkyrimNet Proxy"
            return f"{ISSUER}/api/accounts/authorize?{urlencode(params)}"

    async def token_request(self, session, fields):
        async with session.post(f"{ISSUER}/api/accounts/oauth/token", data=fields,
                                timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                raise HTTPException(401, "ChatGPT authorization expired or was rejected. Sign in again.")
            data = await response.json()
        if not data.get("access_token") or data.get("token_type", "").lower() != "bearer":
            raise HTTPException(502, "Invalid ChatGPT authorization response")
        return data

    async def callback(self, session, params):
        async with self.lock:
            attempt = self.pending.pop(params.get("state", ""), None)
            if not attempt or attempt["expires"] < time.time():
                raise HTTPException(400, "Sign-in expired or state did not match. Start again.")
            if params.get("iss", ISSUER) != ISSUER:
                raise HTTPException(400, "ChatGPT sign-in issuer did not match")
            if params.get("error"):
                raise HTTPException(400, "ChatGPT sign-in was declined or failed")
            client_id = params.get("client_id") or attempt["account_id"]
            if not client_id or client_id == "dynamic_agent_client" or not params.get("code"):
                raise HTTPException(400, "ChatGPT registration is incomplete")
            if attempt["account_id"] and client_id != attempt["account_id"]:
                raise HTTPException(400, "ChatGPT registration did not match")
            data = await self.token_request(session, {
                "grant_type": "authorization_code", "client_id": client_id,
                "code": params["code"], "code_verifier": attempt["verifier"],
                "redirect_uri": attempt["redirect_uri"], "resource": RESOURCE})
            try:
                identity_token = data["id_token"]
                signing_key = await asyncio.to_thread(self.jwks.get_signing_key_from_jwt, identity_token)
                claims = jwt.decode(identity_token, signing_key.key, algorithms=["RS256"],
                                    issuer=ISSUER, audience=client_id, leeway=5,
                                    options={"require": ["sub", "exp", "iat", "nonce"]})
                if not secrets.compare_digest(claims["nonce"], attempt["nonce"]):
                    raise ValueError("Nonce mismatch")
                if not isinstance(claims["sub"], str) or not claims["sub"]:
                    raise ValueError("Missing subject")
                previous = self.state["accounts"].get(client_id)
                if previous and previous["subject"] != claims["sub"]:
                    raise ValueError("Account changed")
            except (jwt.PyJWTError, KeyError, ValueError, TypeError) as error:
                raise HTTPException(400, "Could not validate ChatGPT identity") from error
            record = {**data, "subject": claims["sub"], "email": claims.get("email", ""),
                      "scopes": data.get("scope", "").split(),
                      "welcome_seen": True,
                      "expires_at": time.time() + float(data.get("expires_in", 0))}
            updated = {**self.state, "accounts": {**self.state["accounts"], client_id: record},
                       "active": client_id}
            self.save(updated)
            return PLAN_SCOPE in record["scopes"], not (previous or {}).get("welcome_seen", False)

    async def select(self, account_id):
        async with self.lock:
            if account_id is not None and account_id not in self.state["accounts"]:
                raise HTTPException(400, "Unknown ChatGPT account")
            self.save({**self.state, "active": account_id})

    async def access_token(self, session):
        async with self.lock:
            client_id = self.state["active"]
            record = self.state["accounts"].get(client_id)
            if not record or PLAN_SCOPE not in record.get("scopes", []):
                raise HTTPException(401, "Continue with ChatGPT and enable ChatGPT plan usage first")
            if record["expires_at"] <= time.time() + 60:
                if not record.get("refresh_token"):
                    raise HTTPException(401, "Sign in to ChatGPT again")
                data = await self.token_request(session, {
                    "grant_type": "refresh_token", "client_id": client_id,
                    "refresh_token": record["refresh_token"], "resource": RESOURCE})
                record = {**record, **data,
                          "scopes": data.get("scope", " ".join(record["scopes"])).split(),
                          "expires_at": time.time() + float(data.get("expires_in", 0))}
                self.save({**self.state, "accounts": {**self.state["accounts"], client_id: record}})
            if PLAN_SCOPE not in record["scopes"]:
                raise HTTPException(403, "ChatGPT plan usage permission is missing. Sign in again.")
            return record["access_token"]

    async def models(self, session):
        token = await self.access_token(session)
        async with session.get(f"{RESOURCE}/models", headers={"Authorization": f"Bearer {token}"},
                               timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                raise HTTPException(response.status, "Could not retrieve this account's ChatGPT models")
            data = await response.json()
        return [{"id": "chatgpt/" + entry["slug"], "object": "model", "owned_by": "chatgpt",
                 "name": entry.get("display_name", entry["slug"])}
                for entry in data.get("models", []) if entry.get("visibility") == "list"]


def responses_body(messages, model):
    history = []
    for message in messages:
        if message.role not in ("system", "developer", "user", "assistant"):
            raise HTTPException(400, "Unsupported message role for ChatGPT")
        role = "developer" if message.role == "system" else message.role
        content = message.content
        if isinstance(content, list):
            blocks = []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    text_type = "output_text" if role == "assistant" else "input_text"
                    blocks.append({"type": text_type, "text": block["text"]})
                elif kind == "image_url" and role == "user":
                    image = block["image_url"]
                    blocks.append({"type": "input_image", "image_url": image["url"],
                                   "detail": image.get("detail", "auto")})
                else:
                    raise HTTPException(400, "Unsupported ChatGPT content block")
            content = blocks
        history.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in history):
        raise HTTPException(400, "No user message provided")
    return {"model": model, "input": history,
            "store": False, "stream": True, "reasoning": {"effort": "none"}}


async def response_events(content):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    data = []
    async for chunk in content.iter_any():
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                raw = "\n".join(data)
                data = []
                if raw != "[DONE]":
                    yield json.loads(raw)
    # A complete event is terminated by a blank line. Anything else is truncated.


async def completion_events(response):
    try:
        async for event in response_events(response.content):
            kind = event.get("type")
            if kind == "response.output_text.delta":
                yield {"delta": event["delta"]}
            elif kind == "response.refusal.delta":
                yield {"delta": event["delta"]}
            elif kind == "response.completed":
                yield {"usage": event.get("response", {}).get("usage", {}), "completed": True}
                return
            elif kind in ("response.failed", "response.incomplete", "error"):
                code = event.get("code") or (event.get("response", {}).get("error") or {}).get("code") or kind
                raise HTTPException(502, f"ChatGPT request failed ({code}). Check ChatGPT usage settings or sign in again.")
        raise HTTPException(502, "ChatGPT stream ended before completion")
    finally:
        response.close()


def chat_chunk(completion_id, model, created, delta, finish_reason=None):
    return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


async def complete(provider, session, request, model):
    token = await provider.access_token(session)
    body = responses_body(request.messages, model.removeprefix("chatgpt/"))
    response = await session.post(f"{RESOURCE}/responses", json=body,
                                  headers={"Authorization": f"Bearer {token}"},
                                  timeout=aiohttp.ClientTimeout(total=180, sock_read=90))
    if response.status != 200:
        status = response.status
        response.close()
        raise HTTPException(status, "ChatGPT request rejected. Check model access and ChatGPT usage settings.")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    events = completion_events(response)
    if request.stream:
        async def stream():
            try:
                yield "data: " + json.dumps(chat_chunk(completion_id, model, created, {"role": "assistant"})) + "\n\n"
                async for event in events:
                    if "delta" in event:
                        yield "data: " + json.dumps(chat_chunk(completion_id, model, created, {"content": event["delta"]})) + "\n\n"
                    elif event.get("completed"):
                        yield "data: " + json.dumps(chat_chunk(completion_id, model, created, {}, "stop")) + "\n\n"
                        yield "data: [DONE]\n\n"
            except (HTTPException, aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
                detail = error.detail if isinstance(error, HTTPException) else "ChatGPT stream interrupted"
                yield "data: " + json.dumps({"error": {"message": detail, "type": "upstream_error"}}) + "\n\n"
            finally:
                await events.aclose()
                response.close()
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    text = []
    usage = {}
    async for event in events:
        if "delta" in event:
            text.append(event["delta"])
        else:
            usage = event.get("usage", {})
    return {"id": completion_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                      "prompt_tokens_details": usage.get("input_tokens_details", {}),
                      "completion_tokens": usage.get("output_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0)}}
