import asyncio
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException
from chatgpt_provider import (ChatGPTProvider, PLAN_SCOPE, complete,
                             completion_events, protect, responses_body)


class Content:
    def __init__(self, events):
        self.wire = b"".join(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\r\n\r\n"
                             for event in events)

    async def iter_any(self):
        for index in range(0, len(self.wire), 3):
            yield self.wire[index:index + 3]


class Response:
    def __init__(self, events, status=200):
        self.status = status
        self.content = Content(events)
        self.closed = False

    def close(self):
        self.closed = True


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.provider = ChatGPTProvider(Path(self.directory.name) / "accounts.protected")

    async def test_stable_host_and_encrypted_storage(self):
        self.provider.save(self.provider.state)
        restored = ChatGPTProvider(self.provider.path)
        self.assertEqual(restored.state["host_id"], self.provider.state["host_id"])
        self.assertEqual(protect(protect(b"sample"), decrypt=True), b"sample")

    async def test_pkce_and_single_use_state(self):
        url = await self.provider.authorize("http://127.0.0.1:8000/auth/callback")
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["client_id"], ["dynamic_agent_client"])
        self.assertIn(PLAN_SCOPE, params["scope"][0])
        self.assertEqual(params["code_challenge_method"], ["S256"])
        state = params["state"][0]
        with self.assertRaises(HTTPException):
            await self.provider.callback(None, {"state": state, "error": "access_denied"})
        with self.assertRaises(HTTPException):
            await self.provider.callback(None, {"state": state, "code": "replay"})

    async def test_identity_and_scope_validation(self):
        url = await self.provider.authorize("http://127.0.0.1:8000/auth/callback")
        state = parse_qs(urlparse(url).query)["state"][0]
        nonce = self.provider.pending[state]["nonce"]
        data = {"access_token": "sample", "id_token": "identity", "token_type": "Bearer",
                "expires_in": 3600, "scope": "openid profile"}
        self.provider.token_request = AsyncMock(return_value=data)
        with patch.object(self.provider.jwks, "get_signing_key_from_jwt", return_value=SimpleNamespace(key="key")), \
             patch("chatgpt_provider.jwt.decode", return_value={"sub": "person", "nonce": nonce}):
            enabled, first = await self.provider.callback(None, {"state": state, "code": "code", "client_id": "oaiapp_sample"})
        self.assertFalse(enabled)
        self.assertTrue(first)
        with self.assertRaises(HTTPException):
            await self.provider.access_token(None)
        self.assertNotIn("access_token", json.dumps(self.provider.status()))

    async def test_refresh_is_single_flight_and_preserves_rotating_token(self):
        record = {"subject": "person", "scopes": [PLAN_SCOPE], "access_token": "old",
                  "refresh_token": "refresh-old", "expires_at": 0}
        self.provider.state.update(active="client", accounts={"client": record})
        self.provider.token_request = AsyncMock(return_value={
            "access_token": "new", "refresh_token": "refresh-new", "expires_in": 3600})
        results = await asyncio.gather(*(self.provider.access_token(None) for _ in range(8)))
        self.assertEqual(results, ["new"] * 8)
        self.provider.token_request.assert_awaited_once()
        self.assertEqual(self.provider.state["accounts"]["client"]["refresh_token"], "refresh-new")

    def test_roleplay_payload_has_no_agent_or_tools(self):
        messages = [SimpleNamespace(role="system", content="You are Lydia. <action> stays intact."),
                    SimpleNamespace(role="user", content="Good morning."),
                    SimpleNamespace(role="assistant", content="My thane.")]
        body = responses_body(messages, "gpt-6.1-terra")
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertEqual(body["input"][0], {"role": "developer", "content": messages[0].content})
        self.assertEqual(body["input"][2]["content"], "My thane.")
        self.assertEqual(set(body), {"model", "input", "reasoning", "store", "stream"})
        self.assertFalse(body["store"])

    def test_cache_blocks_keep_message_order_for_chatgpt(self):
        messages = [SimpleNamespace(role="system", content=[{
                        "type": "text", "text": "Stable character biography",
                        "cache_control": {"type": "ephemeral"}}]),
                    SimpleNamespace(role="user", content="Earlier dialogue"),
                    SimpleNamespace(role="system", content="Current scene"),
                    SimpleNamespace(role="user", content="Hello")]
        body = responses_body(messages, "gpt-6.1-terra")
        self.assertEqual([message["role"] for message in body["input"]],
                         ["developer", "user", "developer", "user"])
        self.assertEqual(body["input"][0]["content"],
                         [{"type": "input_text", "text": "Stable character biography"}])
        self.assertIn("cache_control", messages[0].content[0])

    async def test_openrouter_keeps_cache_annotations(self):
        import proxy
        from starlette.requests import Request
        content = [{"type": "text", "text": "Stable biography",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        request = proxy.ChatRequest(model="vendor/roleplay", messages=[
            {"role": "system", "content": content},
            {"role": "system", "content": "Current scene"},
            {"role": "user", "content": "Hello"}])
        with patch.object(proxy, "openrouter_api_key", "sample"), \
             patch.object(proxy, "call_openrouter_direct", AsyncMock(return_value="Welcome.")) as call:
            response = await proxy.chat_completions(request, Request({"type": "http", "headers": []}))
        self.assertEqual(response["choices"][0]["message"]["content"], "Welcome.")
        forwarded = call.await_args.args[2]
        self.assertEqual(forwarded[0]["content"], content)
        self.assertEqual(forwarded[1]["content"], "Current scene")

    def test_claude_cache_blocks_survive_body_building(self):
        import proxy
        blocks = [{"type": "text", "text": "Biography", "cache_control": {"type": "ephemeral"}}]
        template = {"system": [{"type": "text", "text": "billing"}], "messages": []}
        with patch.object(proxy.auth, "body_template", template):
            body = proxy._build_api_body(blocks, [{"role": "user", "content": blocks}], "claude-sample")
        self.assertEqual(body["system"][1], blocks[0])
        self.assertEqual(body["messages"][0]["content"], blocks)

    async def test_utf8_stream_and_completion(self):
        response = Response([{"type": "response.output_text.delta", "delta": "Héllo, traveller."},
                             {"type": "response.completed", "response": {"usage": {"total_tokens": 10}}}])
        events = [event async for event in completion_events(response)]
        self.assertEqual(events[0]["delta"], "Héllo, traveller.")
        self.assertTrue(events[-1]["completed"])
        self.assertTrue(response.closed)

    async def test_midstream_failure_is_not_success(self):
        response = Response([{"type": "response.output_text.delta", "delta": "Hello"},
                             {"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}}])
        with self.assertRaises(HTTPException):
            _ = [event async for event in completion_events(response)]
        self.assertTrue(response.closed)

    async def test_truncated_stream_is_not_success(self):
        with self.assertRaises(HTTPException):
            _ = [event async for event in completion_events(Response([]))]

    async def test_http_errors_surface_before_stream_starts(self):
        session = SimpleNamespace(post=AsyncMock(return_value=Response([], 429)))
        provider = SimpleNamespace(access_token=AsyncMock(return_value="sample"))
        request = SimpleNamespace(messages=[SimpleNamespace(role="user", content="Hello")], stream=True)
        with self.assertRaises(HTTPException) as raised:
            await complete(provider, session, request, "chatgpt/gpt-6.1-terra")
        self.assertEqual(raised.exception.status_code, 429)

    async def test_stream_translates_text_and_done(self):
        upstream = Response([{"type": "response.output_text.delta", "delta": "Welcome."},
                             {"type": "response.completed", "response": {}}])
        session = SimpleNamespace(post=AsyncMock(return_value=upstream))
        provider = SimpleNamespace(access_token=AsyncMock(return_value="sample"))
        request = SimpleNamespace(messages=[SimpleNamespace(role="user", content="Hello")], stream=True)
        response = await complete(provider, session, request, "chatgpt/gpt-6.1-terra")
        wire = "".join([part async for part in response.body_iterator])
        self.assertIn('"content": "Welcome."', wire)
        self.assertIn('"finish_reason": "stop"', wire)
        self.assertTrue(wire.endswith("data: [DONE]\n\n"))


if __name__ == "__main__":
    unittest.main()
