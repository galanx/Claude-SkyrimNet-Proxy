import json
import unittest
from unittest.mock import patch

import proxy

PROMPT = "PRIVATE-PROMPT-TEXT"


class Content:
    def __init__(self, events):
        self.wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)

    async def iter_any(self):
        for index in range(0, len(self.wire), 7):
            yield self.wire[index:index + 7]


class Response:
    def __init__(self, events):
        self.status = 200
        self.headers = {"request-id": "req_test"}
        self.content = Content(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class Session:
    def __init__(self, events):
        self.events = events

    def post(self, *args, **kwargs):
        return Response(self.events)


class FakeAuth:
    def __init__(self, events):
        self.headers = {}
        self.body_template = {"system": [{"type": "text", "text": "billing"}], "messages": []}
        self.session = Session(events)

    async def ensure_ready(self, **kwargs):
        pass

    def invalidate(self):
        pass


async def stream(events):
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    with patch.object(proxy, "auth", FakeAuth(events)):
        return [chunk async for chunk in
                proxy.call_api_streaming_with_retry(PROMPT, messages, "claude-sonnet-5-5", 16)]


START = {"type": "message_start", "message": {}}
STOP = {"type": "message_stop"}


def text_block(text):
    return [{"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0}]


class ClaudeStreamLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_event_mid_stream_is_logged_with_reason(self):
        events = [START, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}]
        with self.assertLogs("proxy", "WARNING") as logs, self.assertRaises(RuntimeError):
            await stream(events)
        output = "\n".join(logs.output)
        self.assertIn("overloaded_error: Overloaded", output)
        self.assertIn("ended before message_stop", output)
        self.assertIn("req_test", output)
        self.assertNotIn(PROMPT, output)

    async def test_reply_without_text_logs_stop_reason_and_blocks(self):
        events = [START, {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
                  {"type": "message_delta", "delta": {"stop_reason": "refusal"}}, STOP]
        with self.assertLogs("proxy", "WARNING") as logs:
            chunks = await stream(events)
        output = "\n".join(logs.output)
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertIn("no text", output)
        self.assertIn("stop_reason refusal", output)
        self.assertIn("['thinking']", output)
        self.assertNotIn(PROMPT, output)

    async def test_normal_reply_streams_text_and_logs_stop_reason_only(self):
        events = [START, *text_block("Unseen, then."),
                  {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, STOP]
        with self.assertLogs("proxy", "INFO") as logs:
            chunks = await stream(events)
        text = "".join(json.loads(chunk[6:])["choices"][0]["delta"].get("content", "")
                       for chunk in chunks if chunk.startswith("data: {"))
        self.assertEqual(text, "Unseen, then.")
        self.assertFalse([record for record in logs.records if record.levelname == "WARNING"])
        self.assertIn("13 chars", logs.output[-1])
        self.assertIn("stop end_turn", logs.output[-1])
        output = "\n".join(logs.output)
        self.assertNotIn(PROMPT, output)
        self.assertNotIn("Unseen", output)


if __name__ == "__main__":
    unittest.main()
