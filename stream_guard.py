"""Keep proxy SSE responses framed even when an upstream connection fails."""
import asyncio
import json
import logging
from contextlib import suppress

log = logging.getLogger('proxy.stream')

async def guard_sse(source, heartbeat_seconds=15):
    iterator = source.__aiter__()
    pending = None
    completed = False
    try:
        while True:
            pending = asyncio.create_task(iterator.__anext__())
            while not pending.done():
                done, _ = await asyncio.wait({pending}, timeout=heartbeat_seconds)
                if not done:
                    yield ': keep-alive\n\n'
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            # Wrapped generators yield complete SSE frames as strings.
            if any(line.strip() == 'data: [DONE]' for line in chunk.splitlines()):
                completed = True
            yield chunk
        if not completed:
            raise RuntimeError('Upstream ended without a completion marker')
    except asyncio.CancelledError:
        raise
    except Exception as error:
        # Do not put upstream bodies, tokens or private prompts into the error.
        log.warning('SSE upstream interrupted (%s)', type(error).__name__)
        yield 'data: ' + json.dumps({'error': {
            'message': 'Upstream stream interrupted; response may be incomplete. Retry the request.',
            'type': 'upstream_error'}}) + '\n\n'
        yield 'data: [DONE]\n\n'
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pending
        if hasattr(iterator, 'aclose'):
            with suppress(Exception):
                await iterator.aclose()
