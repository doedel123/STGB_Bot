import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("RAGIE_API_KEY", "test-key")

import services.gemini_client as gemini_client


class _FakeTextBlock:
    type = "output_text"

    def __init__(self, text: str):
        self.text = text


class _FakeMessageItem:
    type = "message"

    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeResponse:
    def __init__(self, response_id: str, status: str, text: str = "", error=None, incomplete=None):
        self.id = response_id
        self.status = status
        self.output = [_FakeMessageItem(text)] if text else []
        self.error = error
        self.incomplete_details = incomplete

    @property
    def output_text(self) -> str:
        return "".join(
            block.text
            for item in self.output
            if item.type == "message"
            for block in item.content
            if block.type == "output_text"
        )


class _FakeSyncResponses:
    def __init__(self, created, retrieved):
        self._created = created
        self._retrieved = list(retrieved)

    def create(self, **kwargs):
        return self._created

    def retrieve(self, response_id, **kwargs):
        return self._retrieved.pop(0)


class _FakeAsyncResponses:
    def __init__(self, created, retrieved):
        self._created = created
        self._retrieved = list(retrieved)

    async def create(self, **kwargs):
        return self._created

    async def retrieve(self, response_id, **kwargs):
        return self._retrieved.pop(0)


class OpenAIBackgroundPollingTest(unittest.TestCase):
    def test_sync_openai_polls_until_completed(self):
        client = SimpleNamespace(
            responses=_FakeSyncResponses(
                created=_FakeResponse("resp_1", "queued"),
                retrieved=[
                    _FakeResponse("resp_1", "in_progress"),
                    _FakeResponse("resp_1", "completed", text="fertige Antwort"),
                ],
            )
        )

        with (
            patch.object(gemini_client, "_get_openai_sync", return_value=client),
            patch.object(gemini_client, "_OPENAI_TIMEOUT", 5),
            patch.object(gemini_client, "_OPENAI_POLL_INTERVAL", 0),
            patch("services.gemini_client.time.sleep", return_value=None),
        ):
            msg = gemini_client._openai_invoke_sync([SimpleNamespace(type="human", content="Hallo")])

        self.assertEqual(msg.content, "fertige Antwort")


class OpenAIBackgroundPollingAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_async_openai_raises_on_failed_terminal_state(self):
        error = SimpleNamespace(code="server_error", message="backend failed")
        client = SimpleNamespace(
            responses=_FakeAsyncResponses(
                created=_FakeResponse("resp_2", "queued"),
                retrieved=[_FakeResponse("resp_2", "failed", error=error)],
            )
        )

        with (
            patch.object(gemini_client, "_get_openai_async", return_value=client),
            patch.object(gemini_client, "_OPENAI_TIMEOUT", 5),
            patch.object(gemini_client, "_OPENAI_POLL_INTERVAL", 0),
            patch("services.gemini_client.asyncio.sleep", new=self._noop_sleep),
        ):
            with self.assertRaisesRegex(RuntimeError, "status=failed"):
                await gemini_client._openai_invoke_async(
                    [SimpleNamespace(type="human", content="Hallo")]
                )

    async def _noop_sleep(self, _delay):
        return None


if __name__ == "__main__":
    unittest.main()
