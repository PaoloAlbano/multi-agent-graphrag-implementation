import json

import httpx
import pytest
from pydantic import BaseModel

from multigraphrag.config import LLMSettings
from multigraphrag.llm.vllm_client import VllmClient


class _FakeCallLogger:
    def __init__(self):
        self.calls = []

    def log(self, **kwargs):
        self.calls.append(kwargs)


class _Answer(BaseModel):
    value: str


def _chat_completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


@pytest.mark.asyncio
async def test_complete_structured_json_schema_mode_sends_response_format():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response('{"value": "ok"}')

    settings = LLMSettings(structured_output_mode="json_schema")
    client = VllmClient(
        settings, client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    )

    result = await client.complete_structured(system_prompt="sys", user_prompt="user", response_model=_Answer)

    assert result == _Answer(value="ok")
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["payload"]["messages"][1]["content"] == "user"


@pytest.mark.asyncio
async def test_complete_structured_prompt_mode_embeds_schema_and_skips_response_format():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response('{"value": "ok"}')

    settings = LLMSettings(structured_output_mode="prompt")
    client = VllmClient(
        settings, client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    )

    result = await client.complete_structured(system_prompt="sys", user_prompt="user", response_model=_Answer)

    assert result == _Answer(value="ok")
    assert "response_format" not in captured["payload"]
    user_message = captured["payload"]["messages"][1]["content"]
    assert user_message.startswith("user")
    assert "JSON Schema" in user_message


@pytest.mark.asyncio
async def test_extra_body_is_merged_into_every_request():
    """`extra_body` lets callers disable vendor-specific reasoning (e.g. Qwen3's
    chat_template_kwargs.enable_thinking) or pass any other pass-through field,
    without the adapter hardcoding vendor-specific parameter names.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response("OK")

    settings = LLMSettings(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    client = VllmClient(
        settings, client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    )

    await client.complete(system_prompt="sys", user_prompt="user")

    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_reasoning_effort_sent_by_default():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response("OK")

    client = VllmClient(
        LLMSettings(),
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
    )

    await client.complete(system_prompt="sys", user_prompt="user")

    assert captured["payload"]["reasoning_effort"] == "medium"
    assert "chat_template_kwargs" not in captured["payload"]


@pytest.mark.asyncio
async def test_reasoning_disabled_sends_chat_template_kwargs_and_omits_effort():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response("OK")

    client = VllmClient(
        LLMSettings(reasoning_enabled=False),
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
    )

    await client.complete(system_prompt="sys", user_prompt="user")

    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in captured["payload"]


@pytest.mark.asyncio
async def test_reasoning_effort_none_omits_the_field():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response("OK")

    client = VllmClient(
        LLMSettings(reasoning_effort=None),
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
    )

    await client.complete(system_prompt="sys", user_prompt="user")

    assert "reasoning_effort" not in captured["payload"]
    assert "chat_template_kwargs" not in captured["payload"]


@pytest.mark.asyncio
async def test_max_tokens_default_is_sent():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response("OK")

    client = VllmClient(
        LLMSettings(max_tokens=16000),
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
    )

    await client.complete(system_prompt="sys", user_prompt="user")

    assert captured["payload"]["max_tokens"] == 16000


@pytest.mark.asyncio
async def test_max_tokens_none_omits_the_field():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response("OK")

    client = VllmClient(
        LLMSettings(max_tokens=None),
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
    )

    await client.complete(system_prompt="sys", user_prompt="user")

    assert "max_tokens" not in captured["payload"]


@pytest.mark.asyncio
async def test_max_tokens_minus_one_omits_the_field():
    """`-1` is the conventional \"no cap\" sentinel: the field must be omitted
    so the endpoint applies its own default (avoids huge caps that make some
    backends emit empty padding tokens, e.g. Qwen3.5-27B on the gateway)."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _chat_completion_response("OK")

    client = VllmClient(
        LLMSettings(max_tokens=-1),
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
    )

    await client.complete(system_prompt="sys", user_prompt="user")

    assert "max_tokens" not in captured["payload"]


def test_no_authorization_header_when_api_key_unset():
    client = VllmClient(LLMSettings(api_key=None))
    assert "authorization" not in client._client.headers


def test_authorization_header_sent_when_api_key_set():
    client = VllmClient(LLMSettings(api_key="sk-test"))
    assert client._client.headers["authorization"] == "Bearer sk-test"


def test_client_disables_keepalive_connections():
    """The gateway closes keep-alive connections aggressively; reusing a closed
    connection raises a bare TransportError with an empty message. The client
    must not keep connections alive between requests."""
    client = VllmClient(LLMSettings(max_concurrency=5))
    assert client._client._transport._pool._max_keepalive_connections == 0
    assert client._client._transport._pool._max_connections == 5


@pytest.mark.asyncio
async def test_call_logger_records_successful_call():
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response("the answer")

    settings = LLMSettings(model="test-model")
    call_logger = _FakeCallLogger()
    client = VllmClient(
        settings,
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
        call_logger=call_logger,
        agent_name="query_generator",
    )

    await client.complete(system_prompt="sys", user_prompt="user question")

    assert len(call_logger.calls) == 1
    call = call_logger.calls[0]
    assert call["agent"] == "query_generator"
    assert call["model"] == "test-model"
    assert call["system_prompt"] == "sys"
    assert call["user_prompt"] == "user question"
    assert call["response"] == "the answer"
    assert call["error"] is None


@pytest.mark.asyncio
async def test_call_logger_records_failed_call():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "bad request"})

    settings = LLMSettings(model="test-model")
    call_logger = _FakeCallLogger()
    client = VllmClient(
        settings,
        client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)),
        call_logger=call_logger,
        agent_name="query_generator",
    )

    with pytest.raises(Exception):
        await client.complete(system_prompt="sys", user_prompt="user question")

    assert len(call_logger.calls) == 1
    call = call_logger.calls[0]
    assert call["response"] is None
    assert call["error"] is not None


@pytest.mark.asyncio
async def test_null_content_is_retried_and_recovers():
    """Backends can return a 200 OK with `message.content == null` (observed
    with reasoning-capable models on the gateway). That must be retried, not
    crash with an AttributeError -- which would otherwise mark the question as
    failed and skew the eval results."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
        return _chat_completion_response('{"value": "ok"}')

    settings = LLMSettings(structured_output_mode="json_schema")
    client = VllmClient(
        settings, client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    )

    result = await client.complete_structured(system_prompt="sys", user_prompt="user", response_model=_Answer)

    assert result == _Answer(value="ok")
    assert calls == 2


@pytest.mark.asyncio
async def test_persistent_null_content_raises_after_retries():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    settings = LLMSettings(structured_output_mode="json_schema")
    client = VllmClient(
        settings, client=httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    )

    with pytest.raises(Exception):
        await client.complete_structured(system_prompt="sys", user_prompt="user", response_model=_Answer)

    # initial attempt + 1 retry (stop_after_attempt(2))
    assert calls == 2


@pytest.mark.asyncio
async def test_parse_structured_extracts_json_before_runaway_padding():
    """Qwen3.5-27B on the gateway emits a valid JSON object followed by tens of
    thousands of padding whitespace/newlines (runaway generation). The parser
    must extract the first balanced object instead of failing on the whole
    string."""
    padding = "\n  \n" * 5000  # ~15k chars of whitespace
    content = '{"value": "ok"}' + padding

    result = VllmClient._parse_structured(content, _Answer)

    assert result == _Answer(value="ok")


@pytest.mark.asyncio
async def test_parse_structured_recovers_unclosed_object_with_padding():
    """If the model stops mid-object and the rest is padding, the parser closes
    the missing braces and parses the truncated-but-valid object."""
    content = '{"value": "ok"'  # missing closing brace
    padding = "\n\n\n\n\n\n"

    result = VllmClient._parse_structured(content + padding, _Answer)

    assert result == _Answer(value="ok")


@pytest.mark.asyncio
async def test_parse_structured_ignores_trailing_garbage_after_json():
    """Extra non-JSON content after a complete object must not break parsing."""
    content = '{"value": "ok"}\n\nsome trailing text the model kept generating'

    result = VllmClient._parse_structured(content, _Answer)

    assert result == _Answer(value="ok")
