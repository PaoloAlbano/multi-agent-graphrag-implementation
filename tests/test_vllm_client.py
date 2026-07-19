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


def test_no_authorization_header_when_api_key_unset():
    client = VllmClient(LLMSettings(api_key=None))
    assert "authorization" not in client._client.headers


def test_authorization_header_sent_when_api_key_set():
    client = VllmClient(LLMSettings(api_key="sk-test"))
    assert client._client.headers["authorization"] == "Bearer sk-test"


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
