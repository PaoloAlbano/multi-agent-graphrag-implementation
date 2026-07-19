"""vLLM adapter: talks to vLLM's OpenAI-compatible `/chat/completions` endpoint.

Also works unmodified against any other OpenAI-compatible server (a hosted
API, TGI, etc.), since the wire format is the same -- but vLLM is the primary
target this project is built around. No vendor SDK is used on purpose: raw
`httpx` keeps the dependency surface small and makes it trivial to point the
adapter at any endpoint (base_url + api_key), which are provided by the
caller/composition root, never read from config directly by this class.

Structured output has two modes, chosen per-endpoint by
`LLMSettings.structured_output_mode`:

- "json_schema" (default): sends `response_format: json_schema`, which vLLM
  maps onto its guided-decoding backend (outlines/xgrammar/lm-format-enforcer)
  to constrain generation server-side.
- "prompt": for endpoints/models without a guided-decoding backend (or an
  older vLLM build). No `response_format` is sent; the schema is instead
  embedded directly in the prompt and the raw output is parsed/validated
  client-side only, with no server-side guarantee.

Either way, the response text is always parsed and validated against the
pydantic model client-side, so a model that ignores the instruction still
surfaces a clear `LLMResponseParsingError` instead of failing silently.
"""

import json
import logging

import httpx
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multigraphrag.config import LLMSettings
from multigraphrag.llm.base import LLMClient, TModel
from multigraphrag.llm.call_log import CallLogger

logger = logging.getLogger(__name__)


class LLMRequestError(RuntimeError):
    """Raised when the endpoint cannot be reached or returns a non-2xx status."""


class LLMResponseParsingError(RuntimeError):
    """Raised when a structured completion cannot be parsed/validated."""


def _retryable() -> retry:
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((httpx.TransportError, LLMRequestError)),
    )


class VllmClient(LLMClient):
    """LLMClient implementation talking directly to `/chat/completions` via httpx."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: httpx.AsyncClient | None = None,
        call_logger: CallLogger | None = None,
        agent_name: str = "unknown",
    ) -> None:
        self._settings = settings
        self.max_concurrency = settings.max_concurrency
        self._owns_client = client is None
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            headers=headers,
        )
        self._call_logger = call_logger
        self._agent_name = agent_name

    def _log_call(
        self, *, system_prompt: str, user_prompt: str, response: str | None, error: str | None
    ) -> None:
        if self._call_logger is not None:
            self._call_logger.log(
                agent=self._agent_name,
                model=self._settings.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                error=error,
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "VllmClient":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    def _base_payload(self, system_prompt: str, user_prompt: str) -> dict:
        payload: dict = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.temperature,
        }
        if self._settings.max_tokens is not None:
            payload["max_tokens"] = self._settings.max_tokens

        if not self._settings.reasoning_enabled:
            # Qwen3-style "thinking" toggle. Harmless no-op on backends/models
            # that don't recognize chat_template_kwargs.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif self._settings.reasoning_effort:
            # OpenAI/gpt-oss-style hint. Silently ignored by models that don't
            # recognize it (confirmed empirically against Qwen3.5-27B).
            payload["reasoning_effort"] = self._settings.reasoning_effort

        payload.update(self._settings.extra_body)
        return payload

    async def _post_chat_completion(self, payload: dict) -> dict:
        attempt = _retryable()

        @attempt
        async def _call() -> dict:
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                raise LLMRequestError(f"Transport error calling LLM endpoint: {exc}") from exc
            if response.status_code >= 500:
                raise LLMRequestError(f"LLM endpoint returned {response.status_code}: {response.text[:500]}")
            if response.status_code >= 400:
                # 4xx errors are not retried: they indicate a malformed request
                # (bad schema, bad model name) that a retry will not fix.
                raise LLMResponseParsingError(
                    f"LLM endpoint rejected request ({response.status_code}): {response.text[:500]}"
                )
            return response.json()

        return await _call()

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = self._base_payload(system_prompt, user_prompt)
        try:
            data = await self._post_chat_completion(payload)
            content = self._extract_content(data)
        except Exception as exc:
            self._log_call(
                system_prompt=system_prompt, user_prompt=user_prompt, response=None, error=str(exc)
            )
            raise
        self._log_call(system_prompt=system_prompt, user_prompt=user_prompt, response=content, error=None)
        return content

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
    ) -> TModel:
        if self._settings.structured_output_mode == "json_schema":
            sent_user_prompt = user_prompt
            payload = self._base_payload(system_prompt, sent_user_prompt)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            }
        else:
            sent_user_prompt = self._with_schema_instructions(user_prompt, response_model)
            payload = self._base_payload(system_prompt, sent_user_prompt)

        try:
            data = await self._post_chat_completion(payload)
            content = self._extract_content(data)
        except Exception as exc:
            self._log_call(
                system_prompt=system_prompt, user_prompt=sent_user_prompt, response=None, error=str(exc)
            )
            raise
        self._log_call(
            system_prompt=system_prompt, user_prompt=sent_user_prompt, response=content, error=None
        )
        return self._parse_structured(content, response_model)

    @staticmethod
    def _with_schema_instructions(user_prompt: str, response_model: type[TModel]) -> str:
        schema = json.dumps(response_model.model_json_schema())
        return (
            f"{user_prompt}\n\n"
            "Respond with ONLY a single JSON object -- no prose, no markdown code "
            f"fences -- that strictly matches this JSON Schema:\n{schema}"
        )

    @staticmethod
    def _extract_content(data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseParsingError(f"Unexpected completion payload shape: {data}") from exc

    @staticmethod
    def _parse_structured(content: str, response_model: type[TModel]) -> TModel:
        text = content.strip()
        # Some backends wrap JSON output in a markdown fence even when a
        # json_schema response_format was requested; strip it defensively.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseParsingError(f"Model did not return valid JSON: {text[:500]}") from exc
        try:
            return response_model.model_validate(raw)
        except ValidationError as exc:
            raise LLMResponseParsingError(
                f"Model JSON did not match {response_model.__name__}: {exc}"
            ) from exc
