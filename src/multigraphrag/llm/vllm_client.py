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
        stop=stop_after_attempt(2),
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
            # The gateway closes keep-alive connections aggressively (observed
            # after long/runaway generations). Reusing a closed connection
            # raises a bare TransportError with an empty message that stalls
            # the pipeline until the 300s timeout. Opening a fresh connection
            # per request avoids this entirely.
            limits=httpx.Limits(
                max_keepalive_connections=0,
                max_connections=settings.max_concurrency or 10,
            ),
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
        # `None` and `-1` both mean "let the endpoint apply its own default":
        # either way the field is omitted from the payload. `-1` is the
        # conventional sentinel (e.g. vLLM/OpenAI CLI) for "no cap", and some
        # backends misbehave when max_tokens is huge (Qwen3.5-27B on the
        # gateway was observed emitting lots of empty tokens past the real
        # answer, which can trip long/empty responses).
        if self._settings.max_tokens not in (None, -1):
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

    async def _post_chat_completion(self, payload: dict) -> str:
        """POST /chat/completions and return the assistant message content.

        Extracting the content *inside* the retry is important: some backends
        return a 200 OK with `message.content == null` (observed with
        reasoning-capable models on the gateway). That is a transient,
        retryable condition -- not a malformed request -- so we surface it as
        `LLMRequestError` from within `_call()` where tenacity can retry it.
        """
        attempt = _retryable()

        @attempt
        async def _call() -> str:
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                # Log the *type* and repr of the underlying exception: some
                # TransportError subclasses have an empty str() (observed:
                # `Transport error calling LLM endpoint: ` with nothing after
                # the colon), which hides whether it was a ConnectError,
                # ReadTimeout, RemoteProtocolError, etc.
                logger.warning(
                    "LLM transport failure: type=%s repr=%r",
                    type(exc).__name__,
                    repr(exc),
                )
                raise LLMRequestError(f"Transport error calling LLM endpoint: {exc}") from exc
            if response.status_code >= 500:
                raise LLMRequestError(f"LLM endpoint returned {response.status_code}: {response.text[:500]}")
            if response.status_code >= 400:
                # 4xx errors are not retried: they indicate a malformed request
                # (bad schema, bad model name) that a retry will not fix.
                raise LLMResponseParsingError(
                    f"LLM endpoint rejected request ({response.status_code}): {response.text[:500]}"
                )
            try:
                content = self._extract_content(response.json())
            except LLMResponseParsingError as exc:
                # e.g. `content: null` under a 200 OK -- transient on some
                # backends, so retry instead of failing the question.
                raise LLMRequestError(str(exc)) from exc
            return content

        return await _call()

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = self._base_payload(system_prompt, user_prompt)
        try:
            content = await self._post_chat_completion(payload)
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
            content = await self._post_chat_completion(payload)
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
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseParsingError(f"Unexpected completion payload shape: {data}") from exc
        if content is None:
            raise LLMResponseParsingError(
                "LLM endpoint returned empty content (message.content is null)"
            )
        return content

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
        raw = VllmClient._extract_first_json_object(text)
        try:
            return response_model.model_validate(raw)
        except ValidationError as exc:
            raise LLMResponseParsingError(
                f"Model JSON did not match {response_model.__name__}: {exc}"
            ) from exc

    @staticmethod
    def _extract_first_json_object(text: str) -> object:
        """Parse the first valid JSON object in ``text``.

        Some models (observed: Qwen3.5-27B on the gateway) emit a valid JSON
        object and then keep generating padding whitespace/newlines past the
        closing brace -- a runaway-generation artifact that can be tens of
        thousands of characters long. ``json.loads`` on the whole string fails
        in that case (e.g. `Expecting ',' delimiter`), so we:

        1. try a plain parse of the trimmed text first (fast path);
        2. otherwise scan for the first ``{`` and extract the *balanced*
           object, ignoring trailing garbage;
        3. if the object is unbalanced (missing its closing brace), try
           appending the minimal closing brace(s) so a truncated-but-valid
           object still parses.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        if start == -1:
            raise LLMResponseParsingError(f"Model did not return valid JSON: {text[:500]}")

        # Scan the substring from the first `{` tracking brace depth, while
        # skipping braces inside string literals.
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # not a clean object; fall through to recovery

        # Unbalanced: the model stopped mid-object (runaway padding). Try
        # closing the object with the missing braces and re-parse.
        if depth > 0:
            candidate = text[start:] + "}" * depth
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise LLMResponseParsingError(f"Model did not return valid JSON: {text[:500]}")
