"""Model-provider-agnostic LLM client interface.

Every agent depends only on `LLMClient`, never on a concrete HTTP client or
SDK. This is the seam the user asked for: swapping vLLM for a hosted API, or
giving different agents different models/endpoints, means touching
`config.py` and/or `factory.py` only -- never the agent code.

`complete_many` / `complete_structured_many` give callers (e.g. the
Verification Module ranking several hallucinated entities at once) a single
place to fan out concurrent requests bounded by a configurable semaphore,
instead of each agent hand-rolling its own `asyncio.gather`.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


class Prompt(BaseModel):
    """A single system+user prompt pair, the unit of work for batching."""

    system_prompt: str
    user_prompt: str


class LLMClient(ABC):
    """Abstract chat-completion client.

    Concrete implementations adapt a specific wire protocol (an
    OpenAI-compatible HTTP API served by vLLM, etc.) to this interface.
    """

    #: Bounds concurrent in-flight requests issued by the batch helpers below.
    #: Set by concrete implementations from `LLMSettings.max_concurrency`.
    max_concurrency: int = 5

    async def aclose(self) -> None:
        """Release any underlying connection resources. No-op by default."""
        return None

    @abstractmethod
    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion for a single-turn system+user prompt."""
        raise NotImplementedError

    @abstractmethod
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
    ) -> TModel:
        """Return a completion validated/parsed into `response_model`.

        Implementations should use the strongest structured-output mechanism
        available on the backend (JSON schema / guided decoding) and fall
        back to prompt-enforced JSON + pydantic validation.
        """
        raise NotImplementedError

    async def complete_many(self, prompts: list[Prompt]) -> list[str]:
        """Run several independent completions concurrently, capped by a semaphore.

        Order of results matches the order of `prompts`. Intended for
        embarrassingly-parallel workloads such as ranking multiple candidate
        entity replacements or scoring several query variants at once.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run(prompt: Prompt) -> str:
            async with semaphore:
                return await self.complete(
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                )

        return await asyncio.gather(*(_run(p) for p in prompts))

    async def complete_structured_many(
        self,
        prompts: list[Prompt],
        response_model: type[TModel],
    ) -> list[TModel]:
        """Structured-output counterpart of `complete_many`."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run(prompt: Prompt) -> TModel:
            async with semaphore:
                return await self.complete_structured(
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    response_model=response_model,
                )

        return await asyncio.gather(*(_run(p) for p in prompts))
