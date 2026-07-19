"""Application configuration -- plain settings data, nothing else.

This module intentionally contains only pydantic-settings/pydantic models: no
factory functions, no resolution logic. Reading `Settings()` from the
environment and turning it into wired-up objects (LLM clients, agents, the
pipeline) is the composition root's job (`composition.py`), not this
module's -- see that file's docstring for the rationale.

All settings are defined with pydantic-settings so they can be provided via
environment variables (or a `.env` file) in addition to explicit values. Nested
settings use the `__` delimiter, e.g.:

    MULTIGRAPHRAG_LLM__BASE_URL=http://localhost:8000/v1
    MULTIGRAPHRAG_LLM__API_KEY=sk-...
    MULTIGRAPHRAG_LLM__MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
    MULTIGRAPHRAG_MEMGRAPH__URI=bolt://localhost:7687
"""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

StructuredOutputMode = Literal["json_schema", "prompt"]


class LLMSettings(BaseModel):
    """Connection details for an OpenAI-compatible LLM endpoint (e.g. vLLM).

    Kept independent from any single agent so the same settings object can be
    reused as a default, then overridden per-agent in `AgentModelSettings`.
    """

    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible base URL, e.g. a vLLM OpenAI server.",
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Bearer token sent as `Authorization: Bearer <api_key>`. Optional: many "
            "local vLLM deployments require no authentication at all, in which case "
            "leave this unset and no Authorization header is sent."
        ),
    )
    model: str = Field(
        default="default",
        description="Model name as registered on the serving endpoint.",
    )
    temperature: float = 0.0
    max_tokens: int | None = Field(
        default=16000,
        description=(
            "Max completion tokens per call. Reasoning models spend a variable, "
            "often large share of this on chain-of-thought before ever writing "
            "the final `content` (observed: 100-350+ reasoning tokens for a "
            "trivial prompt), so a low/unset cap risks silently truncating the "
            "structured JSON output every agent depends on. Set to None to let "
            "the endpoint apply its own default instead."
        ),
    )
    reasoning_enabled: bool = Field(
        default=True,
        description=(
            "Whether the model should be allowed to use its default chain-of-"
            "thought/'thinking' behavior. False actively suppresses it via "
            "chat_template_kwargs.enable_thinking=false (the mechanism for "
            "Qwen3-style 'thinking' models served on vLLM) -- cutting completion "
            "tokens drastically (observed: ~101 -> ~2 for a trivial prompt) since "
            "agents only ever read the final `content`, never `reasoning_content`. "
            "True (the default) leaves the endpoint's own default behavior alone "
            "and, if `reasoning_effort` is set, forwards it as a hint."
        ),
    )
    reasoning_effort: str | None = Field(
        default="medium",
        description=(
            "Forwarded verbatim as a top-level `reasoning_effort` field when "
            "`reasoning_enabled` is True (the OpenAI/gpt-oss-style knob, e.g. "
            "'low'/'medium'/'high'). Ignored (harmlessly, confirmed empirically) "
            "by endpoints/models that don't recognize the field, e.g. Qwen3. Set "
            "to None to omit it entirely."
        ),
    )
    request_timeout: float = 120.0
    max_retries: int = 3
    max_concurrency: int = Field(
        default=5,
        description=(
            "Max number of in-flight HTTP requests to this endpoint when a caller "
            "batches several completions (e.g. verifying many entities at once). "
            "Bounds an asyncio.Semaphore used around httpx calls."
        ),
    )
    structured_output_mode: StructuredOutputMode = Field(
        default="json_schema",
        description=(
            "How agent responses are constrained to a schema. 'json_schema' sends "
            "response_format=json_schema, relying on the endpoint's guided-decoding "
            "backend (e.g. vLLM with outlines/xgrammar) to enforce it server-side. "
            "'prompt' never sends response_format: the schema is instead embedded in "
            "the prompt and the raw JSON output is validated client-side only. Use "
            "'prompt' when the serving endpoint/model does not support structured "
            "outputs (older vLLM builds, no guided-decoding backend installed, or a "
            "model without JSON-mode support)."
        ),
    )
    extra_body: dict = Field(
        default_factory=dict,
        description=(
            "Extra top-level fields merged verbatim into every chat-completion "
            "request payload, applied last so they override `reasoning_enabled`/"
            "`reasoning_effort`/anything else computed here. An escape hatch for "
            "vendor-specific parameters that don't warrant a dedicated setting."
        ),
    )


class AgentModelSettings(BaseModel):
    """Per-agent model overrides.

    Any field left as None falls back to `Settings.llm` (the default model).
    This lets you, e.g., run the Query Generator on a strong coder model while
    keeping cheaper models for evaluation/extraction.
    """

    query_generator: LLMSettings | None = None
    query_evaluator: LLMSettings | None = None
    entity_extractor: LLMSettings | None = None
    verification_ranker: LLMSettings | None = None
    instructions_generator: LLMSettings | None = None
    feedback_aggregator: LLMSettings | None = None
    interpreter: LLMSettings | None = None


class MemgraphSettings(BaseModel):
    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="")
    password: str = Field(default="")
    database: str | None = Field(default=None)
    encrypted: bool = Field(default=False)
    connection_timeout: float = 30.0


class WorkflowSettings(BaseModel):
    max_refinement_iterations: int = Field(
        default=5,
        description=(
            "Max total Query Generator calls per question (the initial attempt "
            "plus subsequent corrections). The paper's Algorithm 1 runs one "
            "unconditional initial generation, then loops 'while status != "
            "Accept and t <= 4', i.e. up to 4 further corrections -- 5 total "
            "generate calls -- even though its prose describes this as 'a "
            "maximum of four iterations'. Set to 5 to match the pseudocode "
            "exactly; set to 4 to match the prose description instead."
        ),
    )
    max_query_result_rows: int = Field(
        default=200,
        description="Truncate large result sets before handing them to the LLM.",
    )
    levenshtein_top_k: int = Field(
        default=5,
        description="Number of fuzzy-match candidates surfaced per hallucinated entity.",
    )
    levenshtein_min_score: float = Field(
        default=60.0,
        description="Minimum normalized similarity (0-100) to suggest a candidate.",
    )


class EvaluationSettings(BaseModel):
    """Config for the CypherBench evaluation harness itself (not the agent workflow)."""

    concurrency: int = Field(
        default=1,
        description=(
            "Max number of CypherBench questions processed concurrently per "
            "domain/mode. Different questions against the same graph are fully "
            "independent, so this bounds an asyncio.Semaphore around parallel "
            "pipeline.ask()/SinglePassRunner.ask() calls to speed up evaluation "
            "runs. Keep in mind each LLM agent's own `LLMSettings.max_concurrency` "
            "still caps in-flight requests to that specific endpoint."
        ),
    )
    judge: LLMSettings | None = Field(
        default=None,
        description=(
            "Optional LLM used as an LLM-as-a-judge to score whether a pipeline "
            "answer is semantically equivalent to CypherBench's gold answer -- "
            "the paper's own scoring methodology (a dedicated judge model, e.g. "
            "GigaChat 2 MAX, distinct from the models under test). Left unset "
            "(the default), scoring falls back to the deterministic value-overlap "
            "heuristic in `evaluation/matching.py`. Enabled per-run via "
            "`multigraphrag cypherbench evaluate --use-judge`."
        ),
    )


class Settings(BaseSettings):
    """Root settings object. Instantiate via `Settings()` to read env/`.env`."""

    model_config = SettingsConfigDict(
        env_prefix="MULTIGRAPHRAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent_models: AgentModelSettings = Field(default_factory=AgentModelSettings)
    memgraph: MemgraphSettings = Field(default_factory=MemgraphSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
