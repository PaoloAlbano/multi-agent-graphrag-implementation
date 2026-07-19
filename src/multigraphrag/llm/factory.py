"""Builds `LLMClient` instances from settings.

Kept separate from `vllm_client.py` so agents/workflow code depend on this
thin factory (or directly on `LLMClient`) rather than on the concrete
implementation, matching the requested adapter/config decoupling: swapping
the transport later means changing this one function.
"""

from multigraphrag.config import LLMSettings
from multigraphrag.llm.base import LLMClient
from multigraphrag.llm.call_log import CallLogger
from multigraphrag.llm.vllm_client import VllmClient


def build_llm_client(
    settings: LLMSettings,
    *,
    agent_name: str = "unknown",
    call_logger: CallLogger | None = None,
) -> LLMClient:
    """Instantiate the configured LLMClient implementation for one agent.

    `agent_name` and `call_logger` are purely for the optional per-call JSONL
    transcript (`llm/call_log.py`); they don't affect request behavior.
    """
    return VllmClient(settings, agent_name=agent_name, call_logger=call_logger)
