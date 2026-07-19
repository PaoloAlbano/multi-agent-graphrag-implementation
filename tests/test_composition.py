from multigraphrag.composition import resolve_agent_llm
from multigraphrag.config import AgentModelSettings, LLMSettings, Settings


def test_resolve_agent_llm_falls_back_to_default():
    settings = Settings(_env_file=None)
    resolved = resolve_agent_llm(settings, "query_generator")
    assert resolved is settings.llm


def test_resolve_agent_llm_uses_override_when_set():
    override = LLMSettings(base_url="http://other:8000/v1", model="other-model")
    settings = Settings(
        _env_file=None,
        agent_models=AgentModelSettings(query_generator=override),
    )
    resolved = resolve_agent_llm(settings, "query_generator")
    assert resolved.base_url == "http://other:8000/v1"
    assert resolved.model == "other-model"
