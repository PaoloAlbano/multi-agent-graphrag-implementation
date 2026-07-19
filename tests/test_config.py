from multigraphrag.config import Settings


def test_default_settings_load_without_env():
    settings = Settings(_env_file=None)
    assert settings.workflow.max_refinement_iterations == 5
    assert settings.memgraph.uri.startswith("bolt://")
