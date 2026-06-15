"""SkyRL + mini-swe-agent integration for true GRPO SWE Agent RL."""

__all__ = ["MiniSWEGeneratorConfig", "MiniSweAgentGenerator"]


def __getattr__(name: str):
    if name in __all__:
        from integrations.skyrl_miniswe.generator import MiniSWEGeneratorConfig, MiniSweAgentGenerator

        return {"MiniSWEGeneratorConfig": MiniSWEGeneratorConfig, "MiniSweAgentGenerator": MiniSweAgentGenerator}[
            name
        ]
    raise AttributeError(name)
