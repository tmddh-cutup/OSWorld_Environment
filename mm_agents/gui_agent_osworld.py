from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_GUI_AGENT_REPO = r"D:\SeungO\gui_agent\guiAgent_OSworld_benchmark"

gui_agent_repo = os.environ.get(
    "GUI_AGENT_BENCHMARK_REPO",
    DEFAULT_GUI_AGENT_REPO,
)

gui_agent_repo_path = Path(gui_agent_repo).resolve()

if not gui_agent_repo_path.exists():
    raise FileNotFoundError(
        f"GUI agent benchmark repo not found: {gui_agent_repo_path}"
    )

if str(gui_agent_repo_path) not in sys.path:
    sys.path.insert(0, str(gui_agent_repo_path))

try:
    from gui_agent.osworld.agent_wrapper import GUIAgentOSWorld
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Failed to import legacy GUIAgentOSWorld. "
        f"Expected gui_agent package under: {gui_agent_repo_path}. "
        "Check GUI_AGENT_BENCHMARK_REPO and PYTHONPATH."
    ) from exc


__all__ = ["GUIAgentOSWorld"]