"""
Configuration module for AgenticSZZ

Usage:
    from agentic_szz.config import get_config, set_config, AgenticSZZConfig

    config = get_config()
    print(config.llm.model)  # deepseek-chat
"""

from agentic_szz.config.settings import (
    AgenticSZZConfig,
    LLMConfig,
    Neo4jConfig,
    TKGConfig,
    ResourceLimits,
    get_config,
    set_config,
    # Path constants
    PROJECT_ROOT,
    REPOS_DIR,
    DATASET_DIR,
    RESULTS_DIR,
)

__all__ = [
    "AgenticSZZConfig",
    "LLMConfig",
    "Neo4jConfig",
    "TKGConfig",
    "ResourceLimits",
    "get_config",
    "set_config",
    "PROJECT_ROOT",
    "REPOS_DIR",
    "DATASET_DIR",
    "RESULTS_DIR",
]
