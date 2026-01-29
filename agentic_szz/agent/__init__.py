"""
Agent module for AgenticSZZ

Uses pydantic-ai framework with Graphiti tools for LLM-based BIC identification.

Components:
- BICSearchAgent: Agent wrapper using Graphiti tools
- BICPrediction: Result dataclass
- GraphitiContext: Dependency injection for Graphiti access

Usage:
    from agentic_szz.agent import BICSearchAgent

    agent = BICSearchAgent(repo_path)
    result = await agent.search_async(bfc, graphiti)
"""

from agentic_szz.agent.bic_search_agent import (
    BICSearchAgent,
    BICPrediction,
)

from agentic_szz.agent.graphiti_tools import (
    GraphitiContext,
    CommitInfo,
    BICPrediction as GraphitiBICPrediction,
)

__all__ = [
    # Main exports
    "BICSearchAgent",
    "BICPrediction",
    # Graphiti tools
    "GraphitiContext",
    "CommitInfo",
    "GraphitiBICPrediction",
]
