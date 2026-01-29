"""
AgenticSZZ - Temporal Knowledge Graph + Agent-based BIC Identification

This package provides:
- AgenticSZZ: Main pipeline for BIC identification (Blame + TKG + Agent)
- BlameAnalyzer: Git blame extraction and confidence scoring
- SuspiciousCommitTKG: Build minimal TKG containing suspicious commits
- BICSearchAgent: pydantic-ai based search for BIC identification

Usage:
    from agentic_szz import AgenticSZZ

    szz = AgenticSZZ(repo_path)
    result = szz.predict(bfc)
    print(result.predicted_bic)

Architecture (see CLAUDE.md for details):
    Step 1: Build minimal TKG containing suspicious commits
    Step 2: Use pydantic-ai Agent to search and identify the correct BIC
"""

__version__ = "0.2.0"

# Core pipeline
from agentic_szz.pipeline import AgenticSZZ, PipelineResult

# Components (also available from submodules)
from agentic_szz.szz.blame import BlameAnalyzer, BlameResult

__all__ = [
    '__version__',
    # Main API
    'AgenticSZZ',
    'PipelineResult',
    # Components
    'BlameAnalyzer',
    'BlameResult',
]
