"""
SZZ Core module for AgenticSZZ

Components:
- BlameAnalyzer: Git blame-based candidate identification

Usage:
    from agentic_szz.szz import BlameAnalyzer

    analyzer = BlameAnalyzer(repo_path)
    result = analyzer.analyze(bfc)

For the full pipeline, import from the main package:
    from agentic_szz import AgenticSZZ
"""

from agentic_szz.szz.blame import BlameAnalyzer, BlameResult

__all__ = [
    'BlameAnalyzer',
    'BlameResult',
]
