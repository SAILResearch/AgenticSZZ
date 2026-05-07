"""
AgenticSZZ Pipeline - TKG + Agent approach for BIC identification

This module provides the main entry point for AgenticSZZ:
1. Run BlameAnalyzer to get initial candidates
2. Build TKG using Graphiti/Neo4j
3. Search with pydantic-ai Agent for BIC identification

Usage:
    from agentic_szz import AgenticSZZ

    szz = AgenticSZZ(repo_path)
    result = szz.predict(bfc)
    print(result.predicted_bic)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import git

if TYPE_CHECKING:
    from agentic_szz.config.settings import AgenticSZZConfig

from agentic_szz.agent.bic_search_agent import BICPrediction, BICSearchAgent
from agentic_szz.kg.tkg import SuspiciousCommitTKG, TKGResult
from agentic_szz.szz.blame import BlameAnalyzer, BlameResult


@dataclass
class PipelineResult:
    """Result from AgenticSZZ pipeline"""

    bfc: str
    predicted_bic: str
    confidence: float
    reasoning: str

    # Stage results
    blame_result: Optional[BlameResult] = None
    tkg_result: Optional[TKGResult] = None
    agent_result: Optional[BICPrediction] = None  # Contains trajectory via agent_result.trajectory

    # Timing metrics
    total_time_ms: float = 0.0
    blame_time_ms: float = 0.0
    tkg_time_ms: float = 0.0
    agent_time_ms: float = 0.0

    # Token/Cost tracking (RQ3)
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    agent_steps: int = 0


class AgenticSZZ:
    """
    Main AgenticSZZ pipeline for BIC identification.

    Two-step approach:
    1. Build TKG with suspicious commits (blame + ancestors + descendants)
    2. Use agent to search TKG and identify BIC

    Usage:
        szz = AgenticSZZ(repo_path)
        result = szz.predict(bfc)
        print(result.predicted_bic)
    """

    def __init__(
        self,
        repo_path: str,
        config: "AgenticSZZConfig" = None,
        db_config: dict = None,
        use_context_fallback: bool = True,
        context_expand_lines: int = 0,
    ):
        """
        Initialize AgenticSZZ pipeline.

        Args:
            repo_path: Path to git repository
            config: Optional AgenticSZZConfig (uses default if None)
            db_config: Optional database config override for worker isolation.
                      Dict with 'uri', 'user', 'password' keys.
                      Used in evaluation to give each worker its own Neo4j instance.
            use_context_fallback: If True, use context fallback for blameless cases.
                         Set to False for pure SZZ blame_only ablation.
            context_expand_lines: Expand ±N lines around each deleted line (experimental).
                         Set to 2 for LLM4SZZ-style context expansion for ancestor detection.
        """
        from agentic_szz.config.settings import Neo4jConfig, get_config

        self.config = config or get_config()

        # Override Neo4j config if provided (for worker isolation)
        if db_config:
            self.config.neo4j = Neo4jConfig.from_dict(db_config)

        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)

        # Initialize components
        # use_context_fallback: for blameless cases (0 deleted lines),
        # blame nearby context lines to get BIC candidates
        # context_expand_lines: expand ±N lines around ALL deleted lines (for ancestor detection)
        self.blame_analyzer = BlameAnalyzer(
            repo_path, 
            use_context_fallback=use_context_fallback,
            context_expand_lines=context_expand_lines,
        )
        self.agent = BICSearchAgent(repo_path)

    def _save_trajectory(
        self,
        bfc: str,
        predicted_bic: str,
        confidence: float,
        reasoning: str,
        trajectory: List[Dict[str, Any]],
        trajectory_dir: str = None,
        dataset_name: str = None,
        case_id: str = None,
        add_timestamp: bool = False,
    ) -> None:
        """
        Save agent trajectory to JSON file.

        Args:
            bfc: Bug-fixing commit SHA
            predicted_bic: Predicted BIC SHA
            confidence: Prediction confidence
            reasoning: Agent reasoning
            trajectory: Agent tool call history
            trajectory_dir: Directory to save (default: results/trajectories)
            dataset_name: Name of the dataset (optional, for legacy compatibility)
            case_id: Case ID in format DATASET_BFC (e.g., 'LINUX_abc123...') - used as filename
            add_timestamp: If True, add timestamp to filename to prevent overwrites
        """
        from datetime import datetime as dt

        # Default directory: results/trajectories relative to project root
        # If trajectory_dir provided, use it directly (caller handles structure)
        if trajectory_dir is None:
            # Try to find project root (where results/ should be)
            project_root = Path(self.repo_path).parent.parent
            if (project_root / "results").exists():
                save_dir = project_root / "results" / "trajectories"
            else:
                # Fallback to current working directory
                save_dir = Path.cwd() / "results" / "trajectories"
        else:
            save_dir = Path(trajectory_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        # Build filename from case_id (format: DATASET_BFC)
        if case_id:
            filename = case_id
        elif dataset_name:
            filename = f"{dataset_name}_{bfc}"
        else:
            filename = f"trajectory_{bfc}"

        if add_timestamp:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"

        trajectory_file = save_dir / f"{filename}.json"

        # Save trajectory
        with open(trajectory_file, "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "bfc": bfc,
                    "predicted_bic": predicted_bic,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "trajectory": trajectory,
                },
                f,
                indent=2,
            )

    def predict(
        self,
        bfc: str,
        skip_tkg: bool = False,
        skip_agent: bool = False,
        cleanup_after: bool = False,
        save_trajectory: bool = True,
        trajectory_dir: str = None,
        dataset_name: str = None,
        case_id: str = None,
        add_timestamp: bool = False,
    ) -> PipelineResult:
        """
        Predict the BIC for a given bug-fixing commit.

        Args:
            bfc: Bug-fixing commit SHA
            skip_tkg: If True, skip TKG building (use blame only)
            skip_agent: If True, skip agent search (return TKG top candidate)
            cleanup_after: If True, clean TKG data after prediction (for evaluation)
            save_trajectory: If True, save agent trajectory to JSON file (default: True)
            trajectory_dir: Directory to save trajectories (default: results/trajectories)
            dataset_name: Name of dataset (optional, for legacy compatibility)
            case_id: Case ID in format DATASET_BFC (e.g., 'LINUX_abc123...')
            add_timestamp: If True, add timestamp to prevent overwrites

        Returns:
            PipelineResult with predicted BIC and detailed results
        """
        # Use new_event_loop + run_until_complete instead of asyncio.run()
        # to properly clean up resources before closing the loop.
        # This prevents "Event loop is closed" errors from httpx cleanup.
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._predict_async(
                    bfc,
                    skip_tkg=skip_tkg,
                    skip_agent=skip_agent,
                    cleanup_after=cleanup_after,
                    save_trajectory=save_trajectory,
                    trajectory_dir=trajectory_dir,
                    dataset_name=dataset_name,
                    case_id=case_id,
                    add_timestamp=add_timestamp,
                )
            )
            return result
        finally:
            # Clean up pending tasks before closing loop
            # Note: Agent caching was removed so httpx clients are properly
            # scoped to this event loop and clean up automatically
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _predict_async(
        self,
        bfc: str,
        skip_tkg: bool = False,
        skip_agent: bool = False,
        cleanup_after: bool = False,
        save_trajectory: bool = True,
        trajectory_dir: str = None,
        dataset_name: str = None,
        case_id: str = None,
        add_timestamp: bool = False,
    ) -> PipelineResult:
        """
        Async implementation of predict.

        Keeps Graphiti connection alive to pass to agent.
        Optionally cleans up TKG data after prediction.
        Saves agent trajectory to JSON if save_trajectory is True.
        """
        import time

        start_time = time.time()

        # Stage 1: Blame Analysis
        blame_start = time.time()
        blame_result = self.blame_analyzer.analyze(bfc)
        blame_time = (time.time() - blame_start) * 1000

        # Stage 2: Build TKG
        tkg_result = None
        tkg_time = 0.0

        # Determine candidates and modified files
        candidates = blame_result.blame_commits
        scores = {c: 1.0 for c in candidates}

        # Get modified files for TKG building
        modified_files = []
        if blame_result.files:
            modified_files = list(blame_result.files.keys())

        if skip_tkg:
            # Return top blame candidate if skipping TKG and agent
            if skip_agent:
                # Use top_candidate (most blamed lines) instead of alphabetical first
                predicted_bic = blame_result.top_candidate or (candidates[0] if candidates else "")

                # Provide informative reasoning for empty predictions
                if not blame_result.is_blameable:
                    reasoning = "Blameless case: BFC only adds new files/code (no deletions/modifications to blame)"
                elif not predicted_bic:
                    reasoning = "No blame commits found (investigation needed)"
                else:
                    reasoning = "Blame top candidate (TKG and agent skipped)"

                total_time = (time.time() - start_time) * 1000
                return PipelineResult(
                    bfc=bfc,
                    predicted_bic=predicted_bic,
                    confidence=blame_result.confidence,
                    reasoning=reasoning,
                    blame_result=blame_result,
                    total_time_ms=total_time,
                    blame_time_ms=blame_time,
                )
            # Skip TKG but run agent - Agent-only ablation (RQ2)
            # Agent reasons over blame candidates without TKG/Neo4j
            agent_start = time.time()
            agent_result = await self.agent.search_async(bfc, candidates=candidates, graphiti=None)
            agent_time = (time.time() - agent_start) * 1000
            total_time = (time.time() - start_time) * 1000

            # Extract token usage for RQ3 cost tracking
            token_usage = getattr(agent_result, "token_usage", None)
            if token_usage is None:
                from agentic_szz.utils.token_tracking import TokenUsage

                token_usage = TokenUsage(model=self.config.llm.model)

            # Save trajectory for Agent-only mode (in separate subfolder)
            trajectory = getattr(agent_result, "trajectory", [])
            if save_trajectory and trajectory:
                agent_only_dir = trajectory_dir or "results/trajectories"
                agent_only_dir = str(Path(agent_only_dir) / "agent_only")
                self._save_trajectory(
                    bfc=bfc,
                    predicted_bic=agent_result.predicted_bic,
                    confidence=agent_result.confidence,
                    reasoning=agent_result.reasoning,
                    trajectory=trajectory,
                    trajectory_dir=agent_only_dir,
                    dataset_name=dataset_name,
                    case_id=case_id,
                    add_timestamp=add_timestamp,
                )

            return PipelineResult(
                bfc=bfc,
                predicted_bic=agent_result.predicted_bic,
                confidence=agent_result.confidence,
                reasoning=agent_result.reasoning,
                blame_result=blame_result,
                agent_result=agent_result,
                total_time_ms=total_time,
                blame_time_ms=blame_time,
                agent_time_ms=agent_time,
                tokens_input=token_usage.input_tokens,
                tokens_output=token_usage.output_tokens,
                tokens_cached=token_usage.cached_tokens,
                cost_usd=token_usage.cost_usd,
                agent_steps=len(agent_result.trajectory) if agent_result.trajectory else 0,
            )

        # Build TKG and keep connection alive for agent
        async with SuspiciousCommitTKG(self.repo_path, neo4j_config=self.config.neo4j) as tkg:
            # Build TKG
            tkg_start = time.time()
            tkg_result = await tkg.build(
                bfc,
                blame_result.blame_commits,
                modified_files=modified_files,
                is_blameless=not blame_result.is_blameable,  # Use blameless thresholds if no deleted lines
                blame_distribution=blame_result.blame_distribution,  # Pass line counts for blame ranking
            )
            tkg_time = (time.time() - tkg_start) * 1000

            # Update candidates from TKG
            candidates = tkg_result.candidates
            scores = tkg_result.scores

            # Return TKG search result if skipping agent (TKG-only ablation)GMT
            if skip_agent:
                # Use TKG graph search without agent reasoning
                predicted_bic, confidence, reasoning = await tkg.search_bic(bfc)
                total_time = (time.time() - start_time) * 1000
                return PipelineResult(
                    bfc=bfc,
                    predicted_bic=predicted_bic,
                    confidence=confidence,
                    reasoning=reasoning,
                    blame_result=blame_result,
                    tkg_result=tkg_result,
                    total_time_ms=total_time,
                    blame_time_ms=blame_time,
                    tkg_time_ms=tkg_time,
                )

            # Stage 3: Agent Search - pass Graphiti instance for TKG queries
            agent_start = time.time()
            agent_result = await self.agent.search_async(
                bfc,
                graphiti=tkg.graphiti,  # Pass live Graphiti connection
            )
            agent_time = (time.time() - agent_start) * 1000

            # Cleanup TKG data if requested (for evaluation runs)
            if cleanup_after:
                await tkg.clean_tkg(group_id=bfc)

        total_time = (time.time() - start_time) * 1000

        # Extract trajectory from agent result
        trajectory = agent_result.trajectory if hasattr(agent_result, "trajectory") else []

        # Save trajectory to JSON file if enabled
        if save_trajectory and trajectory:
            self._save_trajectory(
                bfc=bfc,
                predicted_bic=agent_result.predicted_bic,
                confidence=agent_result.confidence,
                reasoning=agent_result.reasoning,
                trajectory=trajectory,
                trajectory_dir=trajectory_dir,
                dataset_name=dataset_name,
                case_id=case_id,
                add_timestamp=add_timestamp,
            )

        # Extract token usage from agent result (for RQ3 cost tracking)
        token_usage = getattr(agent_result, "token_usage", None)
        if token_usage is None:
            from agentic_szz.utils.token_tracking import TokenUsage

            token_usage = TokenUsage(model=self.config.llm.model)

        return PipelineResult(
            bfc=bfc,
            predicted_bic=agent_result.predicted_bic,
            confidence=agent_result.confidence,
            reasoning=agent_result.reasoning,
            blame_result=blame_result,
            tkg_result=tkg_result,
            agent_result=agent_result,
            total_time_ms=total_time,
            blame_time_ms=blame_time,
            tkg_time_ms=tkg_time,
            agent_time_ms=agent_time,
            tokens_input=token_usage.input_tokens,
            tokens_output=token_usage.output_tokens,
            tokens_cached=token_usage.cached_tokens,
            cost_usd=token_usage.cost_usd,
            agent_steps=len(agent_result.trajectory) if agent_result.trajectory else 0,
        )
