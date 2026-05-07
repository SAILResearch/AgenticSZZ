"""
BIC Search using pydantic-ai with Graphiti

This module provides BIC identification using pydantic-ai Agent with Graphiti tools.
The agent interacts ONLY with Graphiti (TKG interface), never with raw git or Neo4j.

Usage:
    from agentic_szz.agent import BICSearchAgent

    agent = BICSearchAgent(repo_path)
    result = await agent.search_async(bfc, graphiti)
    print(result.predicted_bic)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import git
from graphiti_core import Graphiti
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentic_szz.utils.token_tracking import TokenUsage

from .graphiti_tools import (
    BICPrediction as GraphitiBICPrediction,
)
from .graphiti_tools import (
    GraphitiContext,
    list_candidates,
    query_node,
    read_node_content,
    traverse_graph,
)

# =============================================================================
# Agent Configuration
# =============================================================================

SYSTEM_PROMPT = """You are a Bug-Inducing Commit (BIC) identification expert.

## Task
Given a Bug-Fixing Commit (BFC), identify which commit first INTRODUCED the bug.

## Commit Types (Priority Order)
1. **blame**: Code deleted/modified by BFC was last touched by this commit. HIGHEST priority.
2. **blame_ancestor**: Commit BEFORE blame in file history. Lower priority.
3. **bfc_ancestor**: Commit AFTER blame but BEFORE BFC. Lowest priority.

## CRITICAL: Fast-Path Rules (MUST FOLLOW)

### IF "SINGLE BLAME COMMIT" appears:
- STOP IMMEDIATELY. Return that commit.
- Do NOT call any other tools. Do NOT investigate alternatives.

### IF "DOMINANT COMMIT" or "TOP CANDIDATE" with >70% appears:
- This is the LIKELY BIC based on blame line analysis.
- Return it UNLESS you have specific evidence another candidate is better.
- Check other candidates only if uncertain.

### IF "TOP CANDIDATE" is marked:
- This candidate has the highest fitness score based on code overlap.
- Return it unless you find strong evidence it's wrong.

## Multi-Blame Cases Only
If no single/dominant commit and multiple [blame] commits exist:
1. Focus on candidates marked with high fitness scores
2. VERIFY the top candidate using `read_node_content`:
   - "Did this commit ADD the code that the BFC REMOVES?"
   - "Does this commit introduce the LOGIC showing the bug?"
3. If the top blame candidate is just a REFACTORing or IMPORT change:
   - REJECT it (even if fitness is high).
   - Check [ancestor] candidates - they may contain the actual bug introduction.
4. IMPORTANT: [ancestor] commits have lower fitness but may be the true BIC.
   - If blame commits look like cleanup/refactoring, explore ancestors.

## MANDATORY: Minimum Investigation
- You MUST call at least ONE investigation tool before `final_result`
- Exception: SINGLE BLAME fast-path with explicit "STOP!" instruction
- Use `query_node` or `read_node_content` to verify your choice

## Available Tools (TKG Graph Operations)
1. **list_candidates** - Call FIRST. Enumerates nodes with fitness scores.
2. **read_node_content** - CRITICAL for verification. Reads diff stored in node.
3. **traverse_graph** - Finds connected commits via File/Function edges.
4. **query_node** - Access commit node properties and related facts.

## COST EFFICIENCY: Diff Tool Limit
- **LIMIT read_node_content calls to 3 maximum per case** (saves cost without hurting accuracy)
- Prioritize: Check TOP candidate first, then only check others if genuinely uncertain
- If top 2 candidates both look plausible after diff check, pick the higher-fitness one
- Do NOT exhaustively diff all candidates - diminishing returns after 3 calls

## Output
Return the FULL 40-character commit SHA. Explain briefly why.
"""

# Simplified prompt for Agent-only mode (no TKG, just blame candidates)
AGENT_ONLY_PROMPT = """You are a Bug-Inducing Commit (BIC) identification expert.

Given a Bug-Fixing Commit (BFC) and a list of candidate commits from git blame,
identify which candidate commit most likely introduced the bug.

## Your Task
Analyze the provided candidates and select the most likely BIC based on:
1. **Semantic relevance**: Does the commit message relate to the bug?
2. **Temporal proximity**: Earlier commits that touch related code are often suspects
3. **Blame confidence**: Commits blamed for deleted/modified lines are strong candidates

## Response Format
Return your prediction with:
- **predicted_bic**: The full 40-character SHA of your chosen BIC
- **confidence**: 0.0-1.0 confidence score
- **reasoning**: Brief explanation of why you chose this commit

**IMPORTANT**: Always return the FULL 40-character commit SHA in your prediction.
"""


@dataclass
class BICPrediction:
    """Result from BIC search agent - compatible with existing pipeline interface"""

    bfc: str
    predicted_bic: str
    confidence: float
    reasoning: str
    candidates_considered: int
    search_time_ms: float = 0.0
    trajectory: List[Dict[str, Any]] = field(default_factory=list)  # Agent tool call history
    token_usage: "TokenUsage" = None  # Token/cost tracking for RQ3

    @classmethod
    def from_graphiti_prediction(
        cls,
        bfc: str,
        pred: GraphitiBICPrediction,
        candidates_count: int,
        search_time_ms: float,
        trajectory: List[Dict[str, Any]] = None,
        token_usage: TokenUsage = None,
    ) -> "BICPrediction":
        """Convert from graphiti_tools.BICPrediction to this format."""
        return cls(
            bfc=bfc,
            predicted_bic=pred.predicted_bic,
            confidence=pred.confidence,
            reasoning=pred.reasoning,
            candidates_considered=candidates_count,
            search_time_ms=search_time_ms,
            trajectory=trajectory or [],
            token_usage=token_usage or TokenUsage(),
        )


class BICSearchAgent:
    """
    Agent for identifying Bug-Inducing Commits using pydantic-ai with Graphiti.

    This is a complete rewrite that queries the TKG instead of copying the repo.

    Usage:
        agent = BICSearchAgent(repo_path)

        # With TKG (Full pipeline)
        result = await agent.search_async(bfc, graphiti=graphiti)

        # Without TKG (Agent-only ablation)
        result = await agent.search_async(bfc, candidates=candidates)
    """

    def __init__(self, repo_path: str, model: str = None, output_dir: str = None):
        from agentic_szz.config.settings import get_config

        config = get_config()
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.llm_config = config.llm

        # Model setup - support both deepseek and openai format
        self.model_name = model or self._get_model_string()
        self.output_dir = output_dir  # Kept for compatibility, not used in new impl
        # Note: Agents are created fresh per prediction to avoid httpx client
        # lifecycle issues with event loops (fixes "Event loop is closed" errors)

    def _get_model_string(self) -> str:
        """Get model string for pydantic-ai."""
        # DeepSeek uses OpenAI-compatible API with custom base_url
        if "deepseek" in self.llm_config.base_url.lower():
            return f"deepseek:{self.llm_config.model}"
        # OpenAI uses standard format
        if "openai" in self.llm_config.base_url.lower():
            return f"openai:{self.llm_config.model}"
        return self.llm_config.model

    def _model_settings(self) -> dict:
        """Return extra_body for non-thinking mode on DeepSeek models."""
        if "deepseek" in self.llm_config.base_url.lower():
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {}

    def _get_model(self):
        """Get configured pydantic-ai model instance.

        Supports DeepSeek API, OpenAI API, and any OpenAI-compatible local endpoint.
        Local endpoints (vLLM, etc.) require no real API key — "local" is used as a
        dummy value since the OpenAI client library requires a non-empty string.
        """
        base_url = self.llm_config.base_url
        api_key = self.llm_config.api_key or "local"

        if "deepseek" in base_url.lower():
            return OpenAIChatModel(
                model_name=self.llm_config.model,
                provider=OpenAIProvider(
                    base_url=base_url + "/v1",
                    api_key=api_key,
                ),
            )
        if "openai" in base_url.lower():
            return OpenAIChatModel(
                model_name=self.llm_config.model,
                provider=OpenAIProvider(api_key=api_key),
            )
        # Generic OpenAI-compatible endpoint (local vLLM, etc.)
        return OpenAIChatModel(
            model_name=self.llm_config.model,
            provider=OpenAIProvider(
                base_url=base_url,
                api_key=api_key,
            ),
        )

    def _create_tkg_agent(self) -> Agent[GraphitiContext, GraphitiBICPrediction]:
        """Create a fresh TKG agent (with Graphiti tools).

        Creates a new agent instance each call to ensure httpx clients
        are bound to the current event loop, avoiding cleanup errors.
        """
        agent = Agent(
            self._get_model(),
            deps_type=GraphitiContext,
            output_type=GraphitiBICPrediction,
            instructions=SYSTEM_PROMPT,
            retries=2,
        )
        # Register tools for TKG mode (graph-native operations)
        agent.tool(list_candidates)  # Node enumeration
        agent.tool(traverse_graph)  # Edge traversal
        agent.tool(query_node)  # Property access
        agent.tool(read_node_content)  # Content access
        return agent

    def _create_simple_agent(self) -> Agent:
        """Create a fresh simple agent (no tools, for Agent-only mode).

        Creates a new agent instance each call to ensure httpx clients
        are bound to the current event loop, avoiding cleanup errors.
        """
        return Agent(
            self._get_model(),
            output_type=GraphitiBICPrediction,
            instructions=AGENT_ONLY_PROMPT,
            retries=2,
        )

    def _extract_trajectory(self, messages: list) -> List[Dict[str, Any]]:
        """
        Convert pydantic-ai messages to JSON-serializable trajectory.

        Args:
            messages: List of pydantic-ai message objects from result.all_messages()

        Returns:
            List of dicts representing each step in the agent's reasoning
        """
        trajectory = []
        step_num = 0

        for msg in messages:
            msg_type = type(msg).__name__
            step_data = {
                "step": step_num,
                "type": msg_type,
            }

            # Extract content based on message type
            if hasattr(msg, "parts"):
                # ModelRequest or similar - has parts
                parts_data = []
                for part in msg.parts:
                    part_type = type(part).__name__
                    part_info = {"type": part_type}

                    if hasattr(part, "content"):
                        part_info["content"] = str(part.content)

                    if hasattr(part, "tool_name"):
                        part_info["tool_name"] = part.tool_name

                    if hasattr(part, "args"):
                        # Tool call arguments
                        try:
                            part_info["args"] = (
                                part.args.args_json
                                if hasattr(part.args, "args_json")
                                else str(part.args)
                            )
                        except Exception:
                            part_info["args"] = str(part.args)

                    if hasattr(part, "tool_call_id"):
                        part_info["tool_call_id"] = part.tool_call_id

                    parts_data.append(part_info)

                step_data["parts"] = parts_data
                step_num += 1

            elif hasattr(msg, "content"):
                # Simple text message
                step_data["content"] = str(msg.content)
                step_num += 1

            trajectory.append(step_data)

        return trajectory

    async def search_async(
        self,
        bfc: str,
        graphiti: Graphiti = None,
        candidates: List[str] = None,
        bfc_info: str = None,
    ) -> BICPrediction:
        """
        Unified async search for BIC identification.

        Supports two modes for ablation study:
        1. **TKG Mode** (graphiti provided): Agent uses Graphiti tools to query TKG
        2. **Agent-only Mode** (candidates provided, no graphiti): Agent reasons
           over blame candidates directly without TKG/Neo4j

        Args:
            bfc: Bug-fixing commit SHA
            graphiti: Connected Graphiti instance (for TKG mode)
            candidates: List of candidate commit SHAs (for Agent-only mode)
            bfc_info: Optional additional context about the BFC

        Returns:
            BICPrediction with identified BIC and reasoning

        Ablation Mapping:
            - Full pipeline: graphiti=tkg.graphiti
            - Agent-only: candidates=[...], graphiti=None
        """
        # Mode switching based on parameters
        if graphiti is not None:
            return await self._search_with_tkg(bfc, graphiti, bfc_info)
        elif candidates:
            return await self._search_without_tkg(bfc, candidates, bfc_info)
        else:
            # No graphiti and no candidates - return empty result
            return BICPrediction(
                bfc=bfc,
                predicted_bic="",
                confidence=0.0,
                reasoning="No graphiti or candidates provided",
                candidates_considered=0,
            )

    async def _search_with_tkg(
        self,
        bfc: str,
        graphiti: Graphiti,
        bfc_info: str = None,
    ) -> BICPrediction:
        """
        Search using TKG with Graphiti tools (Full pipeline mode).

        Agent uses list_candidates, query_node, read_node_content, traverse_graph tools.
        """
        import time

        start_time = time.time()

        # Create context for tools
        ctx = GraphitiContext(graphiti=graphiti, bfc=bfc)

        # Get BFC info for prompt
        try:
            bfc_commit = self.repo.commit(bfc)
            bfc_files = []
            if bfc_commit.parents:
                diffs = bfc_commit.parents[0].diff(bfc_commit)
                bfc_files = [d.a_path or d.b_path for d in diffs]
        except Exception:
            bfc_files = []

        files_str = ", ".join(bfc_files) if bfc_files else "unknown"

        # Extract functions modified by BFC for comparison
        from agentic_szz.kg.tkg import _extract_functions_from_diff

        try:
            bfc_functions = _extract_functions_from_diff(self.repo, self.repo.commit(bfc))
            funcs_str = ", ".join(bfc_functions) if bfc_functions else "(no function context)"
        except Exception:
            funcs_str = "(no function context)"

        prompt = f"""Identify the bug-inducing commit (BIC) for this bug-fixing commit:

**BFC**: {bfc}
**Files Modified**: {files_str}
**Functions Fixed**: {funcs_str}

Look for candidates that modified the SAME FUNCTIONS as the BFC. Start by calling list_candidates.
{bfc_info or ""}"""

        # Run agent with usage limits to prevent runaway calls
        trajectory = []
        token_usage = TokenUsage(model=self.llm_config.model)
        try:
            from pydantic_ai.usage import UsageLimits

            agent = self._create_tkg_agent()
            result = await agent.run(
                prompt,
                deps=ctx,
                usage_limits=UsageLimits(request_limit=30),
                model_settings=self._model_settings(),
            )
            prediction = result.output

            # Capture agent trajectory from messages
            trajectory = self._extract_trajectory(result.all_messages())

            # Extract token usage for cost tracking (RQ3)
            token_usage = TokenUsage.from_pydantic_ai(result.usage(), model=self.llm_config.model)

        except Exception as e:
            prediction = GraphitiBICPrediction(
                predicted_bic="",
                confidence=0.0,
                reasoning=f"Agent error: {str(e)}",
                alternative_candidates=[],
            )

        search_time = (time.time() - start_time) * 1000

        return BICPrediction.from_graphiti_prediction(
            bfc=bfc,
            pred=prediction,
            candidates_count=0,
            search_time_ms=search_time,
            trajectory=trajectory,
            token_usage=token_usage,
        )

    async def _search_without_tkg(
        self,
        bfc: str,
        candidates: List[str],
        bfc_info: str = None,
    ) -> BICPrediction:
        """
        Search using Agent-only mode (no TKG, no Neo4j).

        Agent receives blame candidates as text and reasons directly.
        Used for RQ2 Agent-only ablation.
        """
        import time

        start_time = time.time()

        # Get BFC info
        try:
            bfc_commit = self.repo.commit(bfc)
            bfc_files = []
            if bfc_commit.parents:
                diffs = bfc_commit.parents[0].diff(bfc_commit)
                bfc_files = [d.a_path or d.b_path for d in diffs]
        except Exception:
            bfc_files = []

        # Build candidate info text (consistent with TKG mode - no limits)
        candidate_info = []
        for i, sha in enumerate(candidates, 1):
            try:
                commit = self.repo.commit(sha)
                # Use full commit message (matches TKG mode which stores full message)
                msg = commit.message.strip()
                candidate_info.append(f"{i}. `{sha}` - {msg}")
            except Exception:
                candidate_info.append(f"{i}. `{sha}` - (unable to get commit info)")

        files_str = ", ".join(bfc_files) if bfc_files else "unknown"
        candidates_str = "\n".join(candidate_info)

        prompt = f"""Identify the bug-inducing commit (BIC) for this bug-fixing commit:

**BFC**: {bfc}
**Files Modified**: {files_str}
{bfc_info or ""}

## Blame Candidates (from git blame on deleted/modified lines):
{candidates_str}

Analyze the candidates above and predict which one is the BIC."""

        # Run simplified agent (no tools, just direct reasoning)
        trajectory = []
        token_usage = TokenUsage(model=self.llm_config.model)
        try:
            from pydantic_ai.usage import UsageLimits

            simple_agent = self._create_simple_agent()
            result = await simple_agent.run(
                prompt,
                usage_limits=UsageLimits(request_limit=3),
                model_settings=self._model_settings(),
            )
            prediction = result.output

            # Capture trajectory
            trajectory = self._extract_trajectory(result.all_messages())

            # Extract token usage for cost tracking (RQ3)
            token_usage = TokenUsage.from_pydantic_ai(result.usage(), model=self.llm_config.model)

        except Exception as e:
            # Fallback to top candidate if agent fails
            prediction = GraphitiBICPrediction(
                predicted_bic=candidates[0] if candidates else "",
                confidence=0.3,
                reasoning=f"Agent-only error, falling back to top blame: {str(e)}",
                alternative_candidates=[],
            )

        search_time = (time.time() - start_time) * 1000

        return BICPrediction.from_graphiti_prediction(
            bfc=bfc,
            pred=prediction,
            candidates_count=len(candidates),
            search_time_ms=search_time,
            trajectory=trajectory,
            token_usage=token_usage,
        )
