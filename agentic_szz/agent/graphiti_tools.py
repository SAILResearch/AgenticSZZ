"""
Graphiti Tools for BIC Search Agent

This module provides pydantic-ai tools that wrap Graphiti APIs for BIC identification.
The agent interacts ONLY with Graphiti (TKG interface), never with raw git or Neo4j.

Tools (Graph-Native Operations):
- list_candidates: Enumerate all commit nodes in BFC subgraph with fitness scores
- query_node: Query properties of a specific commit node
- read_node_content: Read code diff content stored in a commit node
- traverse_graph: Traverse MODIFIES_* edges to find structurally related commits

Usage:
    from agentic_szz.agent.graphiti_tools import (
        GraphitiContext, BICPrediction
    )

    ctx = GraphitiContext(graphiti=graphiti, bfc=bfc_sha)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from graphiti_core import Graphiti


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class GraphitiContext:
    """Dependencies passed to agent tools via pydantic-ai's dependency injection."""

    graphiti: Graphiti
    bfc: str  # BFC SHA, used as group_id for filtering


class CommitInfo(BaseModel):
    """Information about a commit returned by tools."""

    sha: str = Field(description="Commit SHA (full or partial)")
    message: str = Field(description="Commit message (first line)")
    files: List[str] = Field(default_factory=list, description="Files modified")
    functions: List[str] = Field(default_factory=list, description="Functions/methods modified")
    timestamp: str = Field(description="Commit timestamp")
    commit_type: str = Field(description="bfc, blame, blame_ancestor, or bfc_ancestor")
    line_count: int = Field(default=0, description="Number of deleted/modified lines blamed on this commit")


class BICPrediction(BaseModel):
    """Agent's final BIC prediction output."""

    predicted_bic: str = Field(description="SHA of the predicted bug-inducing commit")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(description="Explanation for the prediction")
    alternative_candidates: List[str] = Field(
        default_factory=list, description="Other commits that could be the BIC"
    )


# =============================================================================
# Helper Functions
# =============================================================================


def parse_episode_to_commit(episode) -> CommitInfo:
    """Parse a Graphiti EpisodicNode to CommitInfo."""
    # Episode name format: "{commit_type}_{sha}" where commit_type may contain underscores
    # (e.g., "blame_ancestor_abcd1234...")
    # The SHA should be extractable from the episode content which is more reliable

    content = episode.content or ""
    message = ""
    files = []
    functions = []
    sha = ""
    line_count = 0

    # Parse content - Message field may be multi-line, so we need special handling
    # Format: "SHA: xxx\nModified files: xxx\nFunctions changed: xxx\nBlame lines: N\nMessage: xxx"
    lines = content.split("\n")
    message_start_idx = -1

    for i, line in enumerate(lines):
        if line.startswith("SHA:"):
            sha = line.replace("SHA:", "").strip()
        elif line.startswith("Modified files:"):
            files_str = line.replace("Modified files:", "").strip()
            files = [f.strip() for f in files_str.split(",") if f.strip()]
        elif line.startswith("Functions changed:"):
            funcs_str = line.replace("Functions changed:", "").strip()
            if funcs_str and funcs_str != "(no function context)":
                functions = [f.strip() for f in funcs_str.split(",") if f.strip()]
        elif line.startswith("Blame lines:"):
            try:
                line_count = int(line.replace("Blame lines:", "").strip())
            except ValueError:
                line_count = 0
        elif line.startswith("Message:"):
            # Message starts here and continues to the end
            message = line.replace("Message:", "").strip()
            message_start_idx = i

    # If message was found, concatenate remaining lines
    if message_start_idx >= 0 and message_start_idx + 1 < len(lines):
        remaining_lines = lines[message_start_idx + 1:]
        if remaining_lines:
            message = message + "\n" + "\n".join(remaining_lines)

    # Fallback: get SHA from episode name (name IS the SHA now)
    if not sha:
        # Episode name is now just the full 40-char SHA
        sha = episode.name

    # Extract commit type from source_description
    if hasattr(episode, 'source_description') and episode.source_description:
        commit_type = episode.source_description
    else:
        commit_type = "unknown"

    # Format timestamp
    ts = (
        episode.valid_at if hasattr(episode, "valid_at") else datetime.now(timezone.utc)
    )
    timestamp_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "unknown"

    return CommitInfo(
        sha=sha,
        message=message or "(no message)",
        files=files,
        functions=functions,
        timestamp=timestamp_str,
        commit_type=commit_type,
        line_count=line_count,
    )

def format_candidates_for_display(candidates: List[CommitInfo], is_multi_blame: bool = False, limit: int = None) -> str:
    """Format candidates list for agent display with fitness scores.
    
    Candidates are sorted by computed fitness score to help agent
    prioritize the most likely BIC.
    
    Args:
        candidates: List of CommitInfo objects
        is_multi_blame: Whether this is a multi-blame case
        limit: Maximum number of candidates to display (None = no limit)
    
    Fitness score formula:
    - Base score by commit type:
      - blame: 1.0 (highest priority - git blame provides causal analysis)
      - blame_ancestor: 0.6 (lower priority)
      - bfc_ancestor: 0.3 (lowest priority)
    
    With same type scores, blame commits are ordered by recency (chronological
    order from blame analysis) which serves as tiebreaker.
    """
    if not candidates:
        return "No candidates found."

    # Compute fitness score for each candidate
    scored_candidates = []
    for c in candidates:
        # Base score by type (empirically validated)
        # Blame commits ranked by recency (chronological order from blame analysis)
        # No line-count bonus: analysis showed it overrides correct blame answers
        type_scores = {'blame': 1.0, 'blame_ancestor': 0.6, 'bfc_ancestor': 0.3}
        score = type_scores.get(c.commit_type, 0.1)
        
        scored_candidates.append((c, score))
    
    # Sort by fitness score (highest first)
    scored_candidates.sort(key=lambda x: -x[1])
    
    # Apply limit if specified
    if limit is not None and limit < len(scored_candidates):
        scored_candidates = scored_candidates[:limit]

    lines = []
    
    # Check if there's a clear top candidate (fitness gap > 0.1)
    has_clear_top = len(scored_candidates) == 1 or (
        len(scored_candidates) > 1 and 
        scored_candidates[0][1] - scored_candidates[1][1] > 0.1
    )
    
    for i, (c, score) in enumerate(scored_candidates, 1):
        files_str = ", ".join(c.files) if c.files else "no files"
        funcs_str = ", ".join(c.functions) if c.functions else "(no function context)"
        line_count_str = f"   Blame lines: {c.line_count}\n" if c.line_count > 0 else ""
        
        # Only mark top candidate when there's a clear fitness gap
        top_marker = ">>> TOP CANDIDATE <<<\n" if i == 1 and has_clear_top else ""
        fitness_str = f"   Fitness: {score:.2f}"
        
        lines.append(
            f"{top_marker}"
            f"{i}. [{c.commit_type}] {c.sha}\n"
            f"{fitness_str}\n"
            f"   Date: {c.timestamp}\n"
            f"   Message: {c.message}\n"
            f"   Files: {files_str}\n"
            f"   Functions: {funcs_str}\n"
            f"{line_count_str}"
        )
    return "\n".join(lines)


def extract_shas_from_search_results(results) -> List[str]:
    """Extract commit SHAs from Graphiti SearchResults."""
    shas = set()

    # From episodes - name IS the SHA now
    if hasattr(results, "episodes"):
        for ep in results.episodes:
            shas.add(ep.name)  # Episode name is just the SHA

    # From edges (facts about commits)
    if hasattr(results, "edges"):
        for edge in results.edges:
            # Try to extract SHA from edge fact
            fact = edge.fact if hasattr(edge, "fact") else str(edge)
            # Look for patterns like commit SHAs (40-char hex)
            import re

            sha_matches = re.findall(r"\b[a-f0-9]{40}\b", fact)
            shas.update(sha_matches)

    return list(shas)  # Return all found SHAs


# =============================================================================
# Agent Tools (wrap Graphiti APIs)
# =============================================================================


async def list_candidates(ctx: RunContext[GraphitiContext]) -> str:
    """
    List top BIC candidates with blame statistics overview.
    
    Call this FIRST to understand:
    1. Blame landscape (simple vs complex case)
    2. Top candidates by fitness score (limited to most relevant)
    
    Blame statistics indicate:
    - Single blame commit → High confidence, likely BIC
    - Dominant commit (>70% lines) → Strong candidate
    - Multiple balanced commits → Complex case, use similarity tools
    
    Returns:
        Formatted list of top-K candidates sorted by fitness score.
    """
    from agentic_szz.config.settings import get_config
    candidate_limit = get_config().tkg.candidate_limit
    
    # PART 1: Get blame statistics from BFC episode
    blame_query = """
    MATCH (e:Episodic {name: $bfc, source_description: 'bfc'})
    RETURN e.blame_total_commits AS total_commits,
           e.blame_single_commit AS single_commit,
           e.blame_dominant_commit AS dominant_commit,
           e.blame_total_lines AS total_lines
    """
    blame_result = await ctx.deps.graphiti.driver.execute_query(
        blame_query, bfc=ctx.deps.bfc
    )
    
    # PART 2: Get all candidates (exclude BFC) - need this first to find blame SHA
    episodes = await ctx.deps.graphiti.retrieve_episodes(
        reference_time=datetime.now(timezone.utc),
        group_ids=[ctx.deps.bfc],
        last_n=200,  # Retrieve more than limit to ensure we get all for scoring
    )

    # Filter out the BFC itself (BFC cannot be BIC)
    candidates = []
    blame_commits = []
    for ep in episodes:
        commit_info = parse_episode_to_commit(ep)
        if commit_info.commit_type != "bfc":
            candidates.append(commit_info)
            if commit_info.commit_type == "blame":
                blame_commits.append(commit_info)
    
    output = "=== FAST-PATH DECISION ===\n"
    fast_path_used = False
    
    if blame_result.records and len(blame_result.records) > 0:
        rec = blame_result.records[0]
        total = rec.get('total_commits', 0) or 0
        single = rec.get('single_commit', False) or False
        dominant = rec.get('dominant_commit')
        
        if single and len(blame_commits) == 1:
            # SINGLE BLAME - show exact SHA
            sha = blame_commits[0].sha
            output += f">>> SINGLE BLAME COMMIT: {sha} <<<\n"
            output += f">>> STOP! Return this SHA as BIC <<<\n"
            fast_path_used = True
        elif dominant:
            # DOMINANT COMMIT - softer guidance (50/50 correct in analysis)
            # Keep ranking signal but allow agent to explore alternatives
            output += f">>> TOP CANDIDATE: {dominant} (>70% of blamed lines) <<<\n"
            output += f">>> Likely BIC. Review other candidates below if unsure. <<<\n"
            # Don't set fast_path_used - let agent see all candidates
        elif total > 1:
            output += f"Multiple blame commits ({total}). Use TOP CANDIDATE below.\n"
            output += f"NOTE: If top blame seems like refactoring/cleanup, check [ancestor] commits.\n"
        else:
            output += "No fast-path available.\n"
    else:
        output += "Blameless case (only additions). Use TOP CANDIDATE below.\n"
    
    # Apply candidate limit - show top K candidates to agent
    total_candidates = len(candidates)
    output += f"\n=== TOP {min(candidate_limit, total_candidates)} CANDIDATES (of {total_candidates} total, sorted by fitness) ===\n"
    
    # Determine if this is a multi-blame case (need differentiation via line_count bonus)
    is_multi_blame = len(blame_commits) > 1 and not fast_path_used
    output += format_candidates_for_display(candidates, is_multi_blame=is_multi_blame, limit=candidate_limit)
    return output


async def query_node(ctx: RunContext[GraphitiContext], sha: str) -> str:
    """
    Query properties of a specific commit node in the TKG.

    Use this to access detailed properties stored in a commit node:
    - Commit message, timestamp, modified files/functions
    - Related semantic facts extracted during TKG construction

    Args:
        sha: Full or partial commit SHA (node identifier)

    Returns:
        Node properties including message, files, and related graph facts.
    """
    # First, search all episodes to find the one matching this SHA
    episodes = await ctx.deps.graphiti.retrieve_episodes(
        reference_time=datetime.now(timezone.utc),
        group_ids=[ctx.deps.bfc],
        last_n=100,
    )

    # Find episode matching the SHA
    matching_episode = None
    for ep in episodes:
        # Episode name is the SHA, or check content for partial match
        if ep.name == sha or sha in ep.name or (hasattr(ep, "content") and sha in ep.content):
            matching_episode = ep
            break

    if not matching_episode:
        return f"No information found for commit {sha}"

    # Parse and format the episode content
    commit_info = parse_episode_to_commit(matching_episode)

    lines = [
        f"Details for commit {sha}:",
        f"  Type: {commit_info.commit_type}",
        f"  Full SHA: {commit_info.sha}",
        f"  Date: {commit_info.timestamp}",
        f"  Message: {commit_info.message}",
        f"  Files: {', '.join(commit_info.files)}",
    ]

    # Show code overlap using our MODIFIES edges (instead of Entity edges)
    try:
        # Find other commits that modified the same functions
        overlap_query = """
        MATCH (e:Episodic {name: $episode_name})-[:MODIFIES_FUNCTION]->(f:Function)
        <-[:MODIFIES_FUNCTION]-(other:Episodic)
        WHERE e <> other AND other.group_id = $group_id
        RETURN collect(DISTINCT f.name) AS shared_functions,
               collect(DISTINCT other.source_description) AS commit_types
        """
        
        result = await ctx.deps.graphiti.driver.execute_query(
            overlap_query,
            episode_name=matching_episode.name,
            group_id=ctx.deps.bfc,
        )
        
        if result.records and result.records[0]["shared_functions"]:
            shared_funcs = result.records[0]["shared_functions"]
            lines.append(f"\nCode overlap with BFC:")
            lines.append(f"  Shared functions: {', '.join(shared_funcs)}")
    except Exception:
        pass  # Ignore query errors

    return "\n".join(lines)


async def read_node_content(ctx: RunContext[GraphitiContext], sha: str) -> str:
    """
    Read the code diff content stored in a commit node.
    
    CRITICAL for causality verification:
    - Check if lines ADDED by this commit match what the BFC REMOVES
    - Identify if a high-scoring candidate is just refactoring (reject it)
    - Verify actual code changes that could introduce the bug
    
    Args:
        sha: Full or partial commit SHA (node identifier)
        
    Returns:
        The unified diff content stored in the node (truncated if large).
    """
    # First find the UUID for this SHA
    episodes = await ctx.deps.graphiti.retrieve_episodes(
        reference_time=datetime.now(timezone.utc),
        group_ids=[ctx.deps.bfc],
        last_n=100,
    )
    
    # Simple matching logic same as query_node
    matching_episode = None
    for ep in episodes:
        if ep.name == sha or sha in ep.name or (hasattr(ep, "content") and sha in ep.content):
            matching_episode = ep
            break
            
    if not matching_episode:
        return f"No node found matching SHA {sha}"
        
    # Query the diff_content property from the node
    query = """
    MATCH (e:Episodic {uuid: $uuid})
    RETURN e.diff_content as diff
    """
    
    result = await ctx.deps.graphiti.driver.execute_query(
        query, uuid=matching_episode.uuid
    )
    
    if result.records and result.records[0]['diff']:
        return f"Diff for commit {sha}:\n{result.records[0]['diff']}"
    else:
        # Fallback if no diff stored (e.g. binary file or extraction failed)
        return f"No diff content available for commit {sha} (might be binary file or too large)"


async def traverse_graph(ctx: RunContext[GraphitiContext], sha: str) -> str:
    """
    Traverse the TKG via MODIFIES_* edges to find structurally related commits.
    
    This tool follows edges through File and Function nodes to discover
    commits that share code structure with the given commit:
    - MODIFIES_FUNCTION edges: commits touching the same functions
    - MODIFIES_FILE edges: commits touching the same files
    
    Example: Call with BFC's SHA to find candidates that modified the same
    code entities that the BFC is fixing.
    
    Args:
        sha: Commit SHA (node) to start traversal from
    
    Returns:
        Commits connected via shared File/Function nodes, grouped by type.
    """
    # First, find the episode for this SHA
    episodes = await ctx.deps.graphiti.retrieve_episodes(
        reference_time=datetime.now(timezone.utc),
        group_ids=[ctx.deps.bfc],
        last_n=100,
    )
    
    target_episode = None
    for ep in episodes:
        # Episode name is the SHA, or check content for partial match
        if ep.name == sha or sha in ep.name or (hasattr(ep, "content") and sha in (ep.content or "")):
            target_episode = ep
            break
    
    if not target_episode:
        return f"No node found with SHA {sha}"
    
    target_name = target_episode.name
    
    # Traverse edges to find connected commit nodes
    function_query = """
    MATCH (target:Episodic {name: $target_name})-[:MODIFIES_FUNCTION]->(f:Function)<-[:MODIFIES_FUNCTION]-(other:Episodic)
    WHERE target <> other AND other.group_id = $group_id
    RETURN DISTINCT other.name AS episode_name, other.source_description AS commit_type,
           collect(DISTINCT f.name) AS shared_functions
    """
    
    # Query Neo4j for commits sharing files (but not the same functions)
    file_query = """
    MATCH (target:Episodic {name: $target_name})-[:MODIFIES_FILE]->(file:File)<-[:MODIFIES_FILE]-(other:Episodic)
    WHERE target <> other AND other.group_id = $group_id
    AND NOT EXISTS {
        MATCH (target)-[:MODIFIES_FUNCTION]->(:Function)<-[:MODIFIES_FUNCTION]-(other)
    }
    RETURN DISTINCT other.name AS episode_name, other.source_description AS commit_type,
           collect(DISTINCT file.path) AS shared_files
    """
    
    lines = [f"Nodes connected to {sha} via code edges:\n"]
    
    try:
        # Define priority order for commit types
        type_priority = {
            'blame': 0,
            'blame_ancestor': 1,
            'bfc_ancestor': 2,
        }
        
        # Collect all results with their metadata
        all_commits = []
        
        # Find commits sharing functions
        func_results = await ctx.deps.graphiti.driver.execute_query(
            function_query, 
            target_name=target_name,
            group_id=ctx.deps.bfc,
        )
        
        if func_results.records:
            for record in func_results.records:
                all_commits.append({
                    'sha': record["episode_name"],
                    'ctype': record["commit_type"] or "unknown",
                    'shared': record["shared_functions"],
                    'section': 'functions',
                })
        
        # Find commits sharing files only
        file_results = await ctx.deps.graphiti.driver.execute_query(
            file_query,
            target_name=target_name,
            group_id=ctx.deps.bfc,
        )
        
        if file_results.records:
            for record in file_results.records:
                all_commits.append({
                    'sha': record["episode_name"],
                    'ctype': record["commit_type"] or "unknown",
                    'shared': record["shared_files"],
                    'section': 'files',
                })
        
        # Sort by commit type priority FIRST, then by section (functions before files)
        section_priority = {'functions': 0, 'files': 1}
        all_commits.sort(key=lambda c: (
            type_priority.get(c['ctype'], 99),
            section_priority.get(c['section'], 99)
        ))
        
        # Output grouped by commit type
        current_type = None
        for commit in all_commits:
            if commit['ctype'] != current_type:
                current_type = commit['ctype']
                # Add section header for new commit type
                lines.append(f"\n**[{current_type}] commits:**")
            
            section_label = "functions" if commit['section'] == 'functions' else "files"
            lines.append(f"- {commit['sha']}")
            lines.append(f"  Shared {section_label}: {', '.join(commit['shared'])}")
                
    except Exception as e:
        lines.append(f"Error querying graph: {str(e)}")
    
    if len(lines) == 1:
        lines.append("No connected nodes found via code edges.")
    
    return "\n".join(lines)
