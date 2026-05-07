"""
Temporal Knowledge Graph for AgenticSZZ

This module builds a TKG containing suspicious commits for BIC identification.
Uses Graphiti for graph management with Neo4j backend.

Commit Types (from RQ0 insights):
- bfc: Bug-fixing commit (the fix we start from)
- blame: Direct git blame result (57% of BICs)
- blame_ancestor: Commits older than blame (10% of BICs)
- bfc_ancestor: Commits between blame and BFC (18% of BICs, "descendant" cases)

File-History Depth Limits (80% coverage from RQ0):
- blame_ancestor: 24 commits (blame → older direction)
- bfc_ancestor: 14 commits (BFC → blame direction)

Usage:
    from agentic_szz.kg import SuspiciousCommitTKG

    async with SuspiciousCommitTKG(repo_path) as tkg:
        result = await tkg.build(bfc, blame_commits)
        print(result.candidates)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import git
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from pydantic import BaseModel


def _sanitize_commit_message(message: str) -> str:
    """
    Sanitize commit messages to prevent data leakage of BIC information.

    Removes:
    1. Fixes: tags (e.g., "Fixes: 68b99e9 ("description")")
    2. Inline commit references (e.g., "Commit 68b99e9 ("description...")")

    Both SHA references and their quoted descriptions are removed because:
    - SHA references directly leak the BIC identity
    - Quoted descriptions are the BIC's commit message, enabling text-matching shortcuts

    The surrounding context explaining the bug nature is preserved.
    """
    if not message:
        return ""

    sanitized = message

    # Pattern 1: Remove Fixes: lines entirely
    # Matches: Fixes: <sha> ("description") or Fixes: <sha> (description)
    fixes_pattern = re.compile(
        r"^\s*Fixes:\s+[a-f0-9]+\s*\(.*?\)\s*$", re.MULTILINE | re.IGNORECASE
    )
    sanitized = fixes_pattern.sub("", sanitized)

    # Pattern 2: Remove inline commit references with quoted descriptions
    # Matches: [Cc]ommit <sha> ("description...") - handles multi-line descriptions
    # This pattern handles quotes that may span lines
    inline_commit_pattern = re.compile(
        r'\b[Cc]ommit\s+[a-f0-9]{7,40}\s*\(["\'].*?["\']\)',
        re.DOTALL,  # Allow . to match newlines for multi-line descriptions
    )
    sanitized = inline_commit_pattern.sub("[referenced commit]", sanitized)

    # Pattern 3: Remove any remaining bare commit references (without quoted description)
    # Matches: [Cc]ommit <sha> followed by word boundary
    bare_commit_pattern = re.compile(r"\b[Cc]ommit\s+[a-f0-9]{7,40}\b")
    sanitized = bare_commit_pattern.sub("[referenced commit]", sanitized)

    # Clean up extra blank lines and whitespace
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    sanitized = re.sub(r"  +", " ", sanitized)  # Collapse multiple spaces

    return sanitized.strip()


def _extract_functions_from_diff(repo: git.Repo, commit: git.Commit) -> List[str]:
    """
    Extract function/method names from a commit's diff.

    Uses git diff's built-in function detection via the @@ hunk headers.
    Git automatically shows the enclosing function name for changes in
    C/C++, Python, and other languages with built-in diff drivers.

    Args:
        repo: Git repository object
        commit: Commit to extract functions from

    Returns:
        List of unique function names modified in this commit
    """
    if not commit.parents:
        return []

    functions = set()

    try:
        # Get the raw diff with function context (-p includes @@ headers)
        # The @@ line format is: @@ -old,len +new,len @@ function_name
        diff_text = repo.git.diff(commit.parents[0].hexsha, commit.hexsha, "-p")

        # Pattern to match the function context in @@ headers
        # Format: @@ -start,len +start,len @@ optional_function_context
        hunk_pattern = re.compile(
            r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@\s*(.*)$", re.MULTILINE
        )

        for match in hunk_pattern.finditer(diff_text):
            context = match.group(1).strip()
            if context:
                # Extract function name from context
                # Common patterns:
                # - C/C++: "static int function_name(params)" -> "function_name"
                # - Python: "def function_name(params):" -> "function_name"
                # - Java: "public void methodName(params)" -> "methodName"

                # Try to extract just the function name
                func_name = _parse_function_name(context)
                if func_name:
                    functions.add(func_name)

    except git.GitCommandError:
        # Fallback: no function extraction on diff error
        pass

    return list(functions)[:10]  # Limit to 10 functions


def _parse_function_name(context: str) -> Optional[str]:
    """
    Parse function name from a diff hunk context line.

    Examples:
        "static int foo_bar(void)" -> "foo_bar"
        "def process_data(self, x):" -> "process_data"
        "public void handleRequest()" -> "handleRequest"
    """
    if not context:
        return None

    # Pattern 1: C-style function definition
    # Matches: [modifiers] type function_name(
    c_pattern = re.compile(r"\b(\w+)\s*\([^)]*\)\s*(?:\{|;|$)")
    c_match = c_pattern.search(context)
    if c_match:
        func = c_match.group(1)
        # Filter out common C keywords that aren't function names
        if func not in ("if", "while", "for", "switch", "return", "sizeof"):
            return func

    # Pattern 2: Python def/async def
    py_pattern = re.compile(r"\bdef\s+(\w+)\s*\(")
    py_match = py_pattern.search(context)
    if py_match:
        return py_match.group(1)

    # Pattern 3: Class/struct definition
    class_pattern = re.compile(r"\b(?:class|struct)\s+(\w+)")
    class_match = class_pattern.search(context)
    if class_match:
        return class_match.group(1)

    return None


# Custom entity type for controlled extraction
class FileEntity(BaseModel):
    """A source code file path"""

    path: str


@dataclass
class CommitInfo:
    """Information about a commit in the TKG"""

    sha: str
    timestamp: datetime
    author: str
    message: str
    files_changed: List[str] = field(default_factory=list)
    # Suspicion metadata
    is_blame_commit: bool = False
    is_bfc_ancestor: bool = False
    is_bfc_descendant: bool = False
    suspicion_score: float = 0.0


@dataclass
class TKGStats:
    """Statistics about the built TKG"""

    total_commits: int = 0
    blame_commits: int = 0
    blame_ancestor_commits: int = 0
    bfc_ancestor_commits: int = 0
    build_time_ms: float = 0.0


@dataclass
class TKGResult:
    """Result from TKG construction"""

    bfc: str
    candidates: List[str]  # Commit SHAs ranked by suspicion
    scores: Dict[str, float]  # SHA -> suspicion score
    stats: TKGStats = field(default_factory=TKGStats)


class SuspiciousCommitTKG:
    """
    Builds a TKG containing commits suspicious of being the BIC.

    Uses Graphiti for graph storage with Neo4j backend.
    Episodes are used to store commit information.

    Usage:
        async with SuspiciousCommitTKG(repo_path) as tkg:
            result = await tkg.build(bfc, blame_commits)
            print(result.candidates)
    """

    def __init__(self, repo_path: str, neo4j_config=None):
        from agentic_szz.config.settings import get_config

        self.config = get_config()
        # Override Neo4j config if provided (for worker isolation)
        if neo4j_config:
            self.config.neo4j = neo4j_config
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.graphiti: Optional[Graphiti] = None
        # Cache repo stats for adaptive thresholds
        self._repo_stats: Optional[Tuple[int, int]] = None

    def _get_repo_stats(self) -> Tuple[int, int]:
        """Get repository commit and file counts for adaptive thresholds.

        Caches result to avoid repeated git commands.

        Returns:
            Tuple of (commit_count, file_count)
        """
        if self._repo_stats is not None:
            return self._repo_stats

        try:
            # Count commits (use rev-list for efficiency)
            commit_count = int(self.repo.git.rev_list("--count", "HEAD"))
        except Exception:
            commit_count = 1000  # Fallback to medium-sized estimate

        try:
            # Count tracked files
            file_count = len(self.repo.git.ls_files().split("\n"))
        except Exception:
            file_count = 500  # Fallback to medium-sized estimate

        self._repo_stats = (commit_count, file_count)
        return self._repo_stats

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def connect(self):
        """
        Initialize Graphiti client and connect to Neo4j.

        Auto-starts Neo4j Docker container if not running.
        """
        if self.graphiti is not None:
            return

        neo4j = self.config.neo4j

        # Check if Neo4j is accessible, start container if needed
        if not self._check_neo4j_connection(neo4j.uri):
            self._ensure_neo4j_container(neo4j)
            # Wait for Neo4j to be ready
            import time

            print("Waiting for Neo4j to be ready (up to 90 seconds)...")
            for i in range(90):
                if self._check_neo4j_connection(neo4j.uri):
                    # Wait additional 3 seconds for Bolt protocol to fully initialize
                    # Socket may be open before Bolt handshake is ready
                    time.sleep(3)
                    print(f"Neo4j ready after {i + 4} seconds")
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Neo4j failed to start within 90 seconds")

        self.graphiti = Graphiti(neo4j.uri, neo4j.user, neo4j.password)

    async def close(self):
        """Close the Graphiti client and Neo4j connection."""
        if self.graphiti is not None:
            await self.graphiti.close()
            self.graphiti = None

    def _check_neo4j_connection(self, uri: str) -> bool:
        """Check if Neo4j is accessible via socket."""
        import socket

        try:
            # Parse host and port from URI
            uri_clean = uri.replace("bolt://", "").replace("neo4j://", "")
            host, port = uri_clean.split(":")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _ensure_neo4j_container(self, neo4j):
        """Start Neo4j Docker container if not running."""
        import shutil
        import subprocess

        if not shutil.which("docker"):
            raise RuntimeError(
                "Docker is not installed. Please install Docker and run:\n"
                "  docker pull neo4j:latest\n"
            )

        # Check if container exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=^neo4j$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )

        if "neo4j" in result.stdout.strip():
            # Container exists, start it if stopped
            print("Starting existing Neo4j container...")
            subprocess.run(["docker", "start", "neo4j"])
        else:
            # Check if image exists
            image_result = subprocess.run(
                ["docker", "images", "neo4j", "--format", "{{.Repository}}"],
                capture_output=True,
                text=True,
            )

            if "neo4j" not in image_result.stdout:
                raise RuntimeError(
                    "Neo4j Docker image not found. Please pull it manually:\n\n"
                    "  docker pull neo4j:latest\n\n"
                )

            # Create and start new container
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    "neo4j",
                    "-p",
                    "7474:7474",
                    "-p",
                    "7687:7687",
                    "-e",
                    f"NEO4J_AUTH={neo4j.user}/{neo4j.password}",
                    "neo4j:latest",
                ]
            )

        print("Started Neo4j container. Waiting for it to be ready...")

    async def clean_tkg(self, group_id: str = None):
        """
        Clean TKG episode data from Neo4j.

        Args:
            group_id: If provided, only clean episodes with this group_id.
                      If None, cleans ALL episode data.

        Usage:
        - After each case: await tkg.clean_tkg(group_id=bfc)
        - For full database reset, use reset_neo4j_database() instead
        """
        try:
            if not self.graphiti:
                await self.connect()

            if group_id:
                # Clean all nodes with specific group_id (Episodic, Function, File)
                queries = [
                    "MATCH (e:Episodic) WHERE e.group_id = $group_id DETACH DELETE e",
                    "MATCH (f:Function) WHERE f.group_id = $group_id DETACH DELETE f",
                    "MATCH (f:File) WHERE f.group_id = $group_id DETACH DELETE f",
                ]
                for query in queries:
                    await self.graphiti.driver.execute_query(query, group_id=group_id)
            else:
                # Clean ALL TKG data
                queries = [
                    "MATCH (e:Episodic) DETACH DELETE e",
                    "MATCH (f:Function) DETACH DELETE f",
                    "MATCH (f:File) DETACH DELETE f",
                ]
                for query in queries:
                    await self.graphiti.driver.execute_query(query)
        except RuntimeError as e:
            # Handle event loop closed errors gracefully
            if "Event loop is closed" in str(e):
                pass  # Cleanup failed but not critical
            else:
                raise

    async def build(
        self,
        bfc: str,
        blame_commits: List[str],
        modified_files: List[str] = None,
        is_blameless: bool = False,
        blame_distribution: Dict[str, int] = None,
    ) -> TKGResult:
        """
        Build a TKG for the given BFC.

        Args:
            bfc: Bug-fixing commit hash
            blame_commits: Initial blame commits from BlameAnalyzer
            modified_files: List of files modified in BFC
            is_blameless: If True, use blameless-specific depth thresholds (from RQ0)
            blame_distribution: Mapping of blame commit SHA to line count

        Returns:
            TKGResult with ranked candidate commits
        """
        from agentic_szz.config.settings import get_config

        # Use generous upper bounds for traversal depth
        # The agent will only see top-K candidates via list_candidates() anyway
        tkg_config = get_config().tkg
        max_ancestor_depth = tkg_config.ancestor_depth
        max_descendant_depth = tkg_config.descendant_depth
        max_candidates = tkg_config.max_candidates

        import time

        start_time = time.time()

        if not self.graphiti:
            await self.connect()

        # Pre-clean: Remove any existing data for this BFC's group_id
        # This ensures a fresh graph even if same BFC runs twice (e.g., duplicate in sample)
        # Safe to call on non-existing data - Cypher DELETE on 0 matches is a no-op
        await self.clean_tkg(group_id=bfc)

        stats = TKGStats()
        candidates = {}  # sha -> score
        episode_chain = []  # Track temporal order for previous_episode_uuids

        # Get BFC info
        bfc_commit = self.repo.commit(bfc)

        # Get BFC modified files for filtering
        bfc_files = set()
        if bfc_commit.parents:
            bfc_diffs = bfc_commit.parents[0].diff(bfc_commit)
            bfc_files = {d.a_path or d.b_path for d in bfc_diffs}

        # Step 1: Add BFC as episode (no previous episodes)
        bfc_uuid = await self._add_commit_episode(
            bfc, bfc_commit, "bfc", bfc, bfc_files, previous_uuids=[]
        )
        episode_chain.append(bfc_uuid)

        # Add blame statistics to BFC episode for agent access
        if blame_distribution:
            await self._add_blame_stats_to_bfc(bfc, blame_distribution)

        # Step 2: Add blame commits (highest priority, link to BFC)
        blame_uuids = {}  # sha -> uuid for later linking
        blame_distribution = blame_distribution or {}
        for sha in blame_commits:
            try:
                commit = self.repo.commit(sha)
                # Get line count for this blame commit (helps agent rank when multiple blames)
                line_count = blame_distribution.get(sha, None)
                blame_uuid = await self._add_commit_episode(
                    sha,
                    commit,
                    "blame",
                    bfc,
                    bfc_files,
                    previous_uuids=[bfc_uuid],
                    line_count=line_count,
                )
                if blame_uuid:
                    blame_uuids[sha] = blame_uuid
                    episode_chain.append(blame_uuid)
                    candidates[sha] = 1.0  # Highest base score
                    stats.blame_commits += 1
            except Exception:
                continue

        # Step 3: Find blame_ancestors (commits older than blame)
        if modified_files:
            for sha in blame_commits:
                # Check max_candidates limit at outer loop
                if len(candidates) >= max_candidates:
                    break
                ancestors = self._find_file_history_ancestors(
                    sha, modified_files, max_ancestor_depth
                )
                prev_uuid = blame_uuids.get(sha)  # Link to blame commit's UUID
                if not prev_uuid:
                    continue
                for ancestor_sha, depth in ancestors:
                    # Check max_candidates limit
                    if len(candidates) >= max_candidates:
                        break
                    if ancestor_sha not in candidates and ancestor_sha != bfc:
                        try:
                            commit = self.repo.commit(ancestor_sha)
                            ancestor_uuid = await self._add_commit_episode(
                                ancestor_sha,
                                commit,
                                "blame_ancestor",
                                bfc,
                                bfc_files,
                                previous_uuids=[prev_uuid],
                            )
                            if ancestor_uuid:
                                prev_uuid = ancestor_uuid  # Chain ancestors together
                            # Score decreases with depth
                            candidates[ancestor_sha] = 0.8 / (1 + depth * 0.1)
                            stats.blame_ancestor_commits += 1
                        except Exception:
                            continue

        # Step 4: Find bfc_ancestors (commits between blame and BFC)
        # Skip if blame commit is already immediate parent of BFC or max_candidates reached
        if modified_files and len(candidates) < max_candidates:
            # Check if any blame commit is at depth 1 from BFC
            first_bfc_ancestor = self._find_file_history_bfc_ancestors(bfc, modified_files, 1)
            blame_is_immediate = any(sha in blame_commits for sha, _ in first_bfc_ancestor)

            if not blame_is_immediate:
                # Blame is not immediate parent, search for bfc_ancestors
                bfc_ancestors = self._find_file_history_bfc_ancestors(
                    bfc, modified_files, max_descendant_depth
                )
                prev_uuid = bfc_uuid  # Link to BFC's UUID
                for ancestor_sha, depth in bfc_ancestors:
                    # Check max_candidates limit
                    if len(candidates) >= max_candidates:
                        break
                    if ancestor_sha not in candidates and ancestor_sha not in blame_commits:
                        try:
                            commit = self.repo.commit(ancestor_sha)
                            ancestor_uuid = await self._add_commit_episode(
                                ancestor_sha,
                                commit,
                                "bfc_ancestor",
                                bfc,
                                bfc_files,
                                previous_uuids=[prev_uuid],
                            )
                            if ancestor_uuid:
                                prev_uuid = ancestor_uuid  # Chain together
                            # Score by depth from BFC
                            candidates[ancestor_sha] = 0.9 / (1 + depth * 0.1)
                            stats.bfc_ancestor_commits += 1
                        except Exception:
                            continue

        # Finalize stats
        stats.total_commits = len(candidates) + 1  # +1 for BFC
        stats.build_time_ms = (time.time() - start_time) * 1000

        # Add temporal edges between episodes (Neo4j direct)
        await self._add_temporal_edges(bfc)

        # Sort candidates by score
        sorted_candidates = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)

        return TKGResult(bfc=bfc, candidates=sorted_candidates, scores=candidates, stats=stats)

    async def search_bic(self, bfc: str) -> Tuple[str, float, str]:
        """
        Search for BIC using TKG without agent reasoning (TKG-only ablation).

        Uses the SAME scoring logic as list_candidates tool for fair ablation:
        1. Check for SINGLE blame commit → return immediately
        2. Check for DOMINANT commit (>70% lines) → return immediately
        3. Otherwise, score by tiered fitness and return top candidate

        This ensures TKG-only measures TKG's candidate selection quality,
        not a different algorithm. The gap between TKG-only and full
        shows the value of agent reasoning.

        Args:
            bfc: Bug-fixing commit SHA

        Returns:
            Tuple of (predicted_bic, confidence, reasoning)
        """
        if not self.graphiti:
            await self.connect()

        # Step 1: Get blame statistics from BFC episode (same as list_candidates)
        blame_query = """
        MATCH (e:Episodic {name: $bfc, source_description: 'bfc'})
        RETURN e.blame_total_commits AS total_commits,
               e.blame_single_commit AS single_commit,
               e.blame_dominant_commit AS dominant_commit,
               e.blame_total_lines AS total_lines
        """
        try:
            blame_result = await self.graphiti.driver.execute_query(blame_query, bfc=bfc)
        except Exception:
            blame_result = type("obj", (object,), {"records": []})()

        # Step 2: Get all candidates (same as list_candidates)
        from graphiti_core.nodes import EpisodicNode

        episodes = await self.graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            group_ids=[bfc],
            last_n=100,
        )

        # Parse episodes to get commit info
        candidates = []
        blame_commits = []
        for ep in episodes:
            # Skip BFC itself
            if ep.source_description == "bfc":
                continue

            sha = ep.name
            commit_type = ep.source_description or "unknown"

            # Extract line_count from episode content if available
            line_count = 0
            if ep.content and "Blame lines:" in ep.content:
                try:
                    line_count = int(ep.content.split("Blame lines:")[1].split()[0])
                except:
                    pass

            candidates.append(
                {
                    "sha": sha,
                    "commit_type": commit_type,
                    "line_count": line_count,
                }
            )

            if commit_type == "blame":
                blame_commits.append(
                    {
                        "sha": sha,
                        "line_count": line_count,
                    }
                )

        # Step 3: Apply fast-path rules (SAME as list_candidates)
        if blame_result.records and len(blame_result.records) > 0:
            rec = blame_result.records[0]
            single = rec.get("single_commit", False) or False
            dominant = rec.get("dominant_commit")

            # SINGLE BLAME - immediate return (79% correct in analysis)
            if single and len(blame_commits) == 1:
                sha = blame_commits[0]["sha"]
                return sha, 1.0, "TKG-only: SINGLE blame commit (fast-path)"

            # DOMINANT COMMIT - immediate return (50% correct, but consistent with agent)
            if dominant and dominant in [c["sha"] for c in candidates]:
                return dominant, 0.9, f"TKG-only: DOMINANT commit >70% lines (fast-path)"

        # Step 4: Score candidates without line-count bonus (trust blame order)
        if not candidates:
            return "", 0.0, "TKG-only: no candidates found"

        scored_candidates = []
        for c in candidates:
            # Base score by type - no line-count bonus (analysis showed it overrides correct answers)
            type_scores = {"blame": 1.0, "blame_ancestor": 0.6, "bfc_ancestor": 0.3}
            score = type_scores.get(c["commit_type"], 0.1)

            scored_candidates.append((c["sha"], score))

        # Sort by score descending and return top
        scored_candidates.sort(key=lambda x: -x[1])
        best_sha, best_score = scored_candidates[0]

        return best_sha, best_score, f"TKG-only: top fitness score {best_score:.2f}"

    async def _get_all_candidates(self, bfc: str) -> List[str]:
        """Get all candidate SHAs from the TKG for a given BFC."""
        # Episode name IS the SHA now, use source_description to filter out BFC
        query = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id AND e.source_description <> 'bfc'
        RETURN e.name AS sha
        """
        try:
            results = await self.graphiti.driver.execute_query(query, group_id=bfc)
            candidates = []
            for record in results[0]:
                sha = record["sha"]  # Episode name IS the SHA
                candidates.append(sha)
            return candidates
        except Exception:
            return []

    async def _add_temporal_edges(self, group_id: str):
        """
        Add PRECEDES edges between episodes based on temporal order.

        Creates edges: older_episode -[:PRECEDES]-> newer_episode
        Only for episodes in the same group (same BFC case).
        """
        query = """
        MATCH (e1:Episodic), (e2:Episodic)
        WHERE e1.group_id = $group_id
          AND e2.group_id = $group_id
          AND e1.valid_at < e2.valid_at
          AND NOT EXISTS((e1)-[:PRECEDES]->(e2))
        WITH e1, e2
        ORDER BY e1.valid_at, e2.valid_at
        WITH e1, collect(e2)[0] AS next_episode
        WHERE next_episode IS NOT NULL
        CREATE (e1)-[:PRECEDES]->(next_episode)
        """
        await self.graphiti.driver.execute_query(query, group_id=group_id)

    async def _create_code_graph_entities(
        self,
        episode_name: str,
        files: List[str],
        functions: List[str],
        group_id: str,
    ):
        """
        Create Function and File nodes and MODIFIES edges in Neo4j.

        This replaces LLM-based entity extraction with efficient programmatic creation.
        Enables graph-based code traversal queries via traverse_graph tool.

        Args:
            episode_name: Full 40-char SHA of the commit (episode name is just SHA)
            files: List of file paths modified by this commit
            functions: List of function names modified by this commit
            group_id: BFC SHA (for grouping)
        """
        # Create File nodes and MODIFIES_FILE edges
        if files:
            file_query = """
            MATCH (e:Episodic {name: $episode_name})
            UNWIND $files AS file_path
            MERGE (f:File {path: file_path, group_id: $group_id})
            MERGE (e)-[:MODIFIES_FILE]->(f)
            """
            await self.graphiti.driver.execute_query(
                file_query,
                episode_name=episode_name,
                files=files,
                group_id=group_id,
            )

        # Create Function nodes and MODIFIES_FUNCTION edges
        # Use qualified_name (file::function) for uniqueness
        if functions:
            # For each function, associate with its file if possible
            # Functions are extracted from diff, so we know which file they're in
            # For simplicity, use the first file (most commits modify one file)
            primary_file = files[0] if files else "unknown"

            function_query = """
            MATCH (e:Episodic {name: $episode_name})
            UNWIND $functions AS func_name
            WITH e, func_name, $primary_file + '::' + func_name AS qualified_name
            MERGE (fn:Function {qualified_name: qualified_name, group_id: $group_id})
            ON CREATE SET fn.name = func_name, fn.file = $primary_file
            MERGE (e)-[:MODIFIES_FUNCTION]->(fn)
            WITH fn, $primary_file AS file_path
            MATCH (f:File {path: file_path, group_id: $group_id})
            MERGE (fn)-[:DEFINED_IN]->(f)
            """
            await self.graphiti.driver.execute_query(
                function_query,
                episode_name=episode_name,
                functions=functions,
                primary_file=primary_file,
                group_id=group_id,
            )

    async def _add_commit_episode(
        self,
        sha: str,
        commit: git.Commit,
        commit_type: str,  # "bfc", "blame", "blame_ancestor", "bfc_ancestor"
        bfc: str,
        bfc_files: set,
        previous_uuids: List[str] = None,
        line_count: int = None,
    ) -> str:
        """
        Add a commit as an episode to Graphiti.

        Args:
            sha: Commit SHA
            commit: Git commit object
            commit_type: bfc, blame, blame_ancestor, or bfc_ancestor
            bfc: BFC hash (used as group_id)
            bfc_files: Set of files modified by BFC (for filtering)
            previous_uuids: Previous episode UUIDs for temporal chain
            line_count: Number of deleted/modified lines blamed on this commit (for blame type)

        Returns:
            The created episode's UUID (for chaining), or None if skipped
        """
        # Get files changed by this commit
        files_changed = []
        if commit.parents:
            diffs = commit.parents[0].diff(commit)
            files_changed = [d.a_path or d.b_path for d in diffs][:20]  # Limit

        # File relevance handling
        # Note: git log --follow already handles file renames during traversal.
        # Only BFC needs explicit file listing. Other commit types are already
        # relevant by definition (found via blame or file history traversal).
        if commit_type == "bfc":
            # BFC keeps all its files
            relevant_files = files_changed
        else:
            # blame, blame_ancestor, bfc_ancestor: already relevant by definition
            # - blame: identified by git blame on BFC's deleted lines
            # - blame_ancestor: found via git log --follow from blame commit
            # - bfc_ancestor: found via git log --follow from BFC
            # Don't filter - file renames may cause false negatives
            relevant_files = files_changed if files_changed else ["(file history)"]

        # Only skip if truly no file info and not BFC (shouldn't happen normally)
        if not relevant_files:
            return None

        # Extract functions modified by this commit (for agent context)
        functions = _extract_functions_from_diff(self.repo, commit)
        functions_str = ", ".join(functions) if functions else "(no function context)"

        # Create structured episode content
        # Sanitize message to remove Fixes: tags that leak ground truth BIC info
        sanitized_msg = _sanitize_commit_message(commit.message)

        # Include line count for blame commits (helps agent rank multiple blame candidates)
        line_count_str = f"\nBlame lines: {line_count}" if line_count else ""

        episode_body = (
            f"SHA: {sha}\n"
            f"Modified files: {', '.join(relevant_files)}\n"
            f"Functions changed: {functions_str}"
            f"{line_count_str}\n"
            f"Message: {sanitized_msg}"
        )

        # Create EpisodicNode directly (bypass LLM extraction for speed)
        # This is the same approach used internally by add_episode()
        # Episode name is just the SHA; commit_type stored in source_description
        episode_name = sha
        reference_time = (
            commit.committed_datetime.replace(tzinfo=timezone.utc)
            if commit.committed_datetime.tzinfo is None
            else commit.committed_datetime
        )

        episode = EpisodicNode(
            name=episode_name,
            content=episode_body,
            source=EpisodeType.text,
            source_description=commit_type,
            group_id=bfc,
            valid_at=reference_time,
        )

        # Save episode directly to Neo4j - NO LLM CALLS!
        # This is much faster than add_episode() which runs entity extraction
        await add_nodes_and_edges_bulk(
            driver=self.graphiti.driver,
            episodic_nodes=[episode],
            episodic_edges=[],  # No episodic edges needed
            entity_nodes=[],  # We create our own Function/File nodes below
            entity_edges=[],
            embedder=self.graphiti.embedder,
        )

        # Create Function and File nodes with MODIFIES edges programmatically
        # This enables graph-based code traversal queries (traverse_graph)
        await self._create_code_graph_entities(
            episode_name=episode_name,
            files=relevant_files,
            functions=functions,
            group_id=bfc,
        )

        # Return the created episode's UUID for chaining

        # Extract and store diff content (on-demand storage for causal analysis)
        try:
            # Use git show to get the diff text
            # -U1: Minimal context
            # --: Delimiter for files
            files_args = (
                relevant_files if relevant_files and relevant_files[0] != "(file history)" else []
            )
            diff_cmd = ["-U1", sha, "--"] + files_args

            # Use decode explicit with ignore/replace to handle binary/weird chars
            # git.show returns string by default in GitPython, but we can access raw stats or bytes
            # Typically self.repo.git.show returns a string decoded with default encoding
            # Use 'surrogateescape' or cleaner approach?
            # Better: encode/decode to sanitize surrogates which Neo4j hates
            diff_text = self.repo.git.show(*diff_cmd)

            # Sanitize for Neo4j (remove surrogates)
            diff_text = diff_text.encode("utf-8", "replace").decode("utf-8")

            # Truncate if too large to prevent Neo4j bloat and context overflow
            MAX_DIFF_LINES = 500
            diff_lines = diff_text.split("\n")
            if len(diff_lines) > MAX_DIFF_LINES:
                diff_text = (
                    "\n".join(diff_lines[:MAX_DIFF_LINES])
                    + f"\n... (truncated {len(diff_lines) - MAX_DIFF_LINES} lines)"
                )

            # Store diff as separate property on the node
            query = """
            MATCH (e:Episodic {uuid: $uuid})
            SET e.diff_content = $diff_content
            """
            await self.graphiti.driver.execute_query(
                query, uuid=episode.uuid, diff_content=diff_text
            )
        except Exception:
            # Diff extraction failed (e.g. binary files or too large), just skip it
            pass

        return episode.uuid

    async def _add_blame_stats_to_bfc(self, bfc: str, blame_distribution: Dict[str, int]):
        """
        Add blame statistics as properties to BFC episode node.

        These stats help the agent understand the blame landscape without
        needing to iterate through all blame commits.

        Args:
            bfc: BFC SHA (episode name)
            blame_distribution: {commit_sha: line_count} from BlameAnalyzer
        """
        total_lines = sum(blame_distribution.values())
        num_commits = len(blame_distribution)

        # Find dominant commit (>70% of lines) if exists
        dominant_commit = None
        if num_commits > 0 and total_lines > 0:
            sorted_commits = sorted(blame_distribution.items(), key=lambda x: -x[1])
            top_sha, top_count = sorted_commits[0]
            if top_count / total_lines > 0.7:
                dominant_commit = top_sha

        # Update BFC episode with blame properties
        query = """
        MATCH (e:Episodic {name: $bfc, source_description: 'bfc'})
        SET e.blame_total_commits = $num_commits,
            e.blame_total_lines = $total_lines,
            e.blame_single_commit = $single_commit,
            e.blame_dominant_commit = $dominant_commit
        """
        await self.graphiti.driver.execute_query(
            query,
            bfc=bfc,
            num_commits=num_commits,
            total_lines=total_lines,
            single_commit=(num_commits == 1),
            dominant_commit=dominant_commit,
        )

    def _find_file_history_ancestors(
        self, commit_sha: str, file_paths: List[str], max_depth: int
    ) -> List[Tuple[str, int]]:
        """
        Find ancestor commits that modified any of the given files.

        Uses git log --follow to track file renames.
        This is file-history depth (not DAG depth).
        """
        ancestors = []
        try:
            # Get commits before commit_sha that touched these files
            commits_output = self.repo.git.log(
                f"{commit_sha}",
                "--format=%H",
                "--follow",
                f"-{max_depth + 1}",  # +1 to skip the commit itself
                "--",
                *file_paths,
            )

            if commits_output.strip():
                commits = [c.strip() for c in commits_output.strip().split("\n") if c.strip()]
                # Skip the first commit (it's the blame commit itself)
                for i, sha in enumerate(commits[1 : max_depth + 1], start=1):
                    if sha != commit_sha:
                        ancestors.append((sha, i))
        except Exception:
            pass

        return ancestors

    def _find_file_history_bfc_ancestors(
        self, bfc: str, file_paths: List[str], max_depth: int
    ) -> List[Tuple[str, int]]:
        """
        Find commits from BFC backward (BFC's ancestors) that modified given files.

        These are potential BICs in "descendant" cases (BIC is between blame and BFC).
        Uses git log --follow to track file renames.
        """
        bfc_ancestors = []
        try:
            # Get commits before BFC that touched these files
            commits_output = self.repo.git.log(
                f"{bfc}~1",  # Start from parent of BFC
                "--format=%H",
                "--follow",
                f"-{max_depth}",
                "--",
                *file_paths,
            )

            if commits_output.strip():
                commits = [c.strip() for c in commits_output.strip().split("\n") if c.strip()]
                for i, sha in enumerate(commits[:max_depth], start=1):
                    bfc_ancestors.append((sha, i))
        except Exception:
            pass

        return bfc_ancestors

    async def search_similar(self, query: str, limit: int = 10) -> List[dict]:
        """
        Search for similar commits using Graphiti's semantic search.

        Useful for blameless cases where we need semantic similarity.
        """
        if not self.graphiti:
            await self.connect()

        results = await self.graphiti.search(query, num_results=limit)
        return [
            {"fact": r.fact, "uuid": r.uuid, "valid_at": getattr(r, "valid_at", None)}
            for r in results
        ]


def reset_neo4j_database():
    """
    Reset Neo4j database for fresh evaluation runs.

    Performs a full wipe of all nodes in the database.
    Uses synchronous Neo4j driver to avoid event loop conflicts.

    Uses connection settings from get_config().neo4j (loaded from env/.env file).

    Usage:
        from agentic_szz.kg import reset_neo4j_database

        reset_neo4j_database()  # Full wipe before evaluation
        for case in cases:
            result = szz.predict(case.bfc)
    """
    from neo4j import GraphDatabase

    from agentic_szz.config.settings import get_config

    config = get_config()
    neo4j_config = config.neo4j

    driver = GraphDatabase.driver(neo4j_config.uri, auth=(neo4j_config.user, neo4j_config.password))

    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()
