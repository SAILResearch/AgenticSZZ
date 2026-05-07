"""
Blame Analyzer - Core module for git blame-based BIC identification

This module provides:
1. Extraction of deleted lines from bug-fixing commits (BFCs)
2. Git blame to find commits that introduced those lines
3. Confidence scoring based on blame distribution
4. Ground truth comparison for evaluation

Usage:
    analyzer = BlameAnalyzer(repo_path)
    result = analyzer.analyze(bfc_hash)

    # For evaluation
    category, details = analyzer.compare_with_ground_truth(result, ground_truth_bics)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import git


@dataclass
class FileBlameInfo:
    """Per-file blame information"""

    deleted_lines: List[int]
    blame_commits: Dict[int, str]  # line_number -> commit_sha


@dataclass
class BlameResult:
    """Result from blame analysis"""

    bfc: str
    parent: str  # Parent commit hash
    is_blameable: bool  # Has modified/deleted lines
    blame_commits: List[str]  # Unique blame commits found
    confidence: float  # Confidence in top candidate (0.0-1.0)
    top_candidate: Optional[str]  # Best BIC candidate

    # Per-file details
    files: Dict[str, FileBlameInfo] = field(default_factory=dict)

    # Summary statistics
    files_analyzed: int = 0
    total_deleted_lines: int = 0
    total_added_lines: int = 0
    blame_distribution: Dict[str, int] = field(
        default_factory=dict
    )  # commit -> line count


class BlameAnalyzer:
    """
    Blame Analyzer for BIC identification

    Extracts blame commits from bug-fixing commits and computes confidence scores.

    Confidence scoring logic:
    - Single blame commit for all lines -> high confidence (0.95)
    - Dominant commit (>70% of lines) -> medium-high confidence (0.8)
    - Multiple commits, no clear winner -> low confidence (0.5)
    - Blameless case (no deleted lines) -> zero confidence (0.0)
    
    Context expansion (experimental):
    - context_expand_lines: Expand ±N lines around each deleted line for ALL cases
    - This can help find ancestor commits when the actual bug is in nearby code
    - Inspired by LLM4SZZ's context expansion strategy
    """

    def __init__(
        self,
        repo_path: str,
        use_context_fallback: bool = False,
        context_expand_lines: int = 0,
    ):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.use_context_fallback = use_context_fallback  # For blameless cases
        self.context_expand_lines = context_expand_lines  # Expand ±N around ALL deleted lines

    def analyze(self, bfc: str) -> BlameResult:
        """
        Analyze a bug-fixing commit to extract blame commits

        Args:
            bfc: Bug-fixing commit hash

        Returns:
            BlameResult with blame commits, confidence score, and per-file details
        """
        commit = self.repo.commit(bfc)
        if not commit.parents:
            return BlameResult(
                bfc=bfc,
                parent="",
                is_blameable=False,
                blame_commits=[],
                confidence=0.0,
                top_candidate=None,
            )

        parent = commit.parents[0]
        diffs = parent.diff(commit, create_patch=True)

        blame_distribution: Dict[str, int] = {}
        files: Dict[str, FileBlameInfo] = {}
        total_deleted_lines = 0
        total_added_lines = 0
        files_analyzed = 0
        
        # Two-pass approach for context fallback:
        # Pass 1: Check if ANY file has deleted lines (determines if BFC is blameable)
        # Pass 2: Only use context fallback if BFC is truly blameless (no deleted lines anywhere)
        
        # Pass 1: Collect all diff info and check for deleted lines
        diff_info = []
        has_any_deleted_lines = False
        
        for diff in diffs:
            if diff.a_path is None:  # New file, skip
                continue
            if not diff.diff:
                continue
                
            diff_text = diff.diff.decode("utf-8", errors="ignore")
            deleted_lines = self._extract_deleted_lines(diff_text)
            added_line_numbers = self._extract_added_line_numbers(diff_text)
            total_added_lines += len(added_line_numbers)
            
            if deleted_lines:
                has_any_deleted_lines = True
            
            diff_info.append({
                'a_path': diff.a_path,
                'deleted_lines': deleted_lines,
                'added_line_numbers': added_line_numbers,
            })
        
        # Pass 2: Process files, only using fallback if truly blameless
        for info in diff_info:
            deleted_lines = info['deleted_lines']
            added_line_numbers = info['added_line_numbers']
            a_path = info['a_path']
            
            # Context fallback: ONLY if entire BFC is blameless (no deleted lines anywhere)
            # This prevents per-file fallback from polluting blameable cases
            if self.use_context_fallback and not has_any_deleted_lines and added_line_numbers:
                context_lines = self._get_context_lines(
                    added_line_numbers, a_path, parent.hexsha
                )
                if context_lines:
                    deleted_lines = context_lines  # Blame these instead

            # Context expansion for ALL deleted lines (experimental, for ancestor detection)
            # This expands ±N lines around each deleted line to catch nearby buggy code
            # that may have been introduced by an ancestor commit
            if self.context_expand_lines > 0 and deleted_lines:
                expanded_lines = self._expand_context_around_lines(
                    deleted_lines, a_path, parent.hexsha, self.context_expand_lines
                )
                if expanded_lines:
                    deleted_lines = expanded_lines

            if not deleted_lines:
                continue

            files_analyzed += 1
            total_deleted_lines += len(deleted_lines)

            # Blame those lines at parent commit
            blame_map = self._blame_lines(a_path, deleted_lines, parent.hexsha)

            # Store per-file info
            files[a_path] = FileBlameInfo(
                deleted_lines=sorted(deleted_lines), blame_commits=blame_map
            )

            # Aggregate blame distribution
            for line_no, commit_sha in blame_map.items():
                blame_distribution[commit_sha] = (
                    blame_distribution.get(commit_sha, 0) + 1
                )

        # Compute results
        is_blameable = total_deleted_lines > 0
        # Sort blame commits chronologically (most recent first) for traditional SZZ
        # This provides a simple baseline - line counts still available for TKG/agent
        blame_commits = sorted(
            blame_distribution.keys(),
            key=lambda x: self.repo.commit(x).committed_datetime,
            reverse=True
        )
        confidence, top_candidate = self._compute_confidence(
            blame_distribution, total_deleted_lines, blame_commits
        )

        return BlameResult(
            bfc=bfc,
            parent=parent.hexsha,
            is_blameable=is_blameable,
            blame_commits=blame_commits,
            confidence=confidence,
            top_candidate=top_candidate,
            files=files,
            files_analyzed=files_analyzed,
            total_deleted_lines=total_deleted_lines,
            total_added_lines=total_added_lines,
            blame_distribution=blame_distribution,
        )

    # =========================================================================
    # Ground Truth Comparison (for evaluation)
    # =========================================================================

    def compare_with_ground_truth(
        self, result: BlameResult, ground_truth_bics: List[str]
    ) -> Tuple[str, Dict]:
        """
        Compare blame result with ground truth BICs

        Args:
            result: BlameResult from analyze()
            ground_truth_bics: List of ground truth BIC commit hashes

        Returns:
            Tuple of (category, details) where:
            - category: "direct", "ancestor", "descendant", or "others"
            - details: Dictionary with match information

        Categories:
            - direct: GT BIC is in the blame commits
            - ancestor: GT BIC is an ancestor of a blame commit
            - descendant: GT BIC is a descendant of a blame commit
            - others: No relationship found (blameable but unrelated to blame)
        """
        all_blame_commits = set(result.blame_commits)

        for gt_bic in ground_truth_bics:
            # Check direct match (prefix match for short hashes)
            if any(gt_bic in bc for bc in all_blame_commits):
                return "direct", {"ground_truth": gt_bic, "matched_in_blame": True}

            # Check if ground truth is ancestor of any blame commit
            for blame_commit in all_blame_commits:
                if self.is_ancestor(gt_bic, blame_commit):
                    depth = self.get_commit_depth(gt_bic, blame_commit)
                    return "ancestor", {
                        "ground_truth": gt_bic,
                        "blame_commit": blame_commit,
                        "ancestor_depth": depth,
                    }

            # Check if ground truth is descendant of any blame commit
            for blame_commit in all_blame_commits:
                if self.is_ancestor(blame_commit, gt_bic):
                    depth = self.get_commit_depth(blame_commit, gt_bic)
                    return "descendant", {
                        "ground_truth": gt_bic,
                        "blame_commit": blame_commit,
                        "descendant_depth": depth,
                    }

        return "others", {
            "ground_truth": ground_truth_bics,
            "num_blame_commits": len(all_blame_commits),
        }

    def is_ancestor(self, ancestor_sha: str, descendant_sha: str) -> bool:
        """Check if ancestor_sha is an ancestor of descendant_sha"""
        try:
            self.repo.git.merge_base("--is-ancestor", ancestor_sha, descendant_sha)
            return True
        except Exception:
            return False

    def get_commit_depth(self, ancestor_sha: str, descendant_sha: str) -> int:
        """
        Get number of commits between ancestor and descendant (DAG depth)

        Returns:
            Number of commits, or -1 if not an ancestor
        """
        if not self.is_ancestor(ancestor_sha, descendant_sha):
            return -1

        try:
            count_str = self.repo.git.rev_list(
                f"{ancestor_sha}..{descendant_sha}", count=True
            )
            return int(count_str.strip())
        except Exception:
            return -1

    def get_file_history_depth(
        self, commit1: str, commit2: str, file_paths: List[str]
    ) -> int:
        """
        Get number of commits between two commits that modified any of the specified files.

        This is the "file-history depth" - much more relevant than DAG depth for SZZ
        since we only need to traverse commits that touched the same files.

        Args:
            commit1: First commit (older)
            commit2: Second commit (newer)
            file_paths: List of file paths to check

        Returns:
            Number of commits touching these files between commit1 and commit2,
            or -1 if error
        """
        if not file_paths:
            return -1

        try:
            # Get commits between commit1 and commit2 that touched any of these files
            # Use "--follow" to track file renames
            commits_output = self.repo.git.log(
                f"{commit1}..{commit2}", "--format=%H", "--follow", "--", *file_paths
            )

            if not commits_output.strip():
                return 0

            # Count non-empty lines
            commits = [c for c in commits_output.strip().split("\n") if c]
            return len(commits)
        except Exception:
            return -1

    def get_min_file_history_depth(
        self, blame_commits: List[str], target_commit: str, file_paths: List[str]
    ) -> Tuple[int, Optional[str]]:
        """
        Get minimum file-history depth from any blame commit to the target commit.

        Args:
            blame_commits: List of blame commit hashes
            target_commit: Target commit (usually GT BIC)
            file_paths: Files modified in BFC

        Returns:
            Tuple of (min_depth, closest_blame_commit)
        """
        min_depth = float("inf")
        closest_commit = None

        for blame_commit in blame_commits:
            # Check both directions (target could be ancestor or descendant)
            if self.is_ancestor(target_commit, blame_commit):
                # target is ancestor of blame
                depth = self.get_file_history_depth(
                    target_commit, blame_commit, file_paths
                )
            elif self.is_ancestor(blame_commit, target_commit):
                # blame is ancestor of target (descendant case)
                depth = self.get_file_history_depth(
                    blame_commit, target_commit, file_paths
                )
            else:
                continue  # No relationship

            if depth >= 0 and depth < min_depth:
                min_depth = depth
                closest_commit = blame_commit

        if min_depth == float("inf"):
            return -1, None
        return int(min_depth), closest_commit

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _extract_deleted_lines(self, diff_text: str) -> List[int]:
        """Extract line numbers of deleted lines from unified diff"""
        deleted_lines = []
        current_old_line = 0

        for line in diff_text.split("\n"):
            if line.startswith("@@"):
                match = re.match(r"@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
                if match:
                    current_old_line = int(match.group(1))
            elif line.startswith("-") and not line.startswith("---"):
                deleted_lines.append(current_old_line)
                current_old_line += 1
            elif line.startswith(" "):
                current_old_line += 1

        return deleted_lines

    def _extract_added_line_numbers(self, diff_text: str) -> List[int]:
        """Extract line numbers of added lines from unified diff (in new file)"""
        added_lines = []
        current_new_line = 0

        for line in diff_text.split("\n"):
            if line.startswith("@@"):
                match = re.match(r"@@ -\d+,?\d* \+(\d+),?\d* @@", line)
                if match:
                    current_new_line = int(match.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                added_lines.append(current_new_line)
                current_new_line += 1
            elif line.startswith(" "):
                current_new_line += 1
            # Don't increment for deleted lines (they don't exist in new file)

        return added_lines

    def _get_context_lines(
        self, added_lines: List[int], file_path: str, parent_rev: str
    ) -> List[int]:
        """
        Get context lines (±2) around added lines for blameless cases.
        These are lines in the parent commit that we can blame.
        """
        if not added_lines:
            return []

        # Get ±2 context lines around each added line
        context_set = set()
        for line in added_lines:
            for offset in range(-2, 3):  # -2, -1, 0, +1, +2
                ctx_line = line + offset
                if ctx_line > 0:
                    context_set.add(ctx_line)

        # Get file length in parent to filter valid lines
        try:
            parent_content = self.repo.git.show(f"{parent_rev}:{file_path}")
            parent_line_count = len(parent_content.split("\n"))
            context_lines = [
                ln for ln in sorted(context_set) if ln <= parent_line_count
            ]
            return context_lines
        except Exception:
            return []  # File might not exist in parent

    def _expand_context_around_lines(
        self,
        line_numbers: List[int],
        file_path: str,
        rev: str,
        expand_n: int,
    ) -> List[int]:
        """
        Expand context around each line by ±N lines.
        
        This helps capture ancestor commits by blaming surrounding context
        that may point to earlier commits than direct blame.
        
        Args:
            line_numbers: Original line numbers to expand around
            file_path: Path to file in repo
            rev: Git revision to check file length
            expand_n: Number of lines to expand in each direction
            
        Returns:
            Expanded and deduplicated list of line numbers
        """
        if not line_numbers or expand_n <= 0:
            return line_numbers
            
        try:
            # Get file content at revision to know line count
            blob = self.repo.commit(rev).tree / file_path
            parent_line_count = blob.data_stream.read().decode("utf-8", errors="replace").count("\n") + 1
            
            # Expand around each line
            expanded_set = set()
            for ln in line_numbers:
                for offset in range(-expand_n, expand_n + 1):
                    new_ln = ln + offset
                    if 1 <= new_ln <= parent_line_count:
                        expanded_set.add(new_ln)
            
            return sorted(expanded_set)
        except Exception:
            return line_numbers  # Fallback to original if file not found

    def _blame_lines(
        self, file_path: str, line_numbers: List[int], rev: str
    ) -> Dict[int, str]:
        """Get blame commits for specific lines"""
        if not line_numbers:
            return {}

        blame_map = {}
        line_ranges = self._make_line_ranges(line_numbers)

        try:
            for entry in self.repo.blame_incremental(
                rev=rev,
                file=file_path,
                L=line_ranges,
                w=True,  # Ignore whitespace
            ):
                for lineno in entry.linenos:
                    blame_map[lineno] = entry.commit.hexsha
        except Exception:
            pass  # File might not exist at revision

        return blame_map

    def _make_line_ranges(self, line_numbers: List[int]) -> List[str]:
        """
        Convert line numbers to git blame -L format

        Returns list of range strings like ["1,3", "5,6"] for GitPython's blame_incremental
        """
        if not line_numbers:
            return []

        sorted_lines = sorted(set(line_numbers))
        ranges = []
        start = sorted_lines[0]
        end = start

        for line in sorted_lines[1:]:
            if line == end + 1:
                end = line
            else:
                ranges.append(f"{start},{end}")
                start = end = line
        ranges.append(f"{start},{end}")

        return ranges  # Return list, not comma-joined string!

    def _compute_confidence(
        self, blame_dist: Dict[str, int], total_lines: int, blame_commits: List[str]
    ) -> Tuple[float, Optional[str]]:
        """
        Compute confidence score based on blame distribution.
        
        Uses traditional SZZ approach: most recent commit is the top candidate.
        Confidence still based on line dominance for consistency with old behavior.

        Args:
            blame_dist: Mapping of commit SHA to line count
            total_lines: Total number of blamed lines
            blame_commits: List of blame commits sorted chronologically (most recent first)

        Returns:
            (confidence, top_candidate) where top_candidate is most recent commit
        """
        if not blame_dist or total_lines == 0:
            return 0.0, None

        # Top candidate: most recent commit (first in chronologically sorted list)
        top_candidate = blame_commits[0] if blame_commits else None
        
        # Confidence: still based on line dominance for consistency
        # Find the commit with most blamed lines for dominance calculation
        sorted_by_lines = sorted(blame_dist.items(), key=lambda x: -x[1])
        most_blamed_commit, top_count = sorted_by_lines[0]

        # Single commit blamed for everything
        if len(blame_dist) == 1:
            return 0.95, top_candidate

        # Calculate dominance ratio (of most blamed commit, not necessarily top_candidate)
        dominance = top_count / total_lines

        if dominance >= 0.9:
            confidence = 0.9
        elif dominance >= 0.7:
            confidence = 0.8
        elif dominance >= 0.5:
            confidence = 0.6
        else:
            confidence = 0.4

        return confidence, top_candidate
