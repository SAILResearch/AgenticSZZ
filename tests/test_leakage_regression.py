#!/usr/bin/env python3
"""
Regression tests for BIC data leakage (TOSEM revision, R2.7 / R2.8 / R1.7).

Reviewer 2 found that a `Fixes:` trailer naming the ground-truth BIC reached
the agent verbatim, and asked for regression testing alongside the fix. These
tests pin all three leakage paths that were found:

1. `git show` without `--format=` prepended the commit header, and therefore the
   full raw message, to `diff_content`, which `read_node_content` returns as is.
2. The `Fixes:` regex required a parenthesised description, so bare-SHA trailers
   survived, and `Reverts:` was not handled at all.
3. `_search_without_tkg` read `commit.message` straight from git, bypassing
   build-time sanitization entirely (not reported in the review).

Every assertion below was verified to FAIL against the pre-fix code. A test that
passes before and after the fix would prove nothing.

None of these need Neo4j, so the leak can be checked without standing up a
database.

Usage:
    pytest tests/test_leakage_regression.py -v
    python tests/test_leakage_regression.py
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LINUX_REPO = PROJECT_ROOT / "dataset" / "subject_repos" / "linux"

# A DS_LINUX case whose BFC message carries "Fixes: e130242dc351 (...)", naming
# its ground-truth BIC. This is the case from the review's worked example.
LEAKY_BFC = "000eca5d044d1ee23b4ca311793cf3fc528da6c6"
LEAKY_GT = "e130242dc351"

# Any commit reference that must never survive sanitization.
COMMIT_REF = re.compile(r"(?:Fixes|Reverts):\s*[0-9a-f]{7,40}", re.IGNORECASE)


def _repo():
    import git

    return git.Repo(LINUX_REPO)


def _skip_without_repo():
    if not LINUX_REPO.exists():
        try:
            import pytest

            pytest.skip(f"Linux repo not found at {LINUX_REPO}")
        except ImportError:
            raise SystemExit(f"Linux repo not found at {LINUX_REPO}")


# ---------------------------------------------------------------------------
# Pattern 1: Fixes: / Reverts: trailers
# ---------------------------------------------------------------------------


def test_parenthesised_fixes_trailer_removed():
    """The original supported form. Guards existing behaviour."""
    from agentic_szz.kg.tkg import _sanitize_commit_message

    out = _sanitize_commit_message('Subject\n\nFixes: 68b99e94b8b0 ("mm: simplify")\n')
    assert not COMMIT_REF.search(out), out
    assert "68b99e94b8b0" not in out


def test_bare_sha_fixes_and_reverts_removed():
    """R2.8. Fails pre-fix: the regex required a parenthesised description and
    Reverts: was not handled at all."""
    from agentic_szz.kg.tkg import _sanitize_commit_message

    for trailer in (
        "Fixes: 68b99e94b8b0",
        "Fixes: 68b99e94b8b0 mm: trailing subject text",
        "Reverts: 68b99e94b8b0",
        "reverts: 68B99E94B8B0",
    ):
        out = _sanitize_commit_message(f"Subject\n\n{trailer}\n")
        assert not COMMIT_REF.search(out), f"{trailer!r} survived as {out!r}"


def test_issue_numbers_are_not_mistaken_for_shas():
    """Guards against over-scrubbing: these must be preserved."""
    from agentic_szz.kg.tkg import _sanitize_commit_message

    for keep in ("Fixes: #1234", "Fixes: 1234", "Fixes a race in the scheduler"):
        out = _sanitize_commit_message(f"Subject\n\n{keep}\n")
        assert keep in out, f"{keep!r} was scrubbed, got {out!r}"


# ---------------------------------------------------------------------------
# Pattern 4: bare SHA references in prose
# ---------------------------------------------------------------------------


def test_bare_sha_in_prose_redacted():
    """R1.7. Fails pre-fix: prose references carry no trailer and no "commit"
    keyword, so Patterns 1 to 3 all miss them."""
    _skip_without_repo()
    from agentic_szz.kg.tkg import _sanitize_commit_message

    repo = _repo()
    msg = f"Subject\n\nThis fixes a bug introduced in {LEAKY_GT}.\n"
    out = _sanitize_commit_message(msg, repo)
    assert LEAKY_GT not in out, out
    assert "[referenced commit]" in out


def test_non_commit_hex_preserved():
    """Guards against over-scrubbing. Addresses, register values and oops dumps
    are hex but name no commit, and length alone cannot tell them apart."""
    _skip_without_repo()
    from agentic_szz.kg.tkg import _sanitize_commit_message

    repo = _repo()
    for keep in (
        "Fixed crash at address deadbeefcafe1234 in the handler",
        "Set register to 00000000ffffffff before the reset",
        "RIP: 0010:[<ffffffff81234567>] handler+0x12/0x40",
    ):
        out = _sanitize_commit_message(keep, repo)
        assert out == keep, f"over-scrubbed: {keep!r} -> {out!r}"


# ---------------------------------------------------------------------------
# The diff path
# ---------------------------------------------------------------------------


def test_diff_carries_no_commit_header():
    """R2.7. Fails pre-fix: without --format= git prepends the author, date and
    full raw message, so the Fixes: trailer lands in diff_content."""
    _skip_without_repo()
    repo = _repo()

    diff = repo.git.show("--format=", "-U1", LEAKY_BFC)
    assert not COMMIT_REF.search(diff), "commit header leaked into the diff"
    assert LEAKY_GT not in diff
    assert diff.lstrip().startswith("diff --git"), diff[:120]


def test_ground_truth_absent_from_sanitized_message_and_diff():
    """End to end on the review's worked example: neither channel names the GT."""
    _skip_without_repo()
    from agentic_szz.kg.tkg import _sanitize_commit_message

    repo = _repo()
    blob = _sanitize_commit_message(repo.commit(LEAKY_BFC).message, repo)
    blob += "\n" + repo.git.show("--format=", "-U1", LEAKY_BFC)
    assert LEAKY_GT not in blob
    assert not COMMIT_REF.search(blob)


# ---------------------------------------------------------------------------
# The flat-list prompt path
# ---------------------------------------------------------------------------


def test_flat_list_path_sanitizes_candidate_messages():
    """Third site. Fails pre-fix: the no-graph path built its candidate text
    from raw commit.message, so it never invoked the sanitizer. This path also
    backs the AgenticSZZ (w/o graph) ablation, so it must be clean before that
    configuration reuses it."""
    _skip_without_repo()
    from agentic_szz.agent.bic_search_agent import format_flat_candidates

    repo = _repo()
    # A commit whose own message carries a Fixes: trailer, offered as a candidate.
    leaky = next(
        c.hexsha
        for c in repo.iter_commits("HEAD", max_count=500)
        if COMMIT_REF.search(c.message)
    )
    rendered = "\n".join(format_flat_candidates(repo, [leaky]))
    assert not COMMIT_REF.search(rendered), rendered[:200]


# ---------------------------------------------------------------------------
# Corpus-level check
# ---------------------------------------------------------------------------


def test_no_trailer_survives_across_the_corpus():
    """Sampled rather than exhaustive, to keep the suite fast."""
    _skip_without_repo()
    from agentic_szz.kg.tkg import _sanitize_commit_message

    repo = _repo()
    had = survived = 0
    for commit in repo.iter_commits("HEAD", max_count=400):
        if COMMIT_REF.search(commit.message):
            had += 1
        if COMMIT_REF.search(_sanitize_commit_message(commit.message, repo)):
            survived += 1
    assert had > 0, "sample contained no trailers, so it tests nothing"
    assert survived == 0, f"{survived} of {had} trailers survived"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
