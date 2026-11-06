#!/usr/bin/env python3
"""Clone subject repositories for SZZ experiments.

Usage:
    python git_clone_subject_repos.py --dataset linux
    python git_clone_subject_repos.py --dataset apache
    python git_clone_subject_repos.py --dataset github
    python git_clone_subject_repos.py --dataset all
"""
import argparse
import json
import os
import pathlib
import subprocess

DATASET_DIR = pathlib.Path(__file__).resolve().parent
TARGET_ROOT = DATASET_DIR / "subject_repos"


def clone_if_missing(url: str, target: pathlib.Path) -> bool:
    """Clone a repository if it doesn't exist."""
    # Skip if already cloned
    if (target / ".git").exists():
        print(f"✓ Skip: {target.name}")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)

    # Try to clone, continue on failure
    # GIT_TERMINAL_PROMPT=0 disables interactive auth prompts
    # --progress shows download progress even when output is not a terminal
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        print(f"  Cloning {target.name}...")
        result = subprocess.run(
            ["git", "clone", "--progress", url, str(target)],
            env=env,
            # Stream stderr (where git progress goes) to terminal
            stdout=subprocess.DEVNULL,
            stderr=None,  # Show progress on stderr
        )
        if result.returncode == 0:
            print(f"✓ Cloned: {target.name}")
            return True
        else:
            print(f"✗ Failed: {target.name}")
            return False
    except Exception as e:
        print(f"✗ Error: {target.name} - {e}")
        return False


def clone_linux():
    """Clone Linux kernel repository."""
    print("=" * 60)
    print("Cloning Linux kernel repository...")
    print("=" * 60)

    # Linux is a single large repo
    url = "https://github.com/torvalds/linux.git"
    target = TARGET_ROOT / "linux"

    if clone_if_missing(url, target):
        print("✓ Linux kernel cloned successfully!")
        return []
    else:
        return ["linux"]


def clone_apache():
    """Clone Apache dataset repositories."""
    print("=" * 60)
    print("Cloning Apache dataset repositories...")
    print("=" * 60)

    failed = []
    with open(DATASET_DIR / "DS_APACHE.json") as fh:
        apache_names = {entry["repo_name"] for entry in json.load(fh)}

    for slug in sorted(apache_names):
        url = f"https://github.com/apache/{slug}.git"
        success = clone_if_missing(url, TARGET_ROOT / slug)
        if not success and not (TARGET_ROOT / slug / ".git").exists():
            failed.append(slug)

    if not failed:
        print(f"✓ All {len(apache_names)} Apache repositories cloned!")
    return failed



# Alternative URLs for repos that have been deleted, made private, or moved.
# Maps original repo_name -> (alternative_url, note)
# Use symlink_to to point to an already-cloned repo instead of re-cloning.
REPO_OVERRIDES = {
    # Deleted; maddinat0r's fork preserves the full commit history
    "Zeex/sampgdk": ("https://github.com/maddinat0r/sampgdk.git", None),
    # Deleted; gdnsd org repo preserves the full commit history
    "blblack/gdnsd": ("https://github.com/gdnsd/gdnsd.git", None),
    # Deleted fork of mesa; commits exist in the canonical mesa3d repo
    "bminor/mesa-mesa": (None, TARGET_ROOT / "mesa3d" / "mesa"),
    # Deleted; kraj/musl mirror preserves the full commit history
    "bminor/musl": ("https://github.com/kraj/musl.git", None),
    # Moved to freedesktop GitLab
    "mesa3d/mesa": ("https://gitlab.freedesktop.org/mesa/mesa.git", None),
    # Private fork of torvalds/linux; symlink to already-cloned linux repo
    "tmatsuya/linux-2.6": (None, TARGET_ROOT / "linux"),
}


def clone_github():
    """Clone GitHub dataset repositories."""
    print("=" * 60)
    print("Cloning GitHub dataset repositories...")
    print("=" * 60)

    failed = []
    with open(DATASET_DIR / "DS_GITHUB.json") as fh:
        gh_names = {entry["repo_name"] for entry in json.load(fh)}

    for name in sorted(gh_names):
        target = TARGET_ROOT / name
        if name in REPO_OVERRIDES:
            alt_url, symlink_target = REPO_OVERRIDES[name]
            if symlink_target is not None:
                # Point to an already-cloned repo via symlink
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(symlink_target.resolve())
                    print(f"✓ Symlinked {name} -> {symlink_target}")
                else:
                    print(f"✓ Skip: {name}")
                continue
            url = alt_url
            print(f"  Note: using alternative URL for {name}")
        else:
            url = f"https://github.com/{name}.git"
        success = clone_if_missing(url, target)
        if not success and not (target / ".git").exists():
            failed.append(name)

    if not failed:
        print(f"✓ All {len(gh_names)} GitHub repositories cloned!")

    # Create symlinks for path compatibility
    create_github_symlinks()
    return failed


def create_github_symlinks():
    """Create symlinks for GitHub repos for path compatibility.

    The final_all_dataset.json uses short repo names (e.g., "RMG-Java")
    but DS_GITHUB.json uses full paths (e.g., "nickvandewiele/RMG-Java").
    """
    print("\nCreating symlinks for path compatibility...")

    with open(DATASET_DIR / "DS_GITHUB.json") as f:
        github_data = json.load(f)
    with open(DATASET_DIR / "final_all_dataset.json") as f:
        all_data = json.load(f)

    # Create mapping from BFC to full repo path (from DS_GITHUB.json)
    bfc_to_full_path = {}
    for item in github_data:
        bfc = item.get("fix_commit_hash") or item.get("bug_fixing_commit")
        bfc_to_full_path[bfc] = item["repo_name"]

    # Find GitHub entries in final_all_dataset and create symlinks
    symlinks_created = 0
    for item in all_data:
        bfc = item["bug_fixing_commit"]
        short_name = item["repo_name"]
        short_path = TARGET_ROOT / short_name

        if bfc in bfc_to_full_path:
            full_name = bfc_to_full_path[bfc]
            full_path = TARGET_ROOT / full_name

            # Create symlink if short path doesn't exist but full path does
            if not short_path.exists() and full_path.exists():
                short_path.symlink_to(full_path.resolve())
                symlinks_created += 1

    if symlinks_created > 0:
        print(f"✓ Created {symlinks_created} symlinks for path compatibility")
    else:
        print("✓ No additional symlinks needed")


def main():
    parser = argparse.ArgumentParser(description="Clone subject repositories for SZZ experiments")
    parser.add_argument(
        "--dataset",
        choices=["linux", "apache", "github", "all"],
        required=True,
        help="Dataset to clone repositories for"
    )
    args = parser.parse_args()

    all_failed = []

    if args.dataset in ("linux", "all"):
        all_failed.extend(clone_linux())

    if args.dataset in ("apache", "all"):
        all_failed.extend(clone_apache())

    if args.dataset in ("github", "all"):
        all_failed.extend(clone_github())

    print("=" * 60)

    # Save failed repos to file for later investigation
    if all_failed:
        failed_file = DATASET_DIR / "failed_repos.txt"
        with open(failed_file, "w") as f:
            for repo in all_failed:
                f.write(f"{repo}\n")
        print(f"\nFailed repos ({len(all_failed)}) saved to: {failed_file}")
        print("These may require authentication or may not exist.")
    else:
        print("\n✓ All repositories cloned successfully!")


if __name__ == "__main__":
    main()
