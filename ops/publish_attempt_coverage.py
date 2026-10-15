"""Publish a completed attempt batch from its dedicated, already-reviewed branch.

This hook stages generated reports only, preserves every scientific row, runs
the generated-artifact guard, and opens/merges a results PR through GitHub's
normal checks. It never changes recipes, code, tests or original receipts.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def main():
    root = Path(__file__).resolve().parents[1]
    publication = "reports/forge/technique-inventory.json"
    before = json.loads(run("git", "show", "HEAD:" + publication, cwd=root))
    after = json.loads((root / publication).read_text())
    for key in ("rows", "archived_evidence_rows", "archived_policies", "family_selection"):
        if before.get(key) != after.get(key):
            raise ValueError("publisher cannot change scientific/selection data: " + key)
    report = json.loads((root / "reports/forge/attempt-coverage.json").read_text())
    if report["summary"]["pending"] or report["summary"]["running"]:
        raise ValueError("publication requires a drained batch")
    changed = run("git", "diff", "--name-only", "HEAD", cwd=root).splitlines()
    exact = {publication, "reports/forge/technique-inventory.md", "reports/forge/EXPERIMENTS_BY_TIER.md",
             "reports/forge/attempt-coverage.json", "reports/forge/ATTEMPT_COVERAGE.md"}
    if any(path not in exact and not (path.startswith("reports/forge/families/") and path.endswith(".md")) for path in changed):
        raise ValueError("publisher found changes outside generated attempt reports")
    branch = run("git", "branch", "--show-current", cwd=root)
    if not branch.startswith("task/attempt-fill-"):
        raise ValueError("publisher requires the dedicated attempt-fill branch")
    if changed:
        subprocess.run([sys.executable, "-m", "pytest", "tests/test_forge_attempt_coverage.py",
                        "tests/test_forge_family_reports.py", "-q"], cwd=root, check=True)
        run("git", "add", "--", *changed, cwd=root)
        run("git", "commit", "-m", "Publish completed selected and research run attempts", cwd=root)
    run("git", "push", "origin", branch, cwd=root)
    run("git", "fetch", "origin", "develop", cwd=root)
    prs = json.loads(run("gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number", cwd=root))
    if prs:
        number = str(prs[0]["number"])
    elif run("git", "rev-list", "--count", "origin/develop..HEAD", cwd=root) == "0":
        return
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md") as body:
            body.write("The bounded backfill queue has completed all selected-family and non-seed research start attempts. "
                       "This update publishes source-bound outcomes and blocker reasons in the main technical leaderboard. "
                       "Original scientific rows, recipes and selection remain unchanged.\n\n"
                       f"Attempt coverage: {report['summary']['attempted']}/{report['summary']['total']}. "
                       "Validation: attempt-runner tests and generated family-page/link checks.\n")
            body.flush()
            url = run("gh", "pr", "create", "--base", "develop", "--head", branch,
                      "--title", "Publish completed research run-attempt coverage", "--body-file", body.name, cwd=root)
            number = url.rsplit("/", 1)[-1]
    checks = subprocess.run(["gh", "pr", "checks", number, "--required", "--watch", "--interval", "15"],
                            cwd=root, text=True, capture_output=True)
    if checks.returncode and "no required checks" not in (checks.stdout + checks.stderr).lower():
        raise RuntimeError("required GitHub checks did not pass: " + checks.stdout + checks.stderr)
    run("gh", "pr", "merge", number, "--merge", cwd=root)
    print("Completed attempt coverage queued for normal GitHub merge: PR " + number, flush=True)


if __name__ == "__main__":
    main()
