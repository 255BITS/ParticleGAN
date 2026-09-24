"""Independently regrade archived constant-game waves and build a leaderboard.

The immutable manifest and archived executable sources bind each raw episode.
No successful endpoint is promoted to a sustained pass, and the production
common gate must reject every scratch episode. Run after copying RAM evidence
to durable storage; this script does not train models.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.optimistic_regrade import regrade_episode


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def collect(root: Path) -> dict:
    manifest = json.loads((root / "predeclared_manifest.json").read_text())
    report = json.loads((root / "screen_result.json").read_text())
    manifest_sha = digest(root / "predeclared_manifest.json")
    assert manifest_sha == (root / "predeclared_manifest.sha256").read_text().strip()
    assert report["status"] == "complete"
    assert len(manifest["rows"]) == len(report["rows"])
    observations = {row["id"]: row for row in report["rows"]}
    checked = []
    rows = []
    for declared in manifest["rows"]:
        assert digest(root / declared["config_file"]) == declared["config_sha256"]
        observed = observations[declared["id"]]
        assert observed["status"] != "ERROR"
        config = json.loads((root / declared["config_file"]).read_text())
        row = dict(id=declared["id"], core=declared["base"],
                   lr=config["lr"], beta2=config["betas"][1],
                   d_lr_mult=config["d_lr_mult"], prior_lr_mult=config["prior_lr_mult"],
                   alpha=declared["alpha"], amsgrad=declared.get("amsgrad", False),
                   source_commit=manifest["source_commit"],
                   config_sha256=declared["config_sha256"],
                   stages=[], skipped=observed["skipped"],
                   shared_gate_eligible=False)
        for attempt in observed["attempted"]:
            task = attempt["task"]
            directory = root / "runs" / row["id"] / task
            verdict = regrade_episode(
                directory, task=task, alpha=row["alpha"], amsgrad=row["amsgrad"],
                config_sha256=row["config_sha256"],
                optimizer_source_sha256=manifest["optimizer_source_sha256"],
                driver_source_sha256=manifest["driver_source_sha256"],
                regrader_source_sha256=manifest["regrader_source_sha256"],
                manifest_sha256=manifest_sha, source_commit=manifest["source_commit"],
            )
            assert verdict["status"] == attempt["status"]
            protocol = json.loads((directory / "protocol.json").read_text())
            assert protocol["source_sha256"] == manifest["source_sha256"]
            index = json.loads((directory / "index.json").read_text())
            episode = json.loads(gzip.decompress((directory / index["records"][0]["artifact"]).read_bytes()))
            terminal = [{key: point.get(key) for key in ("step", "modes", "hq")}
                        for point in episode["result"]["observations"][-5:]] if task == "mode_hold" else None
            row["stages"].append(dict(
                task=task, status=verdict["status"], final=attempt["final"],
                passing_suffix=attempt["passing_suffix"], terminal=terminal,
                terminal_passing_checks=sum(point["modes"] >= 8 and point["hq"] >= .9
                                            for point in terminal) if terminal else None,
                case_seconds=index["records"][0]["seconds"],
            ))
            checked.append(dict(id=row["id"], **verdict))
        rows.append(row)
    write(root / "independent_regrade.json", dict(status="PASS", episodes=len(checked), rows=checked))
    inventory = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
                 if path.is_file() and path.name not in ("inventory.json", "inventory.sha256")}
    write(root / "inventory.json", inventory)
    inventory_sha = digest(root / "inventory.json")
    (root / "inventory.sha256").write_text(inventory_sha + "\n")
    return dict(artifact=str(root), source_commit=manifest["source_commit"],
                manifest_sha256=manifest_sha, inventory_sha256=inventory_sha,
                files=len(inventory), episodes=len(checked), rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    waves = [collect(root) for root in args.roots]
    write(args.output, dict(status="complete", waves=waves,
                           candidates=sum(len(wave["rows"]) for wave in waves),
                           episodes=sum(wave["episodes"] for wave in waves),
                           common_gate_eligible=False))
    print(json.dumps(dict(status="independently_regraded", waves=len(waves),
                          candidates=sum(len(wave["rows"]) for wave in waves),
                          episodes=sum(wave["episodes"] for wave in waves))), flush=True)


if __name__ == "__main__":
    main()
