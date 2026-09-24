"""Cheap own-state diagnostic using the existing dense continuation validator."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import tarfile

import stability_runner as runner
from particle_geometry import geometry_policy
from continuous_screen import verify_receipt
from critic_signal_screen import append
import geometry_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    status = json.loads((candidate / "status.json").read_text())
    ring = next(r for r in status["stages"] if r["gate"] == "mode_hold")
    config = json.loads((candidate / "config.json").read_text())
    options = json.loads((candidate / "options.json").read_text())
    state_sha = ring["checkpoint_sha256"]

    def verify():
        assert ring["status"] == "PASS"
        assert runner.sha(candidate / "mode_hold/final-state.pt") == state_sha
        receipt = json.loads(gzip.decompress((candidate / "mode_hold/signal-policy.json.gz").read_bytes()))
        verify_receipt(receipt, config, task="mode_hold")
        assert receipt["particle_geometry"]["transformed_tensors"] == 1200
        assert config["lr_floor"] == 1 and config["prior_reg"] == 0

    runner.CANDIDATE = candidate
    runner.STATE_SHA = state_sha
    runner.ARCHIVE_SHA = runner.sha(candidate.parent / "source.tar.gz")
    runner.verify_sources = verify
    runner.signal_policy = geometry_policy
    runner.POLICIES["control"]["family"] = "same-policy own cold particle geometry"
    result = runner.run("control", args.output, 200)
    result.update(gate="own_cold_diagnostic_200", borrowed_H_state=False,
                  status="PASS" if result["status"] == "SHORT_PASS" else result["status"],
                  qualification="diagnostic only; broader cold gates failed or unrun")
    declaration = json.loads((args.output / "declaration.json").read_text())
    declaration.update(state_origin="own same-policy cold acquisition",
                       scope="cheap blocker diagnostic; not promotion", options=options)
    runner.write(args.output / "declaration.json", declaration)
    sources = geometry_runner.sources()
    for path in (Path(__file__), Path(runner.__file__)):
        sources[str(path.relative_to(runner.REPO))] = runner.sha(path)
    runner.write(args.output / "source-manifest.json", sources)
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for name in sources:
            archive.add(runner.REPO / name, arcname=name)
    runner.write(args.output / "summary.json", result)
    append(args.ledger, dict(candidate=candidate.name, gate=result["gate"], status=result["status"],
                            seconds=result["elapsed_seconds"], metrics=result["window"],
                            artifact=str((args.output / "summary.json").resolve())))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
