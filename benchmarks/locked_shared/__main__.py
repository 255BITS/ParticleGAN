"""python -m benchmarks.locked_shared --reference /path/to/conceptmod"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import torch
import particlegan

from .reference import COMMIT, SOURCE_HASHES, run_reference
from .run import compare, markdown, run_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help=f"conceptmod checkout at {COMMIT}")
    parser.add_argument("--output", type=Path, default=Path("reports/locked_shared"))
    parser.add_argument("--reference-wheel", type=Path, help="optional PyPI particlegan 0.5.0 wheel to compare primitive sources")
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "conceptmod_commit": COMMIT,
              "pr36_commit": "eb18bc9ea69b9eb0a42f293c54491e6e06cf627d",
              "reference_source_sha256": SOURCE_HASHES,
              "rows": run_all()}
    package = Path(particlegan.__file__).parent
    report["particlegan_source_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in ("gan_loss.py", "grad_regularizers.py", "locked_shared.py", "particle_prior.py", "vicreg_loss.py")
    }
    if args.reference_wheel:
        with zipfile.ZipFile(args.reference_wheel) as wheel:
            hashes = {}
            for name in ("gan_loss.py", "grad_regularizers.py", "particle_prior.py", "vicreg_loss.py"):
                digest = hashlib.sha256(wheel.read(f"particlegan/{name}")).hexdigest()
                hashes[name] = digest
                if digest != report["particlegan_source_sha256"][name]:
                    raise ValueError(f"primitive differs from reference wheel: {name}")
        report["pypi_reference"] = {
            "filename": args.reference_wheel.name,
            "sha256": hashlib.sha256(args.reference_wheel.read_bytes()).hexdigest(),
            "source_sha256": hashes,
        }
    if args.reference:
        report["reference_rows"] = run_reference(args.reference)
        report["parity"] = compare(report["rows"], report["reference_rows"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    table = markdown(report)
    (args.output / "README.md").write_text(table)
    print(table)
    locked_ok = all(r["verdict"] == "PASS" for r in report["rows"] if r["arm"] == "locked_shared")
    parity_ok = all(c["match"] for c in report.get("parity", []))
    return 0 if locked_ok and parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
