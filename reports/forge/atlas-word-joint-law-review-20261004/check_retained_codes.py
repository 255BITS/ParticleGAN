"""Verify saved word-code evidence only; never import a model or evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


STEPS = [834, 1667, 2501, 3334, 4167, 5001, 5834, 6667, 7501, 8334,
         9168, 10001, 10834, 11668, 12501, 13334, 14168, 15001, 15835,
         16668, 17501, 18335, 19168, 20001]


def verify(report_path: Path) -> dict:
    import numpy as np

    report = json.loads(report_path.read_bytes())
    if report["schema"] != "pg_word_joint_law_review_v1":
        raise ValueError("unknown report schema")
    if report["reproducer"]["sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("passive checker changed")
    pins = report["inputs"]
    if len(pins) != report["input_files"] or len(pins) != 46:
        raise ValueError("proof input denominator changed")
    for name, pin in pins.items():
        raw = Path(name).read_bytes()
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError(f"pinned input changed: {Path(name).name}")
    analysis = json.loads(Path(report["prior_analysis"]["path"]).read_bytes())
    if any(pins.get(name) != pin for name, pin in analysis["inputs"].items()):
        raise ValueError("prior retained-input roster changed")
    snapshot = Path(report["source"]["snapshot_path"])
    manifest = json.loads((snapshot / "forge-source.json").read_bytes())
    if (manifest["digest"] != report["source"]["digest"]
            or manifest["origin_commit"] != report["source"]["commit"]):
        raise ValueError("source identity changed")
    for name, pin in report["source"]["consumed_files"].items():
        if manifest["files"].get(name) != pin["sha256"] or pins.get(str(snapshot / name)) != pin:
            raise ValueError("consumed source no longer matches frozen manifest")
    raw_result = json.loads(Path(report["raw_result"]["path"]).read_bytes())
    observation_receipts = raw_result["evidence"]["policy_observations"]
    descriptors = analysis["retained_array_descriptors"]
    if [r["completed_steps"] for r in observation_receipts] != STEPS:
        raise ValueError("recorded public-observation schedule changed")
    if [r["step"] for r in descriptors] != STEPS:
        raise ValueError("saved descriptor schedule changed")
    rows = []
    for descriptor, receipt in zip(descriptors, observation_receipts):
        step = descriptor["step"]
        selection = receipt["backend_selection"]
        if (selection["actual_backend"] != "knn"
                or selection["sampling_backend"] != "controller_reference"
                or selection["selection_reason"] != "caller_callbacks_own_representation"
                or receipt["selected_source"] != "fast" or receipt["output_noise"] is not False):
            raise ValueError("actual selected word law changed")
        paths = [name for name in pins if Path(name).name == f"step_{step:06d}.npz"]
        if len(paths) != 1:
            raise ValueError("missing or duplicate saved observation")
        with np.load(paths[0], allow_pickle=False) as arrays:
            prior = arrays["prior"]
            raw = arrays["generated_raw_code"]
            effective = arrays["generated_effective_code"]
            encoded = arrays["encoded_code"]
            reconstructed = arrays["reconstruction_effective_code"]
            if ([v.shape for v in [prior, raw, effective, encoded, reconstructed]]
                    != [(11, 2), (1024, 2), (1024, 2), (5, 2), (5, 2)]):
                raise ValueError("retained code shapes changed")
            if not all(v.dtype == np.dtype("float32") and np.isfinite(v).all()
                       for v in [prior, raw, effective, encoded, reconstructed]):
                raise ValueError("retained code dtype/health changed")
            delta = reconstructed.astype("float64") - encoded.astype("float64")
            if not np.array_equal(delta, np.asarray(descriptor["actual_dv12_displacements"])):
                raise ValueError("prior analysis displacement differs from saved code")
            displacement = np.linalg.norm(effective.astype("float64") - raw, axis=1)
            rows.append(dict(step=step, prior_unique_rows=int(len(np.unique(prior, axis=0))),
                generated_code_count=len(raw), generated_changed=int(np.any(effective != raw, axis=1).sum()),
                generated_effective_equal_any_encoder=int(np.any(np.all(
                    effective[:, None, :] == encoded[None, :, :], axis=2), axis=1).sum()),
                inverse_code_count=len(encoded), inverse_changed=int(np.any(reconstructed != encoded, axis=1).sum()),
                generated_displacement_l2_min=float(displacement.min()),
                generated_displacement_l2_max=float(displacement.max())))
    counts = dict(observations=len(rows), generated_codes=sum(r["generated_code_count"] for r in rows),
        generated_changed=sum(r["generated_changed"] for r in rows),
        generated_effective_equal_any_encoder=sum(r["generated_effective_equal_any_encoder"] for r in rows),
        inverse_codes=sum(r["inverse_code_count"] for r in rows),
        inverse_changed=sum(r["inverse_changed"] for r in rows),
        every_prior_table_has_eleven_distinct_rows=all(r["prior_unique_rows"] == 11 for r in rows))
    if counts != report["retained_codes"]["counts"] or rows != report["retained_codes"]["observations"]:
        raise ValueError("new count claims differ from retained arrays")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise ValueError("passive checker imported Torch")
    return dict(schema="pg_word_joint_law_passive_check_v1", status="PASS_SAVED_BYTES_ONLY",
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        checker_sha256=report["reproducer"]["sha256"], verified_input_files=len(pins), counts=counts,
        checkpoint_deserialization=False, model_construction=False, forwards=False,
        draws=False, official_scoring=False, numerical_regrading=False, qualification_input=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("review.json"))
    args = parser.parse_args()
    print(json.dumps(verify(args.report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
