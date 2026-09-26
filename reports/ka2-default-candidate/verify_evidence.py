"""Verify the archived candidate evidence without importing torch or training."""
from pathlib import Path
import hashlib
import json
import gzip


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["files"]:
        data = (root / entry["artifact"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["artifact_sha256"], entry["artifact"]

    def read(name):
        return json.loads((root / "evidence" / name).read_text())

    original = read("original-shift.json")
    extended = read("extended-shift.json")
    frozen = read("frozen-shift.json")
    receipt = read("original-mechanism-receipt.json")
    assert receipt["pure_a_calls"] == 799 and receipt["blend_calls"] == 2801
    assert original["diagnostic"] == extended["diagnostic"][:len(original["diagnostic"])]
    assert original["proof"]["initial_optimizers"] == extended["proof"]["initial_optimizers"]
    assert original["continued_hold"]["passing_checks"] == 120
    assert original["shift_recovery"]["deadline_window"]["passing_checks"] == 50
    assert frozen["shift_recovery"]["deadline_window"]["passing_checks"] == 0
    assert original["continued_hold"] == frozen["continued_hold"]
    window = [row for row in extended["diagnostic"] if 3520 <= row["step"] <= 4600]
    failures = [row["step"] for row in window if row["modes"] < 8 or row["hq"] < .90]
    assert len(window) == 109
    assert failures == [4280, 4300, 4310, 4320]
    assert extended["shift_recovery"]["passing_suffix"] == 28
    assert extended["shift_recovery"]["stable_from_step"] == 4330
    source = root / "source"
    assert (source / "shift.py").read_text().replace("steps=3600", "steps=4600") == (source / "extended_shift.py").read_text()

    mass = read("original-mass.json")
    seeds = [read(f"mass-seed{s}.json") for s in range(4)]
    assert mass["verdict"] == seeds[0]["verdict"]
    assert mass["proof"]["initial_optimizers"] == seeds[0]["proof"]["initial_optimizers"]
    assert mass["randomness"] == seeds[0]["randomness"]
    assert [row["status"] for row in seeds] == ["FAIL", "PASS", "FAIL", "PASS"]
    assert [row["verdict"]["convergence"]["passing_suffix"] for row in seeds] == [4, 8, 0, 8]
    excursion = next(row for row in mass["result"]["observations"] if row["step"] == 1000)
    assert excursion["component_covariance_error"] > .85
    assert excursion["component_min_eigen_ratio"] >= .15

    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "SELECTED_DEFAULT_PENDING_MERGE"
    assert summary["selection_criterion"]["deadline_all_81_required"] is False
    k3p = json.loads(gzip.decompress((root / "matched-replay/results/k3p-canonical.json.gz").read_bytes()))
    r2 = json.loads(gzip.decompress((root / "evidence/r2-shift.json.gz").read_bytes()))
    for name, data in (("KA2", original), ("K3P", k3p), ("R2", r2)):
        rows = [row for row in data["diagnostic"] if row["step"] > 2400]
        good = lambda row: row["modes"] >= 8 and row["hq"] >= .90
        first = next(row["step"] for row in rows if good(row))
        after = [row for row in rows if row["step"] >= first]
        settled = data["shift_recovery"]["stable_from_step"]
        final_run = [row for row in rows if row["step"] >= settled]
        assert summary["arrival_stability"][name] == dict(
            shift_step=2400, end_step=3600,
            prehold_pass=data["continued_hold"]["passing_checks"],
            prehold_checks=data["continued_hold"]["checks"],
            first_observed_arrival=first, first_observed_delay=first - 2400,
            checks_from_first_arrival=len(after), passing_from_first_arrival=sum(map(good, after)),
            published_settled_arrival=settled, published_settled_delay=settled - 2400,
            checks_from_settled_arrival=len(final_run), passing_from_settled_arrival=sum(map(good, final_run)))
    print(f"PASS: {len(manifest['files'])} artifact hashes; exact extension prefix; 105/109 extension checks; mass FAIL/PASS/FAIL/PASS. No training run.")


if __name__ == "__main__":
    main()
