"""Hash trainer state after a few hundred unchanged updates.

The observer stops the existing gate after N outer updates. It does not edit
rates, widths, or RNG draws. ``--require`` uses the scoring guard and exits 2
when the canonical env did not stick. Without it, the script still records
hashes so an Intel path and an AMD-like preload can be compared.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _digest(tensors):
    import torch
    pieces = []
    per = {}
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        per[name] = hashlib.sha256(raw).hexdigest()
        pieces.append(name.encode() + b"\0" + raw)
    return {"sha256": hashlib.sha256(b"".join(pieces)).hexdigest(), "count": len(per), "tensors": per}


def _compare(left, right):
    import torch
    a = torch.load(left, map_location="cpu", weights_only=False)
    b = torch.load(right, map_location="cpu", weights_only=False)
    names = sorted(set(a) | set(b))
    max_abs = 0.0
    mismatched = []
    for name in names:
        if name not in a or name not in b:
            mismatched.append(name)
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            mismatched.append(name)
            continue
        if x.dtype.is_floating_point:
            delta = float((x - y).abs().max())
            max_abs = max(max_abs, delta)
            if x.view(torch.uint8).ne(y.view(torch.uint8)).any():
                mismatched.append(name)
        elif not torch.equal(x, y):
            mismatched.append(name)
    nonfloat_bad = []
    for name in mismatched:
        if name in a and name in b and a[name].dtype.is_floating_point:
            continue
        nonfloat_bad.append(name)
    return {
        "max_abs": max_abs,
        "bit_identical": not mismatched,
        "within_1e-7": max_abs <= 1e-7 and not nonfloat_bad,
        "mismatched_count": len(mismatched),
        "nonfloat_mismatched": nonfloat_bad,
    }


def _install_repo(repo):
    """Run the committed gate and guard, not a drifted checkout copy."""
    repo = repo.resolve()
    dest_guard = repo / "benchmarks/toy100/canonical_env.py"
    dest_probe = repo / "reports/toy100/gan_followup_probe.py"
    dest_guard.write_bytes((ROOT / "benchmarks/toy100/canonical_env.py").read_bytes())
    dest_probe.write_bytes((ROOT / "reports/toy100/gan_followup_probe.py").read_bytes())
    sys.path.insert(0, str(repo))
    return repo


def run_prefix(repo, output, method, updates, require):
    import runpy
    import torch
    repo = _install_repo(repo)
    output.mkdir(parents=True, exist_ok=False)
    import benchmarks.toy100.canonical_env as canonical
    from reports.toy100.alternating_curvature_scratch import BothBoundRecorder
    rows = []
    initial = {}
    final = {}

    class PrefixComplete(Exception):
        pass

    original = BothBoundRecorder.phases

    def observed(self, step, opt_d, opt_g, local):
        def grab(bucket):
            tensors = {}
            for role in ("generator", "critic", "prior"):
                for name, value in local[role].state_dict().items():
                    tensors[f"{role}.{name}"] = value.detach().cpu().clone()
            tensors["torch_rng"] = torch.get_rng_state().clone()
            for key, value in local.items():
                if isinstance(value, torch.Generator):
                    tensors[f"rng.{key}"] = value.get_state().clone()
            bucket.update(tensors)
        if not initial:
            grab(initial)
        for phase in original(self, step, opt_d, opt_g, local):
            yield phase
        rows.append({
            "outer_step": self.records[-1].get("outer_step"),
            "critic_sharpness": self.records[-1].get("critic_sharpness"),
            "g_factor": self.records[-1].get("g", {}).get("factor"),
        })
        if len(rows) >= updates:
            grab(final)
            raise PrefixComplete()

    BothBoundRecorder.phases = observed
    if require:
        canonical.require_canonical_env()
    else:
        def _diagnostic():
            canonical.apply_runtime()
            canonical._RECEIPT = canonical.collect_receipt()
            return canonical._RECEIPT
        canonical.require_canonical_env = _diagnostic
    runner = repo / "reports/toy100/gan_followup_probe.py"
    # stay() rejects a budget under the frozen 1,200 updates. The observer
    # raises after `updates` completed steps, so the budget is only a ceiling.
    sys.argv = [str(runner), "--phase", "stay", "--method", method,
                "--output", str(output / "run"), "--steps", "2400"]
    try:
        runpy.run_path(str(runner), run_name="__main__")
    except PrefixComplete:
        pass
    finally:
        BothBoundRecorder.phases = original
    torch.save(final, output / "final-state.pt")
    receipt = canonical.collect_receipt()
    expected = canonical.expected_cbwr_raw(os.environ.get("MKL_CBWR", ""))
    problems = canonical.env_problems()
    if canonical.read_cbwr_raw() != expected:
        problems.append(
            f"effective {canonical.read_cbwr_raw()} != requested {os.environ.get('MKL_CBWR')}"
        )
    value = {
        "method": method,
        "updates": len(rows),
        "guard_problems": problems,
        "harness": receipt,
        "initial": _digest(initial),
        "final": _digest(final),
        "rows": rows,
    }
    # Drop per-tensor hashes from the human log; they remain in the json.
    (output / "prefix.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "updates": len(rows),
        "final_sha256": value["final"]["sha256"],
        "cbwr_effective": receipt["cbwr_effective"],
        "cbwr_raw": receipt["cbwr_raw"],
        "guard_ok": not value["guard_problems"],
        "sharpness": [row["critic_sharpness"] for row in rows[:3]],
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", default="holdw15")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--require", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        left, right = args.compare
        report = _compare(left / "final-state.pt", right / "final-state.pt")
        a = json.loads((left / "prefix.json").read_text())
        b = json.loads((right / "prefix.json").read_text())
        report["hash_equal"] = a["final"]["sha256"] == b["final"]["sha256"]
        report["left_cbwr"] = a["harness"]["cbwr_effective"]
        report["right_cbwr"] = b["harness"]["cbwr_effective"]
        report["left_guard_ok"] = not a["guard_problems"]
        report["right_guard_ok"] = not b["guard_problems"]
        print(json.dumps(report), flush=True)
        return
    if not args.repo or not args.output:
        parser.error("--repo and --output are required")
    run_prefix(args.repo, args.output, args.method, args.updates, args.require)


if __name__ == "__main__":
    main()
