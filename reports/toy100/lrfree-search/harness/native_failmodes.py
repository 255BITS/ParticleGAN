"""Codified pass criteria and failure-mode labels for the native 100-Gaussian tasks.

Thresholds are copied from the frozen gates (benchmarks/toy100/metrics.py, accuracy.py,
gate.py, accuracy_gate.py in the PR155 checkout); this module only labels results and never
changes a verdict. Scoring of record is NOISY (the model's sampling law).

Usage: python harness/native_failmodes.py [run_dir ...]   (default: every runs/*/{grid,rotated,staggered}100)
"""
import glob
import json
import sys

# Coverage gate: every one of the last MIN_STABLE_CHECKS (5) post-initial observations must pass all of these.
COVERAGE = {
    "modes": ("==", 100),
    "precision": (">=", 0.97),          # HQ radius 3 sigma; oracle 1 - exp(-9/2) = .98889
    "min_hq_mode_mass": (">=", 0.005),  # half of the uniform mode mass
    "max_mode_mass": ("<=", 0.02),      # twice the uniform mode mass
    "mass_tv": ("<=", 0.10),
    "min_cov_eig_ratio": (">=", 0.40),
    "max_cov_eig_ratio": ("<=", 1.70),
    "min_radial_median_ratio": (">=", 0.65),
    "max_radial_median_ratio": ("<=", 1.40),
}
# Accuracy gate: coverage PASS and these hold on the final five quality clouds (budget-1000 .. budget)
# and on an independent 100k-sample holdout.
ACCURACY = {
    "acc_mass_tv": ("<=", 0.06),
    "acc_center_rms_sigma": ("<=", 0.20),
    "abs_trace_bias": ("<=", 0.10),      # |acc_cov_trace_bias|
    "acc_radial_ks": ("<=", 0.04),
}

# Failure modes: (label, description, predicate on final metrics).
FAILMODES = [
    ("MISSING_MODE", "a mode is never covered or holds < .5% mass (e.g. grid100 stuck at 99 under DV12 LR collapse)",
     lambda m: m["modes"] < 100 or m["min_hq_mode_mass"] < 0.005),
    ("STRAGGLERS", "samples between modes: precision < .97 while all modes are covered (bridge particles)",
     lambda m: m["modes"] >= 100 and m["precision"] < 0.97),
    ("STREAKED", "modes are thin anisotropic streaks: cov-eigenvalue ratio outside .40-1.70 "
                 "(e.g. st-10: unsettled prior table + tiny learned sigma)",
     lambda m: m["min_cov_eig_ratio"] < 0.40 or m["max_cov_eig_ratio"] > 1.70),
    ("RADIAL_SHAPE", "radial profile wrong: median radius ratio outside .65-1.40 or radial KS > .04",
     lambda m: not 0.65 <= m["min_radial_median_ratio"] <= m["max_radial_median_ratio"] <= 1.40
     or (m.get("acc_radial_ks") is not None and m["acc_radial_ks"] > 0.04)),
    ("OFF_CENTRE", "mode centres off by > .20 sigma (frozen offsets under LR collapse, or jitter)",
     lambda m: m.get("acc_center_rms_sigma") is not None and m["acc_center_rms_sigma"] > 0.20),
    ("WIDTH_BIAS", "total per-mode variance off by > 10% (too narrow: small/clean sigma; too wide: uncontracted particles)",
     lambda m: m.get("acc_cov_trace_bias") is not None and abs(m["acc_cov_trace_bias"]) > 0.10),
    ("MASS_IMBALANCE", "mass distribution off: TV > .06 (accuracy) / > .10 (coverage) or a mode > 2% mass",
     lambda m: (m.get("acc_mass_tv") or m["mass_tv"]) > 0.06 or m["max_mode_mass"] > 0.02),
]


def noisy_final(result):
    """Final metrics under the scoring of record (noisy)."""
    clean_primary = result.get("eval_output_noise") is False or (
        "noisy_final" in result and result.get("eval_output_noise") is not True)
    return result.get("noisy_final") if clean_primary and result.get("noisy_final") else result.get("final", {})


def labels(result):
    m = noisy_final(result)
    out = [name for name, _, test in FAILMODES if test(m)]
    native = result.get("native", {})
    if not out and result.get("status") != "PASS":
        out.append("NOT_SUSTAINED")  # final checkpoint fine, but not for 5 terminal checks or not on the holdout
    return out, m, native


def main(paths):
    if not paths:
        paths = sorted(p for t in ("grid100", "rotated100", "staggered100")
                       for p in glob.glob(f"runs/*/{t}"))
    for path in paths:
        try:
            result = json.load(open(f"{path}/result.json"))
        except (OSError, json.JSONDecodeError):
            continue
        if not result.get("final"):
            continue
        out, m, native = labels(result)
        status = result["status"] if result.get("eval_output_noise") is not False else result.get("noisy_status")
        print(f"{path:55s} {status or '?':5s} modes {m.get('modes')} prec {m.get('precision', m.get('hq', 0)):.3f} "
              f"-> {', '.join(out) or 'PASS'}")


if __name__ == "__main__":
    main(sys.argv[1:])
