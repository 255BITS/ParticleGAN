"""Fixed covariance-normalized feature sidecar; see REPAIR_SPEC.md."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import probe


def compare(a, b):
    sa, sb = a["stats"], b["stats"]
    out = {
        "logit_max_abs": float((a["logits"] - b["logits"]).abs().max()),
        "covariance_condition": b["covariance_condition"],
        "dR_abs_delta": abs(sa["dR"] - sb["dR"]),
        "dF_abs_delta": abs(sa["dF"] - sb["dF"]),
        "flag_count": len(b["flag_ids"]),
        "flag_ids_equal": a["flag_ids"] == b["flag_ids"],
        "move_count": len(b["pairs"]),
        "move_pairs_equal": a["pairs"] == b["pairs"],
        "final_z_max_abs_delta": float((a["final_z"] - b["final_z"]).abs().max()),
    }
    for name in ("rR", "rF", "x", "scores", "p"):
        out[f"{name}_max_abs_delta"] = float((sa[name] - sb[name]).abs().max())
    out["continuous_pass"] = max(out[k] for k in out if k.endswith("_abs_delta")
                                  and k != "final_z_max_abs_delta") < 1e-8
    out["decision_pass"] = (out["flag_ids_equal"] and out["move_pairs_equal"]
                            and out["final_z_max_abs_delta"] == 0.)
    return out


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260929)
    q0 = torch.from_numpy(rng.standard_normal((probe.N, 2)))
    R = torch.from_numpy(rng.standard_normal((probe.N, 2)))
    I = torch.eye(2, dtype=torch.float64)
    variants = {
        "axis_x16": torch.diag(torch.tensor([16., 1.], dtype=torch.float64)),
        "axis_y16": torch.diag(torch.tensor([1., 16.], dtype=torch.float64)),
        "uniform16": 16 * I,
        "rotation90": torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64),
        "nondiagonal": torch.tensor([[2., .7], [.3, 1.1]], dtype=torch.float64),
    }
    result = {"source_sha256": hashlib.sha256(probe.SOURCE.read_bytes()).hexdigest(),
              "N": probe.N, "cases": {}}
    for mode, knn in (("normal", probe.ORIGINAL_KNN),
                      ("exhaustive_float64", probe.exhaustive_knn)):
        probe.bd_module._knn = knn
        result["cases"][mode] = {}
        for case in ("null", "shifted"):
            q = q0.clone()
            if case == "shifted":
                q[:256, 0] += 2.5
            print(f"{mode} {case}: base", flush=True)
            base = probe.run_one(q, R, I, whiten=True)
            entry = {"base": {"condition": base["covariance_condition"],
                               "flag_count": len(base["flag_ids"]),
                               "move_count": len(base["pairs"]),
                               "flag_ids": base["flag_ids"], "pairs": base["pairs"]},
                     "comparisons": {}}
            for name, S in variants.items():
                print(f"{mode} {case}: {name}", flush=True)
                other = probe.run_one(q, R, S, whiten=True)
                entry["comparisons"][name] = compare(base, other)
            result["cases"][mode][case] = entry
            Path(__file__).with_name("whiten_results.json").write_text(json.dumps(result, indent=2) + "\n")
    probe.bd_module._knn = probe.ORIGINAL_KNN
    assert all(c["continuous_pass"] and c["decision_pass"]
               for cases in result["cases"].values() for e in cases.values()
               for c in e["comparisons"].values())
    print("all sidecar checks passed", flush=True)


if __name__ == "__main__":
    main()
