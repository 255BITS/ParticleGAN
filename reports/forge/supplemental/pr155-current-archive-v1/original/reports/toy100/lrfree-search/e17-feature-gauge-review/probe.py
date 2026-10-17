"""Fixed E17 critic-feature gauge audit; see SPEC.md before interpreting results."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SOURCE = Path("/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17/particlegan/birth_death.py")
SPEC = importlib.util.spec_from_file_location("e17_birth_death_audit", SOURCE)
assert SPEC and SPEC.loader
bd_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bd_module)
ParticleBirthDeath = bd_module.ParticleBirthDeath
ORIGINAL_KNN = bd_module._knn
N = 1024


def exhaustive_knn(query, points, k, exclude=None, chunk=128, shortlist_dtype=None):
    """Reference full distance matrix in float64, without any shortlist."""
    del shortlist_dtype
    ds, ids = [], []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk].double()
        d = torch.linalg.vector_norm(q[:, None, :] - points[None, :, :].double(), dim=2)
        if exclude is not None:
            d.scatter_(1, exclude[start:start + chunk, None], float("inf"))
        v, ix = d.topk(k, dim=1, largest=False)
        v, order = v.sort(1)
        ds.append(v)
        ids.append(ix.gather(1, order))
    return torch.cat(ds), torch.cat(ids)


class Critic(torch.nn.Module):
    def __init__(self, S):
        super().__init__()
        self.hidden = torch.nn.Linear(2, 2, dtype=torch.float64)
        self.head = torch.nn.Linear(2, 1, dtype=torch.float64)
        A = torch.tensor([[1.2, .35], [-.25, .8]], dtype=torch.float64)
        b = torch.tensor([.15, -.2], dtype=torch.float64)
        w = torch.tensor([.7, -.4], dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight.copy_(S @ A)
            self.hidden.bias.copy_(S @ b)
            self.head.weight.copy_(torch.linalg.solve(S.T, w)[None, :])
            self.head.bias.fill_(.1)

    def forward(self, x):
        return self.head(self.hidden(x))


class Prior(torch.nn.Module):
    def __init__(self, q):
        super().__init__()
        self.z = torch.nn.Parameter(q.clone())


def trainer_for(q, S):
    return SimpleNamespace(
        prior=Prior(q), ema_prior=Prior(q), G=torch.nn.Identity(), D=Critic(S),
        recipe=SimpleNamespace(birth_death_space="critic", birth_death_isolation=True),
        device=torch.device("cpu"), dtype=torch.float64, controller=None,
        opt_g=SimpleNamespace(state={}, latent_history=None), completed_steps=0,
    )


def array_summary(t):
    return {"min": float(t.min()), "median": float(t.median()), "max": float(t.max()),
            "mean": float(t.double().mean())}


def sidecar(bd, q, F, R, fast):
    k = bd.k
    knn = bd_module._knn
    rR, iR = knn(q, R, k, shortlist_dtype=fast)
    rF, iF = knn(q, F, k, shortlist_dtype=fast)
    own = torch.arange(N)
    dR, _ = bd_module._levina_bickel(knn(R, R, k, exclude=own, shortlist_dtype=fast)[0])
    dF, _ = bd_module._levina_bickel(knn(F, F, k, exclude=own, shortlist_dtype=fast)[0])
    x = min(max(dR, 1.), q.shape[1]) * (rR[:, -1] / rF[:, -1]).log()
    R1, R2 = R[0::2], R[1::2]
    rho = knn(R1, R1, k, exclude=torch.arange(len(R1)), shortlist_dtype=fast)[0][:, -1]
    positive = rho[rho > 0]
    floor = float(positive.median()) * 1e-3 if len(positive) else 1.

    def score(u):
        d, ix = knn(u, R1, k, shortlist_dtype=fast)
        return d[:, -1] / rho[ix].sort(1).values[:, (k - 1) // 2].clamp_min(floor)

    null = score(R2).sort().values
    scores = score(q)
    p = (1. + (len(null) - torch.searchsorted(null, scores)).double()) / (1. + len(null))
    ps = p.sort().values
    passed = (ps <= torch.arange(1, N + 1, dtype=p.dtype) * bd.Q / N).nonzero()
    flags = p <= ps[int(passed[-1])] if len(passed) else torch.zeros(N, dtype=torch.bool)
    assert torch.equal(flags, bd._isolated(q, R, k, fast))
    return {"rR": rR[:, -1], "rF": rF[:, -1], "iR": iR, "iF": iF,
            "dR": dR, "dF": dF, "x": x, "scores": scores, "p": p,
            "flags": flags}


def run_one(q, R, S, whiten=False):
    trainer = trainer_for(q, S)
    bd = ParticleBirthDeath(trainer, seed=771234)
    bd.observe_real(R)
    assert bd.ready()
    covariance_condition = None
    if whiten:
        extract = bd._features
        reference_features = extract(trainer, R[0::2])
        centered = reference_features - reference_features.mean(0, keepdim=True)
        C = centered.T @ centered / (len(centered) - 1)
        covariance_condition = float(torch.linalg.cond(C))
        assert math.isfinite(covariance_condition) and covariance_condition < 1e8
        L = torch.linalg.cholesky(C)
        W = torch.linalg.inv(L.T)
        bd._features = lambda t, x: extract(t, x) @ W
    replay = torch.Generator(device="cpu")
    replay.set_state(bd.stream.get_state())
    Fraw = q[torch.randint(N, (N,), generator=replay)]
    raw = torch.cat((q, Fraw, R))
    with torch.no_grad():
        logits = trainer.D(raw).flatten()
        features = bd._features(trainer, raw)
        fq, fF, fR = features.split(N)
        centre = fR.mean(0, keepdim=True)
        fq, fF, fR = fq - centre, fF - centre, fR - centre
        fast = torch.float32
        stats = sidecar(bd, fq, fF, fR, fast)
        pairs = []
        original_move = bd._move

        def captured_move(t, children, parents):
            pairs.extend(zip(children.tolist(), parents.tolist()))
            return original_move(t, children, parents)

        bd._move = captured_move
        last = bd.maybe_apply(trainer, 0.0)
    assert last is not None
    assert int(stats["flags"].sum()) == last.get("iso_flagged", -1)
    return {"logits": logits, "features": features, "stats": stats,
            "flag_ids": stats["flags"].nonzero().flatten().tolist(), "pairs": pairs,
            "final_z": trainer.prior.z.detach().clone(), "last": last, "k": bd.k,
            "covariance_condition": covariance_condition}


def compare(a, b, S):
    sa, sb = a["stats"], b["stats"]
    summary = {
        "logit_max_abs": float((a["logits"] - b["logits"]).abs().max()),
        "feature_transform_max_abs": float((a["features"] @ S.T - b["features"]).abs().max()),
        "flag_symmetric_difference": len(set(a["flag_ids"]) ^ set(b["flag_ids"])),
        "moves_equal": a["pairs"] == b["pairs"],
        "final_z_max_abs": float((a["final_z"] - b["final_z"]).abs().max()),
        "flag_count": len(b["flag_ids"]), "move_count": len(b["pairs"]),
        "dR_delta": sb["dR"] - sa["dR"], "dF_delta": sb["dF"] - sa["dF"],
        "rR_max_abs_delta": float((sa["rR"] - sb["rR"]).abs().max()),
        "rF_max_abs_delta": float((sa["rF"] - sb["rF"]).abs().max()),
        "x_max_abs_delta": float((sa["x"] - sb["x"]).abs().max()),
        "score_max_abs_delta": float((sa["scores"] - sb["scores"]).abs().max()),
        "p_max_abs_delta": float((sa["p"] - sb["p"]).abs().max()),
        "rR_knn_id_changed_rows": int((sa["iR"] != sb["iR"]).any(1).sum()),
        "rF_knn_id_changed_rows": int((sa["iF"] != sb["iF"]).any(1).sum()),
        "last_dR": b["last"]["d_R"], "last_dF": b["last"]["d_F"],
        "last_iso_acted": b["last"].get("iso_moves", 0) > 0,
    }
    assert summary["logit_max_abs"] < 1e-12
    assert summary["feature_transform_max_abs"] < 1e-12
    return summary


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260929)
    q0 = torch.from_numpy(rng.standard_normal((N, 2)))
    R = torch.from_numpy(rng.standard_normal((N, 2)))
    I = torch.eye(2, dtype=torch.float64)
    variants = {"base": I, "axis_x16": torch.diag(torch.tensor([16., 1.], dtype=torch.float64)),
                "axis_y16": torch.diag(torch.tensor([1., 16.], dtype=torch.float64)), "uniform16": 16 * I,
                "rotation90": torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)}
    result = {"source": str(SOURCE), "source_sha256": None, "N": N, "cases": {}}
    import hashlib
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for mode, knn in (("normal", ORIGINAL_KNN), ("exhaustive_float64", exhaustive_knn)):
        bd_module._knn = knn
        result["cases"][mode] = {}
        for case in ("null", "shifted"):
            q = q0.clone()
            if case == "shifted":
                q[:256, 0] += 2.5
            print(f"{mode} {case}: base", flush=True)
            base = run_one(q, R, I)
            entry = {"base": {"k": base["k"], "flag_count": len(base["flag_ids"]),
                               "flag_ids": base["flag_ids"], "move_count": len(base["pairs"]),
                               "pairs": base["pairs"], "dR": base["stats"]["dR"],
                               "dF": base["stats"]["dF"],
                               "x": array_summary(base["stats"]["x"]),
                               "isolation_score": array_summary(base["stats"]["scores"]),
                               "isolation_p": array_summary(base["stats"]["p"])},
                     "comparisons": {}}
            for name, S in variants.items():
                if name == "base":
                    continue
                print(f"{mode} {case}: {name}", flush=True)
                other = run_one(q, R, S)
                entry["comparisons"][name] = compare(base, other, S)
                entry["comparisons"][name]["flag_ids"] = other["flag_ids"]
                entry["comparisons"][name]["pairs"] = other["pairs"]
            result["cases"][mode][case] = entry
            Path(__file__).with_name("results.json").write_text(json.dumps(result, indent=2) + "\n")
    bd_module._knn = ORIGINAL_KNN
    print("wrote results.json", flush=True)


if __name__ == "__main__":
    main()
