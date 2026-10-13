"""Exact rational loss-unit algebra plus read-only archived receipt checks.

Standard library only: no tensor/model/autograd/sampling/optimizer API imports.
The affine score map is an analytic coordinate comparison, never an execution
binding or substitution for the required deterministic initialized critic.
"""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[4]
SOURCE = "2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12"
PARENT_COMMIT = "67920d9090727bd74adbd331187f0e6f22533e48"
OLD = ROOT / "reports/forge/r1r2-energy-distance-repair/research/least-squares"
PINS = {
    "README.md": "c8e978bdb914fa81ddef646436e0b3a7f167adaaede4ebdd2cf2b4c6d1e97c85",
    "probe.py": "2b4c7135bcdfd303ddc22b929d61d8752fe31774b9060952e6a26f5e79d59cd2",
    "diagnosis.json": "b4a01f64506c765f8684c4e2ff89251a2d9b264f78ddd4181b06d4867d2581e7",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficients(target):
    """Coefficients of half*(offset+slope*s-target)^2 in s."""
    offset, slope = F(1, 2), F(1, 4)
    difference = offset - target
    return [difference * difference / 2, difference * slope, slope * slope / 2]


def main():
    initial_rng = random.getstate()
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    for name, digest in PINS.items():
        assert sha(OLD / name) == digest
    old = json.loads((OLD / "diagnosis.json").read_text())
    assert old["source_digest"] == SOURCE and old["execution_commit"] == PARENT_COMMIT
    assert old["qualification_input"] is False and old["inherited_labels"] == [0., 1., 1.]
    assert old["models_constructed"] == old["optimizer_updates_added"] == old["sampling_draws_added"] == 0
    assert all(fields == ["loss"] for fields in old["host_recipe_deltas"].values())
    parent_publication = ROOT / "reports/forge/r1r2-energy-distance-repair/publication/results.json"
    assert sha(parent_publication) == old["publication_sha256"]
    publication = json.loads(parent_publication.read_text())
    row = next(r for r in publication["certified_results"] if r["role"] == "non_saturating"
               and r["task_id"] == "ring16_acquisition")
    assert row["attempt_id"] == old["attempt_id"] and row["candidate_revision"] == old["candidate_revision"]
    attempt = Path(row["checkpoint"]["artifact_root"]).parent
    source = json.loads((attempt / "request.json").read_text())["request"]["source"]
    assert source["digest"] == SOURCE and source["origin_commit"] == PARENT_COMMIT
    catalogue = json.dumps(source["files"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(catalogue.encode()).hexdigest() == SOURCE
    assert len(source["files"]) == 1216
    for path, digest in source["files"].items():
        assert sha(ROOT / path) == sha(Path(source["snapshot_path"]) / path) == digest
    assert sha(attempt / "provenance/provenance-state.pt") == old["checkpoint_sha256"]
    assert sha(attempt / "observed-samples.pt") == old["panels_sha256"]

    quarter = F(1, 4)
    polynomials = {}
    for role, target, sign in (("D_real", 1, -1), ("D_fake", 0, 1),
                               ("G_fake", 1, -1), ("E_reversed_real", 0, 1)):
        polynomial = coefficients(target)
        ns_linear, ns_quadratic = F(sign, 2), F(1, 8)
        assert polynomial[1:] == [quarter * ns_linear, quarter * ns_quadratic]
        polynomials[role] = {
            "LS_after_affine_map_constant_linear_quadratic": list(map(str, polynomial)),
            "NS_Taylor_linear_quadratic": list(map(str, (ns_linear, ns_quadratic))),
            "gradient_and_Hessian_scale_relative_to_NS_at_zero": str(quarter),
        }
    slope = F(1, 4)
    penalty_scale = slope * slope
    effective_gamma_ratio = penalty_scale / quarter
    assert penalty_scale == F(1, 16) and effective_gamma_ratio == F(1, 4)
    gamma = F(1, 10)
    assert gamma * effective_gamma_ratio == F(1, 40)
    # Ratio of means, not mean of ratios or a new sampled force measurement.
    mean_score = old["fake_scores"]["mean"]
    mapped_mean_weight = float(slope) * (.5 - float(slope) * mean_score)
    mapped_ratio_of_means = mapped_mean_weight / old["ascent_weights"]["non_saturating"]["mean"]
    assert abs(mapped_ratio_of_means - .25) < 1e-5
    # Explicit deterministic scalar fixtures; no critic evaluation or rescoring.
    fixtures = []
    for score in (old["fake_scores"]["min"], 0., old["fake_scores"]["max"]):
        q = .5 + .25 * score
        ns = 1 / (1 + math.exp(score))
        fixtures.append(dict(NS_score=score, affine_LS_score=q,
            same_raw_score_LS_to_NS_ascent_ratio=(1-score)/ns,
            affine_LS_chain_rule_to_NS_ascent_ratio=.25*(1-q)/ns))
    latest_path = ROOT / "reports/forge/r1r2-energy-prior-freeze-repair/publication/results.json"
    latest = json.loads(latest_path.read_text())
    assert latest["complete"] and latest["aggregate_paid_attempts"] == 238
    assert latest["aggregate_total_paid_seconds"] == 7168.248061007995
    headroom = 8000 - latest["aggregate_total_paid_seconds"]
    assert headroom < 900
    result = dict(schema_version=1, qualification_input=False, status="PASS",
        scope="Exact rational affine/Taylor loss algebra plus byte-bound saved receipt review; not a repair prerequisite PASS.",
        conclusion="Defer native LS; reject the saved twofold-force and raw-curvature comparisons as stronger from-initialization density evidence.",
        source_digest=SOURCE, source_origin_commit=PARENT_COMMIT, source_files_checked=1216,
        current_and_snapshot_scientific_bytes_match=True,
        inputs_sha256={**{str((OLD/name).relative_to(ROOT)): digest for name, digest in PINS.items()},
                      "NS_parent_publication": sha(parent_publication), "latest_closed_publication": sha(latest_path),
                      "reproduction_source": sha(__file__), "archived_NS_request": sha(attempt/"request.json")},
        original_saved_NS_fixture_identity=dict(attempt_id=old["attempt_id"], candidate_revision=old["candidate_revision"],
            checkpoint_sha256=old["checkpoint_sha256"], panels_sha256=old["panels_sha256"],
            original_probe_scope=old["scope"], original_proxy=old["shape_proxy"],
            original_mean_of_raw_LS_to_NS_ascent_weight_ratios=old["least_squares_over_non_saturating_weight"]["mean"]),
        algebra=dict(score_map="q = 1/2 + s/4", NS_expansion="softplus(s)=log(2)+s/2+s^2/8-s^4/192+O(s^6)",
            polynomials=polynomials, R1_R2_penalty_scale=str(penalty_scale),
            LS_adversarial_scale=str(quarter), effective_gamma_ratio_after_normalizing_adversarial_units=str(effective_gamma_ratio),
            gamma_point1_maps_locally_to= float(gamma*effective_gamma_ratio),
            raw_LS_common_shift_curvature=2., NS_common_shift_curvature_at_equilibrium=.5,
            affine_LS_common_shift_curvature=.125,
            mean_mapped_chain_rule_ascent_weight=mapped_mean_weight,
            mapped_chain_rule_ratio_of_recorded_mean_weights=mapped_ratio_of_means,
            scalar_coordinate_fixtures=fixtures),
        non_equivalence_limits=["Affine map is not the native initialized critic path.",
            "Taylor statement is local; logistic fourth/higher terms and LS overshoot/tails differ.",
            "Gradient penalty changes the unregularized best-response relation.",
            "Fixed Adam/AMSGrad epsilon and auxiliary objective units prevent exact optimizer equivalence.",
            "Saved input-force proxies omit G/E Jacobians, optimizer history and future critic refitting."],
        future_scope_if_reconsidered=dict(sole_recipe_delta="loss non_saturating to least_squares",
            parent_energy_weight=0., prior_update="learned", inherited_labels=[0.,1.,1.],
            original_ordinary_hosts=6, separate_own_clock=True, original_task_protocol_unchanged=True,
            full_new_arm_allowance_seconds=2520, exact_cached_NS_control_allowance_seconds=2520,
            complete_comparison_allowance_seconds=5040, new_source_requires_fresh_controls=True),
        budget=dict(aggregate_existing_attempts=238, aggregate_paid_seconds=latest["aggregate_total_paid_seconds"],
            unchanged_cap_seconds=8000, headroom_seconds=headroom, original_word_full_reservation_seconds=900,
            full_word_reservation_fits=False, admission="BLOCKED by existing full-reservation paid ceiling; no cap increase or task reduction"),
        models_constructed=0, autograd_calls=0, optimizer_updates_added=0, sampling_draws_added=0,
        live_evaluations_or_rescoring=0, tensor_runtime_imported=False, python_rng_unchanged=random.getstate()==initial_rng)
    assert result["python_rng_unchanged"]
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
