"""A useful code must help every declared judge; movement alone cannot pass."""
from copy import deepcopy
import marshal
from pathlib import Path

import pytest

from examples import e22_routed_convergence_guided_campaign as guided
from examples import reclassify_e22_routed_convergence_signed as reclassification
from examples import run_e22_routed_convergence_rotated_teacher as runner
from examples import review_e22_routed_convergence_rotated_teacher as reviewer


@pytest.fixture(scope="module")
def evaluators():
    guided_runner, guided_reviewer = guided.adapters()
    return {"rotated_runner": runner.final_gates, "rotated_reviewer": reviewer.independent_gates,
            "guided_runner": guided_runner.final_gates, "guided_reviewer": guided_reviewer.independent_gates}


def inputs():
    scores = {arm + "@6400": {judge: {"test": {"paired_game": value}} for judge in runner.JUDGES}
              for arm, value in zip(runner.law.ARMS, (1., 1.5, .9))}
    witnesses = {arm: {"bridge_still_trainable": True, "bank_still_trainable": True,
                      "router_still_trainable": True, "C_norms": dict.fromkeys(runner.law.SITES, .1),
                      "live_bank_updates": 4, "live_query_updates": 4,
                      "zero_code_minus_live_test_game": dict.fromkeys(runner.JUDGES, .1)}
                 for arm in runner.law.ARMS[1:]}
    return scores, witnesses


@pytest.mark.parametrize("name", ("rotated_runner", "rotated_reviewer", "guided_runner", "guided_reviewer"))
@pytest.mark.parametrize("arm", runner.law.ARMS[1:])
@pytest.mark.parametrize("judge", runner.JUDGES)
def test_one_harmful_judge_rejects_each_arm_despite_other_support(evaluators, name, arm, judge):
    scores, witnesses = inputs()
    positive = evaluators[name](scores, witnesses)
    assert positive["H_b_support_gate"] and positive["neutral_beats_ordinary_all_four"]
    witnesses[arm]["zero_code_minus_live_test_game"][judge] = -.1
    before = deepcopy((scores, witnesses))
    result = evaluators[name](scores, witnesses)
    assert not result["retained_particle_gate"][arm]
    other = next(item for item in runner.law.ARMS[1:] if item != arm)
    assert result["retained_particle_gate"][other]
    assert result["original_particle_gap_reproduced"]
    assert all(value > 1e-4 for value in result["paired_game_H_b_improvement"].values())
    if arm == runner.law.ARMS[2]:
        assert not result["H_b_support_gate"] and not result["neutral_beats_ordinary_all_four"]
    assert (scores, witnesses) == before


@pytest.mark.parametrize("value", (0., 1e-6, -1e-6))
def test_reclassification_requires_strict_benefit(value):
    scores, witnesses = inputs()
    witnesses[runner.law.ARMS[2]]["zero_code_minus_live_test_game"][runner.JUDGES[0]] = value
    result = reclassification.classify_scores(scores, witnesses)
    assert not result["retained_particle_gate"][runner.law.ARMS[2]]
    assert not result["H_b_support_gate"] and not result["neutral_beats_ordinary_all_four"]


@pytest.mark.parametrize("module", (runner, reviewer))
def test_source_proof_rejects_additional_math_or_training_change(module):
    repaired = Path(module.__file__).read_text()
    signed = 'all(value > 1e-6 for value in witness["zero_code_minus_live_test_game"].values())'
    assert repaired.count(signed) == 1
    original = repaired.replace(signed, signed.replace("all(value", "all(abs(value)"))
    assert reclassification.signed_source_proof(original, repaired)["other_source_ast_unchanged"]
    with pytest.raises(ValueError, match="exactly one"):
        reclassification.signed_source_proof(original, original)
    with pytest.raises(ValueError, match="exactly one"):
        reclassification.signed_source_proof(original, repaired.replace("1e-4", "2e-4"))


def test_reclassification_rejects_modified_bound_review(tmp_path):
    path = tmp_path / "review.json"
    path.write_text('{"qualified":true}\n')
    expected = reclassification.sha(path)
    path.write_text('{"qualified":true,"invented_score":1}\n')
    with pytest.raises(ValueError, match="bound artifact changed"):
        reclassification.read_bound(path, expected)


def test_benign_marshal_reference_flags_do_not_change_actual_factory_identity(monkeypatch):
    namespace = {}
    exec(compile("def factory(arm, data, *, bindings=None):\n    return (arm, data, bindings, 3.25)\n",
                 "metadata-control.py", "exec"), namespace)
    factory = namespace["factory"]
    monkeypatch.setattr(guided.factory.baseline, "make_loop", factory)
    code = factory.__code__
    raw_before = marshal.dumps(code)
    before = guided.factory.factory_binding_manifest()
    aliases = (code.co_consts, code.co_names, code.co_varnames, code.co_filename,
               code.co_code, code.co_linetable, getattr(code, "co_exceptiontable", b""))
    raw_after = marshal.dumps(code)
    assert raw_before != raw_after  # The old metadata guard falsely rejects.
    assert marshal.loads(raw_before) == marshal.loads(raw_after)
    assert guided.factory.factory_binding_manifest() == before
    assert factory("arm", "data") == ("arm", "data", None, 3.25)
    assert aliases[0] == code.co_consts


@pytest.mark.parametrize("drift", ("bytecode", "defaults"))
def test_stable_factory_identity_still_rejects_real_code_and_default_drift(monkeypatch, drift):
    bound, _ = guided.adapters()
    factory = guided.factory.baseline.make_loop
    if drift == "bytecode":
        monkeypatch.setattr(factory, "__code__", factory.__code__.replace(
            co_consts=(*factory.__code__.co_consts, "new code constant")))
    else:
        monkeypatch.setattr(factory, "__kwdefaults__", {"bindings": {"changed_default": True}})
    with pytest.raises(ValueError, match="changed after namespace creation"):
        bound.law.make_rotated_loop(guided.factory.ARMS[1], {})


def test_metadata_proof_cannot_admit_guidance_or_host_math_change():
    repaired = Path(guided.factory.__file__).read_text()
    original = "import marshal\n" + repaired.replace("code_sha(baseline.make_loop)",
        "hashlib.sha256(marshal.dumps(baseline.make_loop.__code__)).hexdigest()")
    assert reclassification.metadata_source_proof(original, repaired)["other_source_ast_unchanged"]
    with pytest.raises(ValueError, match="more than factory identity"):
        reclassification.metadata_source_proof(original, repaired.replace("GUIDANCE = 3.", "GUIDANCE = 4."))
