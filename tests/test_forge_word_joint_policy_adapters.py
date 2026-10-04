"""Tiny CPU structure/negative controls; never word learning qualification."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.forge import word_joint_policy_contracts as declaration
from particlegan import UpdatePolicy, get_recipe
from particlegan.birth_death import ParticleBirthDeath
from experiments.forge.word_joint_policy_adapters import (WordJointPolicyFixture,
    WordGenerator, WordEncoder, WordJointCritic, WORD_DIM, WORD_CHARS, WORD_LENGTH,
    joint_generation, join_words, words_only_noise, word_global_rng)
from experiments.forge.policy_adapters import evaluation_state, typed_state_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = torch.get_num_threads(); torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(before)


def definition():
    task = declaration.make_variant(ROOT)
    request = dict(candidate=dict(id="synthetic_word_joint", recipe_preset="atlas", task_cohort=declaration.COHORT,
                                  recipe_overrides=deepcopy(declaration.SHARED_OVERRIDES)), protocol=dict(seed=0))
    return request, task


def fixture():
    return WordJointPolicyFixture(*definition())


def test_original_scaffolds_free_encoder_roles_and_actual_independent_owners():
    value = fixture()
    assert type(value.G) is WordGenerator and type(value.E) is WordEncoder and type(value.D) is WordJointCritic
    assert [list(p.shape) for p in value.G.parameters()] == [[64,2],[64],[128,64],[128],[168,128],[168]]
    assert [list(p.shape) for p in value.E.parameters()] == [[128,168],[128],[64,128],[64],[2,64],[2]]
    assert [list(p.shape) for p in value.D.parameters()] == [[256,170],[256],[128,256],[128],[1,128],[1]]
    assert value.prior.z.shape == (11,2) and value.words.shape[0] == 5
    assert value.recipe.encoder_mode == "none" and value.recipe.total_steps is None and value.max_steps == 20001
    assert value.recipe.prior_kind == "particles" and not value.recipe.standardize and value.recipe.sigma_rel == 0.
    assert value.policy.roles == [["generator","encoder","table","noise"],["critic"]]
    assert value.policy.row_semantics == "independent" and value.policy.routed_control is None
    assert value.policy.birth_death is not None and value.policy.row_evidence is not None
    assert value.opt_g.latent_damping is not None and value.opt_g.direct_response is None
    assert value.policy.ema_encoder is not None and value.policy.encoder is value.E
    assert value.controls()["family"] == declaration.FAMILY and not value.controls()["independent_atlas_qualification"]
    assert isinstance(value.policy.birth_death, ParticleBirthDeath)
    assert value.policy.birth_death.N == 11 and value.policy.birth_death.k == 5
    assert value.policy.birth_death.isolation is True and value.policy.birth_death.Q == .05
    assert value.policy.birth_death.k < (value.policy.birth_death.N+1)//2
    assert value.recipe.birth_death_isolation and value.recipe.particle_birth_death
    assert value.controls()["actual_prior_rows"] == 11 and value.controls()["canonical_target_words"] == 5
    assert value.receipt()["actual_resources"] == dict(num_particles=11,z_dim=2,batch_size=256)
    # Encoder mapping does not depend on table rows, mass or an assigned word ID.
    before = value.E(value.words).detach().clone()
    with torch.no_grad(): value.prior.z.copy_(value.prior.z.flip(0))
    assert torch.equal(value.E(value.words),before)


@pytest.mark.parametrize("rows", [5, 6, 7, 8, 9, 10])
def test_real_original_small_populations_refuse_full_owner_contract(rows):
    # Instantiate the actual public policy, preserving every requested control.
    # These are resource eligibility negatives, not weakened word variants.
    recipe = get_recipe("atlas", **declaration.SHARED_OVERRIDES, num_particles=rows,
                        z_dim=2,batch_size=256,encoder_mode="none",row_policy="independent")
    assert recipe.particle_birth_death and recipe.birth_death_isolation and recipe.row_evidence_gate
    with torch.random.fork_rng(devices=[]):
        prior = recipe.make_prior(generator=torch.Generator().manual_seed(1700))
        g,e,d = WordGenerator(),WordEncoder(),WordJointCritic()
        opt_g=recipe.make_generator_optimizer([
            dict(params=list(g.parameters()),lr=recipe.lr), dict(params=list(e.parameters()),lr=recipe.lr),
            dict(params=[prior.z],lr=recipe.lr*recipe.prior_lr_mult)],latent_table=prior.z)
        opt_d=recipe.make_critic_optimizer(d,ema_critic=deepcopy(d))
        reason = "at least 6 particles" if rows == 5 else "leave-one-out neighbours"
        with pytest.raises(ValueError,match=reason):
            UpdatePolicy(recipe,g,d,prior=prior,encoder=e,generation=joint_generation,
                row_semantics="independent",generator_optimizer=opt_g,critic_optimizer=opt_d,seed=0)


def test_actual_optimizer_groups_and_two_update_roles_have_requested_rates():
    value=fixture()
    assert [g["lr"] for g in value.opt_g.param_groups] == [.0053125,.0053125,.00796875,.0053125]
    assert [g["lr"] for g in value.opt_d.param_groups] == [.0053125]
    for group,parameters in zip(value.opt_g.param_groups[:3],
            [list(value.G.parameters()),list(value.E.parameters()),[value.prior.z]]):
        assert [id(p) for p in group["params"]] == [id(p) for p in parameters]
    before={name:[p.detach().clone() for p in parameters]
            for name,(_,parameters) in value.role_parameters.items()}
    for _ in range(2):value.step()
    for name,(_,parameters) in value.role_parameters.items():
        assert any(not torch.equal(p,old) for p,old in zip(parameters,before[name]))
    assert value.guards()["optimizer_updates"] == dict(generator=2,encoder=2,prior=2,discriminator=2)
    assert value.controls()["lifecycle"]["observed_updates"] == 2


def test_exact_original_clean_joint_function_loss_and_all_G_E_table_gradients():
    value = fixture(); old_g, old_e = deepcopy(value.G), deepcopy(value.E)
    old_z = torch.nn.Parameter(value.prior.z.detach().clone())
    rows = torch.arange(16)%11; real_words = value.words[torch.arange(16)%5]
    original_latent = old_z[rows]
    original_fake = join_words(old_g(original_latent), original_latent)
    actual_fake = joint_generation(value.G, value.prior.z[rows])
    torch.testing.assert_close(original_fake,actual_fake,rtol=0,atol=0)
    value.D.requires_grad_(False)
    original_loss = value.loss.g_loss(value.D(original_fake), value.D(join_words(real_words,old_e(real_words))))
    original_loss = original_loss + value.recipe.prior_reg*value.spread(old_z)
    actual_loss, actual_gan, aux = value.generator_objective(real_words,actual_fake)
    torch.testing.assert_close(original_loss,actual_loss,rtol=0,atol=0)
    original_parameters = [*old_g.parameters(),*old_e.parameters(),old_z]
    actual_parameters = [*value.G.parameters(),*value.E.parameters(),value.prior.z]
    old_gradient = torch.autograd.grad(original_loss,original_parameters)
    new_gradient = torch.autograd.grad(actual_loss,actual_parameters)
    for old, new in zip(old_gradient,new_gradient): torch.testing.assert_close(old,new,rtol=0,atol=0)
    assert any(torch.count_nonzero(g) for g in new_gradient[len(list(value.G.parameters())):-1])
    assert aux == value.recipe.prior_reg * value.spread(value.prior.z)
    assert not value.task["execution"]["policy_contract"]["reconstruction_training_loss"]


def test_original_clean_critic_joint_loss_and_all_critic_gradients_at_matched_rows():
    value=fixture(); old_d=deepcopy(value.D)
    rows=torch.arange(256)%11; words=value.words[torch.arange(256)%5]
    with torch.no_grad():
        original_latent=value.prior.z[rows]
        original_fake=join_words(value.G(original_latent),original_latent)
        actual_fake=joint_generation(value.G,value.prior.z[rows])
        real=join_words(words,value.E(words))
    original=value.loss.d_loss(old_d(real),old_d(original_fake))
    actual=value.loss.d_loss(value.D(real),value.D(actual_fake))
    torch.testing.assert_close(original,actual,rtol=0,atol=0)
    a=torch.autograd.grad(original,list(old_d.parameters())); b=torch.autograd.grad(actual,list(value.D.parameters()))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_actual_DV12_callback_joins_same_effective_code_and_discloses_old_split_law():
    value = fixture()
    with torch.no_grad(): real = join_words(value.words,value.E(value.words))
    value.policy.begin_step(real,execution_limit=value.max_steps); value.policy.abort_step()
    source = value.prior.z.detach().clone(); rows = torch.arange(11)
    stream_a = torch.Generator().manual_seed(991); stream_b = torch.Generator().manual_seed(991)
    expected_code = value.policy.controller.perturb_latent(source,stream_a,value.prior)
    generated = value.policy.generate(source,sigma=0.,stream=stream_b,rows=rows)
    assert torch.equal(generated[:,WORD_DIM:],expected_code)
    assert torch.equal(generated[:,:WORD_DIM],value.G(expected_code).flatten(1))
    assert not torch.equal(expected_code,source)
    assert value.task["execution"]["policy_contract"]["numerical_equivalence_to_parent"] is False
    assert "raw_code" in value.task["execution"]["policy_contract"]["original_split_code_law"]


def test_words_only_noise_preserves_codes_original_draw_shape_and_sigma_gradient():
    value = fixture(); source = joint_generation(value.G,value.prior.z)
    sigma = torch.tensor(.029,requires_grad=True)
    stream_a = torch.Generator().manual_seed(993); stream_b = torch.Generator().manual_seed(993)
    actual = words_only_noise(source,sigma,stream_a)
    draw = torch.randn(11,len(WORD_CHARS),WORD_LENGTH,generator=stream_b).flatten(1)
    expected = torch.cat((source[:,:WORD_DIM]+sigma*draw,source[:,WORD_DIM:]),1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(actual[:,WORD_DIM:],source[:,WORD_DIM:])
    assert torch.autograd.grad(actual[:,WORD_DIM:].sum(),sigma,allow_unused=True,retain_graph=True)[0] == 0
    torch.testing.assert_close(torch.autograd.grad(actual[:,:WORD_DIM].sum(),sigma)[0],draw.sum(),rtol=0,atol=0)
    before=stream_a.get_state().clone()
    assert words_only_noise(source,0.,stream_a) is source and torch.equal(before,stream_a.get_state())


def test_clean_independent_uniform_row_permutation_and_no_encoder_particle_substitution():
    value = fixture(); permutation = torch.tensor([3,0,4,1,2,10,9,8,7,6,5])
    generated = joint_generation(value.G,value.prior.z)
    permuted = joint_generation(value.G,value.prior.z[permutation])
    assert torch.equal(permuted,generated[permutation])
    encoded=value.E(value.words)
    # Free continuous codes are kept; no argmin, row labels or particle projection.
    assert encoded.shape==(5,2)
    assert not torch.equal(encoded,value.prior.z)
    assert torch.equal(joint_generation(value.G,encoded)[:,WORD_DIM:],encoded)


def test_two_updates_observer_purity_complete_checkpoint_and_next_update_resume():
    a,b=fixture(),fixture()
    for _ in range(2):
        assert a.step()==b.step(); b.observe()
    # Evaluation owns advancing named eval streams, while every training state
    # and stream must remain identical to the unobserved trajectory.
    assert typed_state_digest(evaluation_state(a.state_dict()))==typed_state_digest(evaluation_state(b.state_dict()))
    assert b.guards()["optimizer_updates"]==dict(generator=2,encoder=2,prior=2,discriminator=2)
    assert b.guards()["all_finite"] and b.guards()["hooks_exercised"] and all(p["pure"] for p in b.purity)
    saved=b.state_dict(); pin=typed_state_digest(saved); restored=fixture(); restored.load_state_dict(saved)
    assert typed_state_digest(restored.state_dict())==pin==typed_state_digest(saved)
    assert restored.observe()==b.observe()
    assert all(torch.equal(restored.last_views[k],b.last_views[k]) for k in b.last_views)
    assert restored.step()==b.step()
    assert typed_state_digest(restored.state_dict())==typed_state_digest(b.state_dict())
    assert typed_state_digest(saved)==pin


@pytest.mark.parametrize("field",["gate","budget","source","encoder","recipe","observation","code_noise","parent","five_rows","six_rows","resource_provenance","cohort","controls"])
def test_original_contract_or_new_joint_law_drift_rejected(field):
    request,task=definition()
    if field=="gate":task["evaluation"]["thresholds"][4][2]=0
    elif field=="budget":task["execution"]["steps"]=2
    elif field=="source":task["execution"]["policy_contract"]["sources"][declaration.HOST_SOURCE]="0"*64
    elif field=="encoder":task["execution"]["policy_recipe_overrides"]["encoder_mode"]="ae"
    elif field=="recipe":request["candidate"]["recipe_overrides"]["prior_lr_mult"]=1.
    elif field=="observation":task["evaluation"]["policy_observation"]["weight_selector"]="fast"
    elif field=="code_noise":task["execution"]["policy_contract"]["training_output_noise"]="all170_coordinates"
    elif field=="parent":task["policy_parent"]["task_sha256"]="0"*64
    elif field=="five_rows":task["execution"]["resources"]["num_particles"]=5
    elif field=="six_rows":task["execution"]["resources"]["num_particles"]=6
    elif field=="resource_provenance":task["execution"]["resource_adaptation"]["historical_capacity_or_qualification_credit"]=True
    elif field=="cohort":task["task_cohort"]="word_joint_policy_v1"
    else:task["execution"]["policy_contract"]["resource_adaptation"]["controls_disabled"]=True
    with pytest.raises(ValueError):WordJointPolicyFixture(request,task)


@pytest.mark.parametrize("field",["family","clock","nan","words","streams","mechanism","mode","rows"])
def test_complete_checkpoint_tamper_rejects_without_current_owner_mutation(field):
    value=fixture(); value.step(); saved=value.state_dict(); restored=fixture(); before=typed_state_digest(restored.state_dict())
    if field=="family":saved["family"]="atlas"
    elif field=="clock":saved["caller_cursor"]=0
    elif field=="nan":saved["policy"]["models"]["encoder"]["net.1.weight"][0,0]=float("nan")
    elif field=="words":saved["canonical_words"].zero_()
    elif field=="streams":saved["policy"]["streams"]["noise_generator"]=saved["policy"]["streams"]["eval_generator"].clone()
    elif field=="mechanism":saved["mechanism_audit_state"]["a2"]["calls"]=500
    elif field=="mode":saved["module_modes"]["encoder"]=1
    else:saved["policy"]["table"]=saved["policy"]["table"][:5].detach().clone()
    with pytest.raises(ValueError):restored.load_state_dict(saved)
    assert typed_state_digest(restored.state_dict())==before


def test_observer_rejects_a_reduced_population_instead_of_lowering_gate():
    value=fixture()
    with pytest.raises(ValueError,match="full1024"):value.observe(16)


def test_global_rng_observer_model_modes_and_selected_joint_E_checkpoint():
    value=fixture();value.step();value.G.eval();value.E.eval();value.policy.ema_encoder.train()
    before=typed_state_digest(word_global_rng());modes=value.module_modes()
    value.observe()
    assert typed_state_digest(word_global_rng())==before and value.module_modes()==modes
    assert not torch.cuda.is_initialized()
    saved=value.state_dict();copy=fixture();copy.load_state_dict(saved)
    assert copy.module_modes()==modes and typed_state_digest(copy.state_dict())==typed_state_digest(saved)
    assert copy.observe()==value.observe()
    for key in ['generated','reconstruction','prior','generated_effective_code','encoded_code','reconstruction_effective_code']:
        assert torch.equal(copy.last_views[key],value.last_views[key])
    assert copy.last_views['prior'].shape==(11,2) and copy.last_views['target'].shape[0]==5
