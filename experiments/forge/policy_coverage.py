"""Scoped Atlas/E22 policy-cloud variants of every selected-view test.

The selected Atlas and E22 rows refused all 35 distinct tests before training
because ordinary task declarations certify a clean/live MoG measurement, not a
policy-controlled served sampler. This cohort applies the Tier 1 policy
transformation (``tier1_policy._variant``) to all 35 questions:

* an explicit ``particle_cloud`` sigma-0 prior exception with its reason;
* the complete family Recipe, unchanged (schedule-free, external limit = task
  steps, no overrides or tuning) and protocol seed 0;
* the task's architecture, data law, batch sequence, capacity, update budget,
  timeout, evaluation cadence and numerical gates, unchanged;
* state-selected served weights, no output noise, the actual served latent
  policy and isolated evaluation streams, under a cohort-prefixed law name;
* dependents consume their own scoped parent's state and gates.

It is a separate diagnostic cohort: no qualification numerator, other family
row or archived verdict changes, and the original parents receive no credit.
Tests whose original objective needs conditional, encoder, multi-bank or
word-bound row ownership keep an explicit blocker and spend no training.
"""
from pathlib import Path

from .tier1_policy import SOURCES as TIER1_SOURCES, PolicyCohort, materialize

COHORT = 'policy_cloud_coverage_v1'
VIEW = 'policy_cloud_coverage'
LABEL = 'Policy-cloud scoped selected-view tests; seed 0'
# Tiers follow the selected Atlas/E22 views (each test has one tier there).
TIERS = {
    1: ('two_pole', 'gaussian1d_smoke', 'ring16_acquisition', 'clockfree_audit_measurement_v1',
        'unused_token_hold', 'ae_gan_hold', 'five_word_joint_smoke'),
    2: ('gaussian1d_stability', 'grid100', 'rotated100', 'staggered100',
        'img_stripes2', 'img_bars4', 'img_blobs4', 'img_intensity2', 'mode_hold',
        'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width',
        'vector_anisotropic', 'vector_overlap', 'vector_spiral',
        'five_word_joint_hold', 'trajectory', 'residual_student', 'cover_leftover',
        'unipolar', 'mid_scale_identity'),
    3: ('clockfree_audit', 'grid100_14k', 'rotated100_14k', 'staggered100_14k',
        'ring_hold', 'ring_extension', 'target_shift_recovery'),
}
PARENTS = tuple(name for tier in (1, 2, 3) for name in TIERS[tier])
SOURCES = TIER1_SOURCES + (
    'experiments/forge/policy_coverage.py', 'experiments/forge/policy_gaussian.py',
    'experiments/forge/gaussian_tasks.py', 'experiments/forge/adaptation.py',
    'experiments/forge/continuation.py', 'experiments/forge/grouped_evidence.py',
    'experiments/forge/state.py', 'experiments/forge/priors.py')
# Plain-language ownership boundaries. These hosts are not independent tables
# of interchangeable sample rows, so the Atlas/E22 row controls (sampling,
# copying and rebalancing rows) cannot run their original objective.
BLOCKED_HOSTS = {
    'unused_token_hold': (
        'the student is one shared embedding plus per-token slot corrections, and the test checks that an '
        'unused token stays unchanged; there is no table of interchangeable sample rows for Atlas/E22 to '
        'sample, copy or rebalance, so a faithful binding needs a conditional (routed) row contract that '
        'these independent-row recipes do not have'),
    'ae_gan_hold': (
        'an autoencoder GAN whose encoder maps data into the Gaussian-mixture latent law that the test '
        'scores; Atlas/E22 refuse encoder recipes while their row controls are on, and replacing the mixture '
        'with an equal-weight point table would change the law under test, so it needs a routed encoder task'),
    'five_word_joint_smoke': (
        'a joint generator/encoder/critic (BiGAN) in which exactly five latent rows each stay bound to one '
        'word and must be recovered by the encoder; Atlas/E22 birth/death would copy rows between words, and '
        'the encoder, joint critic and five-row minimum population are outside the independent-row recipes'),
    'five_word_joint_hold': (
        'continues the five-word joint BiGAN from its own smoke checkpoint and inherits the same boundary '
        '(an encoder plus rows bound to word identities); there is no compatible scoped smoke state to continue'),
    'trajectory': (
        'each latent row is permanently paired with one source trajectory and the critic judges '
        '(source, continuation) pairs, so rows are not interchangeable samples; copying or rebalancing rows '
        'would break the pairing, which needs a conditional routing contract'),
    'residual_student': (
        'a conditional residual head whose output depends on (source arc, row i), with row i tied to context i '
        'and a source-conditioned critic; rows are bound to contexts rather than independent equal-weight '
        'samples, which needs a conditional routing and protected-context contract'),
    'cover_leftover': (
        'two separately learned point banks plus scale-conditioned residuals, scored on polar contexts with '
        'declared host jitter; one independent equal-weight table cannot represent that joint law'),
    'unipolar': (
        'the trainable rows are three role vectors (odd, even, origin) mixed with fixed scale weights; every '
        'output uses all of them, the critic is told the scale and the fake batch repeats one vector, so '
        'copying one role vector onto another would rewrite the residual law'),
    'mid_scale_identity': (
        'like unipolar plus a fourth mid-scale vector that switches on only near scale 0.5; it is a '
        'scale-conditioned blend of role vectors with no sampled rows, so it cannot be bound without '
        'routed (conditional) semantics'),
}
# Checkpoint dependents continue their own scoped producer even after its
# quality gate fails (user-authorized diagnostic continuation; BISPF precedent).
DIAGNOSTIC_CONTINUATIONS = frozenset({'gaussian1d_stability', 'grid100_14k', 'rotated100_14k', 'staggered100_14k'})

COVERAGE = PolicyCohort(
    id=COHORT, parents=PARENTS, sources=SOURCES, blocked_hosts=BLOCKED_HOSTS,
    law_prefix='policy_cloud_', label='Policy-cloud coverage cohort',
    parameter_parents=frozenset({'two_pole', 'unused_token_hold', 'cover_leftover', 'unipolar', 'mid_scale_identity'}),
    clock_parents=frozenset({'clockfree_audit', 'clockfree_audit_measurement_v1'}),
    rename_dependencies=True)


def variant_id(parent):
    return parent + COVERAGE.suffix


def reporting_tier(parent):
    return next(tier for tier, names in TIERS.items() if parent in names)


def view_declaration():
    assignments = [{'task': variant_id(name), 'qualification_tier': tier, 'importance': 'required', 'order': order}
                   for order, (tier, name) in enumerate((tier, name) for tier in (1, 2, 3) for name in TIERS[tier])]
    return {'schema_version': 1, 'id': VIEW, 'goal': 'discriminator_stability', 'revision': 1,
            'description': ('Separate Atlas/E22 policy-cloud scoped variants of the 35 distinct tests required by the '
                            'selected views, at their selected-view tiers. Each variant keeps its parent architecture, '
                            'data, budget, cadence and gates, binds an explicit equal-mass particle cloud and scores the '
                            'state-selected served model without output noise. Original live/MoG parents receive no '
                            'credit; incompatible hosts keep explicit ownership blockers.'),
            'assignments': assignments, 'eligibility': {}, 'reporting': {'family_totals': False}}


def write_declarations(root):
    """Materialize all variants from the current parent files and source bytes."""
    import json
    from .contracts import atomic_json
    root = Path(root)
    directory = root / COVERAGE.directory
    existing = set(directory.glob('*.json')) if directory.is_dir() else set()
    written = set()
    for name in PARENTS:
        parent = json.loads((root / 'configs/forge/tasks' / (name + '.json')).read_text())
        variant = materialize(parent, root, COVERAGE)
        path = directory / (variant['id'] + '.json')
        atomic_json(path, variant)
        written.add(path)
    for stale in existing - written:
        stale.unlink()
    view = view_declaration()
    atomic_json(root / 'configs/forge/views' / (VIEW + '.json'), view)
    return view
