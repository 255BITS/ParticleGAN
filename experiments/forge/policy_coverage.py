"""Scoped Atlas/E22 policy-cloud variants of every selected-view test.

The selected Atlas and E22 rows refused all 35 distinct tests before training
because ordinary task declarations certify a clean/live MoG measurement, not a
policy-controlled served sampler. The clean cohort applies the Tier 1 policy
transformation (``tier1_policy._variant``) to all 35 questions:

* an explicit ``particle_cloud`` sigma-0 prior exception with its reason;
* the complete family Recipe, unchanged (schedule-free, external limit = task
  steps, no overrides or tuning) and protocol seed 0;
* the task's architecture, data law, batch sequence, capacity, update budget,
  timeout, evaluation cadence and numerical gates, unchanged;
* state-selected served weights, no output noise, the actual served latent
  policy and isolated evaluation streams, under a cohort-prefixed law name;
* dependents consume their own scoped parent's state and gates.

The as-served cohort is the same transformation with exactly one declared
difference: measured samples keep the formulation's learned output noise
exactly as GANTrainer serves it (learned sigma with its policy floor). Atlas
and E22 supply per-mode width through that noise, so the clean law alone can
fail width gates for that reason only. Recipe, training, gates, budgets, data,
cadence and seed are identical; each law trains in its own run, and its
dependents continue their own same-law producer. Law-independent tests (no
measured output samples, or refused by an ownership blocker before any
measurement) are not duplicated: the clean declaration runs once and its
verdict is shown for both laws.

Both are separate diagnostic cohorts: no qualification numerator, other family
row or archived verdict changes, and the original parents receive no credit.
Tests whose original objective needs conditional, encoder, multi-bank or
word-bound row ownership keep an explicit blocker and spend no training.
"""
from copy import deepcopy
from pathlib import Path

from .tier1_policy import SOURCES as TIER1_SOURCES, PolicyCohort, materialize

COHORT = 'policy_cloud_coverage_v1'
VIEW = 'policy_cloud_coverage'
LABEL = 'Policy-cloud scoped tests, clean (learned output noise removed); seed 0'
SERVED_COHORT = 'policy_cloud_coverage_as_served_v1'
SERVED_VIEW = 'policy_cloud_coverage_as_served'
SERVED_LABEL = 'Policy-cloud scoped tests, as served (learned output noise kept); seed 0'
# The selected recipes these scoped tests serve; only their pages display them.
FAMILIES = ('atlas', 'e22')
REPORTING = {'family_totals': False, 'families': list(FAMILIES)}
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
# Only these registered cohort parents may do so; ordinary tasks keep their
# own continuation rules.
DIAGNOSTIC_CONTINUATIONS = frozenset({'gaussian1d_stability', 'grid100_14k', 'rotated100_14k', 'staggered100_14k'})

COVERAGE = PolicyCohort(
    id=COHORT, parents=PARENTS, sources=SOURCES, blocked_hosts=BLOCKED_HOSTS,
    law_prefix='policy_cloud_', label='Policy-cloud coverage cohort',
    parameter_parents=frozenset({'two_pole', 'unused_token_hold', 'cover_leftover', 'unipolar', 'mid_scale_identity'}),
    clock_parents=frozenset({'clockfree_audit', 'clockfree_audit_measurement_v1'}),
    rename_dependencies=True, attest_draws=True, diagnostic_continuations=DIAGNOSTIC_CONTINUATIONS)
# Law-independent tests: the measurement draws no scored output samples, or an
# ownership blocker refuses the test before any measurement. One clean run (or
# refusal) supplies the same verdict to the clean and as-served rows.
LAW_INDEPENDENT = {
    'two_pole': ('parameter measurement of the selected table and the critic gradient; it draws no output samples'),
    'clockfree_audit_measurement_v1': ('clock audit comparing training-state parity across branches; its verdict '
                                       'never scores output samples'),
    'clockfree_audit': 'clock audit comparing training-state parity across branches; its verdict never scores output samples',
    **{name: 'refused by its ownership blocker before training; no measurement exists under either law'
       for name in BLOCKED_HOSTS},
}
SERVED_PARENTS = tuple(name for name in PARENTS if name not in LAW_INDEPENDENT)
SERVED = PolicyCohort(
    id=SERVED_COHORT, parents=SERVED_PARENTS, sources=SOURCES, blocked_hosts={},
    law_prefix='policy_cloud_as_served_', label='Policy-cloud as-served coverage cohort',
    parameter_parents=frozenset(), clock_parents=frozenset(), rename_dependencies=True,
    measurement='as_served', shared=frozenset(LAW_INDEPENDENT), partner=COHORT,
    attest_draws=True, diagnostic_continuations=DIAGNOSTIC_CONTINUATIONS)
COHORTS = (COVERAGE, SERVED)


def variant_id(parent, cohort=COVERAGE):
    return parent + cohort.suffix


def served_row_task(parent):
    """The as-served row's task: its own variant, or the shared clean one."""
    return variant_id(parent, SERVED if parent in SERVED_PARENTS else COVERAGE)


def reporting_tier(parent):
    return next(tier for tier, names in TIERS.items() if parent in names)


def _assignments(task_for):
    return [{'task': task_for(name), 'qualification_tier': tier, 'importance': 'required', 'order': order}
            for order, (tier, name) in enumerate((tier, name) for tier in (1, 2, 3) for name in TIERS[tier])]


def view_declaration():
    return {'schema_version': 1, 'id': VIEW, 'goal': 'discriminator_stability', 'revision': 1,
            'description': ('Separate Atlas/E22 policy-cloud scoped variants of the 35 distinct tests required by the '
                            'selected views, at their selected-view tiers. Each variant keeps its parent architecture, '
                            'data, budget, cadence and gates, binds an explicit equal-mass particle cloud and scores the '
                            'state-selected served model without output noise. Original live/MoG parents receive no '
                            'credit; incompatible hosts keep explicit ownership blockers.'),
            'assignments': _assignments(variant_id), 'eligibility': {}, 'reporting': deepcopy(REPORTING)}


def served_view_declaration():
    return {'schema_version': 1, 'id': SERVED_VIEW, 'goal': 'discriminator_stability', 'revision': 1,
            'description': ('As-served law of the Atlas/E22 policy-cloud scoped tests, at their selected-view tiers. '
                            'Each as-served variant is its clean policy-cloud variant with one declared difference: '
                            'measured samples keep the learned output noise exactly as GANTrainer serves it (learned '
                            'sigma with its policy floor), with state-selected weights, the actual latent policy and '
                            'isolated evaluation streams. Law-independent tests (the two-pole parameter measurement, '
                            'the clock audits and the ownership-blocked hosts) are shared: this view assigns their '
                            'clean declarations, which run once. Original parents receive no credit.'),
            'assignments': _assignments(served_row_task), 'eligibility': {}, 'reporting': deepcopy(REPORTING)}


def write_declarations(root):
    """Materialize both cohorts from the current parent files and source bytes."""
    import json
    from .contracts import atomic_json
    root = Path(root)
    for cohort in COHORTS:
        directory = root / cohort.directory
        existing = set(directory.glob('*.json')) if directory.is_dir() else set()
        written = set()
        for name in cohort.parents:
            parent = json.loads((root / 'configs/forge/tasks' / (name + '.json')).read_text())
            variant = materialize(parent, root, cohort)
            path = directory / (variant['id'] + '.json')
            atomic_json(path, variant)
            written.add(path)
        for stale in existing - written:
            stale.unlink()
    views = view_declaration(), served_view_declaration()
    for view in views:
        atomic_json(root / 'configs/forge/views' / (view['id'] + '.json'), view)
    return views
