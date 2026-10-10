"""Read-only Recipe/definition audit; never construct a host or sample."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

SOURCE = Path('/ml2/hypergan/ParticleGAN-generator-step-20261003')
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import torch
torch.set_num_threads(1)
from particlegan import get_recipe
from particlegan.policy import input_noise_std, output_noise_std
from benchmarks.toy_audit import api_contract as contract, api_family_search as study

KNOBS = {'lr': .006375, 'prior_lr_mult': 1., 'd_lr_mult': 1.}
FAMILIES = ('ka2', 'k3p')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SOURCE,
                            check=True, capture_output=True, text=True).stdout.strip()
    assert commit == '488b792e2fb875894f017cf7043420f2bb66190f'
    assert not torch.cuda.is_initialized() and torch.get_num_threads() == 1
    definitions = contract.discover()
    cases = [definitions[name] for name, _ in study.DEFAULT_CASES]
    records, motivation = [], []
    for family in FAMILIES:
        path = SOURCE / f'reports/forge/configuration-search/{family}-tier1-refresh-v1.json'
        card = json.loads(path.read_text())
        selected = card['selection']['selected_configuration_id']
        trial = next(row for row in card['trials'] if row['configuration_id'] == selected)
        assert trial['settings'] == {'lr': KNOBS['lr'], 'prior_lr_mult': KNOBS['prior_lr_mult']}
        assert trial['resolved_recipe']['d_lr_mult'] == KNOBS['d_lr_mult']
        motivation.append(dict(family=family, card=str(path), card_sha256=sha(path),
                               configuration_id=selected, grade='4/5', qualified=False,
                               source_digest=card['source_digest'], runtime=card['runtime_cohort'],
                               prior=trial['declaration']['prior'], settings=trial['settings'],
                               relevance='Prospective knob motivation only; different tasks, prior, source/runtime. No capacity or training credit.'))
        for case in cases:
            accepted = contract.validate_recipe_overrides(case, family, KNOBS)
            assert accepted == KNOBS
            # Metadata mirror of existing fixture adaptation, without building
            # the fixture, its models, optimizers, controller or RNG streams.
            options = study.host_recipe_options(case)
            options['total_steps'] = case['default_steps']
            options.update(KNOBS)
            recipe = get_recipe(family, **options)
            assert recipe.continuous_policy is None and recipe.serve_average == 0
            assert recipe.total_steps == case['default_steps']
            assert recipe.output_noise_mode == 'fixed'
            primary_noise = case['kind'] == 'native100'
            horizon = recipe.total_steps
            network_horizon = min(horizon, recipe.network_lr_horizon_cap or horizon)
            original = plain(study.host_recipe_options(case))
            vector_construction = None
            if case['kind'] == 'vector':
                masses = case['spec']['masses']
                expected = [recipe.num_particles * mass for mass in masses]
                counts = [math.floor(value) for value in expected]
                order = sorted(range(len(counts)), key=lambda i: (-(expected[i] - counts[i]), i))
                for index in order[:recipe.num_particles - sum(counts)]:
                    counts[index] += 1
                vector_construction = dict(
                    status='ANALYTIC_PROPOSAL_NOT_CONSTRUCTED_NOT_MEASURED',
                    source_template='benchmarks/toy_audit/vector_quality.py:finite_cloud_witness nonrandom population body only; do not call its final private sampler or borrow its verdict.',
                    particles=recipe.num_particles, z_dim=recipe.z_dim, component_rows=counts,
                    finite_population_mass_tv=sum(abs(count / recipe.num_particles - mass)
                                                  for count, mass in zip(counts, masses)) / 2,
                    spread='Componentwise stratified normal quantiles; deterministic bit-reversed second-coordinate strata; arithmetic centering/whitening of the finite design, then full target Cholesky pushforward and mode center. No gradient fitting.',
                    covariance='Preserve the entire declared 2x2 covariance, including signed off-diagonal terms; an isotropic or diagonal substitute changes the anisotropic question.',
                    public_map='Exact first-two-coordinate identity using +/- pairs in every existing LeakyReLU(.2) hidden layer; remaining coordinates may be zero. Preserve original widths/depths and all D architecture.',
                    sampler='Install only prospective fast G/prior parameters in fresh public owners, uniform row indexing, fast serving, output_noise=False, no added jitter or DV12. Evaluate n4096/seed34002 only after root declaration/source freeze.',
                    limits='Finite projected CDF, mass, spread/covariance, spill and sustained gates remain unmeasured. Exact design covariance is not equality to a Gaussian density or proof training reaches the construction.')
            model = ('native nn.Linear(2,2) + Fourier MLP D; 20k learned rows' if primary_noise
                     else 'original raw transpose12 convolutional G/D; 32 learned rows' if case['provider'] == 'api_images'
                     else 'original LeakyReLU(.2) MLP G/D; learned row table')
            records.append(dict(
                family=family, case_id=case['id'], stage=next(stage for name, stage in study.DEFAULT_CASES if name == case['id']),
                api_host_status='SUPPORTED_BY_DEFINITION', numeric_capacity_status='UNRESOLVED',
                execution_status='NOT_EXECUTED', case_sha256=digest(case), goal=case['goal'],
                original_case_sampling=case['sampling'], original_thresholds=case['thresholds'],
                law=case.get('law', case.get('spec')), host=model,
                target_host_gates_changed=False,
                original_default_steps=case['default_steps'], batch_size=case['batch_size'],
                eval_samples=case['eval_samples'], protocol_seed=case.get('protocol_seed', 24002),
                eval_seed=34002, original_host_options=original,
                requested_shared_overrides=KNOBS, fixture_options=plain(options),
                resolved_recipe=plain(asdict(recipe)), resolved_recipe_sha256=digest(asdict(recipe)),
                primary_output_noise=primary_noise,
                named_serving_law='Uniform independent learned-particle index law through fast G; no DV12/feature-cell perturbation; fixed scheduled additive output noise included only on native primary observations.',
                scheduled_noise={
                    'input_sigma_at_0': input_noise_std(recipe, 0),
                    'input_noise_ends_at': recipe.input_noise_anneal_end * horizon,
                    'output_sigma_at_0': output_noise_std(recipe, 0),
                    'output_noise_full_at': recipe.output_noise_warmup * horizon,
                    'output_sigma_at_full_horizon': output_noise_std(recipe, horizon)},
                scheduled_rates={
                    'nominal_G_D_P': [recipe.lr, recipe.lr * recipe.d_lr_mult, recipe.lr * recipe.prior_lr_mult],
                    'network_cosine_start': recipe.lr_anneal_start * network_horizon,
                    'network_cosine_end': network_horizon, 'network_floor': recipe.resolved_network_lr_floor,
                    'prior_cosine_start': recipe.lr_anneal_start * horizon,
                    'prior_cosine_end': horizon, 'prior_floor': recipe.lr_floor},
                vector_construction=vector_construction,
                host_d_mult_overridden=original.get('d_lr_mult', 1.) != KNOBS['d_lr_mult'],
                ordinary_success_imported=False, capacity_artifacts=[], measured_observations=[]))
    assert len(records) == 16 and len({(x['family'], x['case_id']) for x in records}) == 16
    sources = [*sorted((SOURCE / 'particlegan').glob('*.py')),
               SOURCE / 'lib/toy_models.py', SOURCE / 'benchmarks/toy_audit/api_contract.py',
               SOURCE / 'benchmarks/toy_audit/api_vectors.py', SOURCE / 'benchmarks/toy_audit/api_images.py',
               SOURCE / 'benchmarks/toy_audit/api_family_search.py',
               SOURCE / 'benchmarks/toy_audit/definition_quality.py', SOURCE / 'benchmarks/toy_audit/vector_quality.py',
               SOURCE / 'benchmarks/transfer_suite/vector_tasks.py', SOURCE / 'benchmarks/transfer_suite/stress_tasks.py',
               *sorted((SOURCE / 'benchmarks/toy100').glob('*.py')),
               SOURCE / 'experiments/forge/boundaries.py', SOURCE / 'experiments/forge/configuration_search.py',
               SOURCE / 'configs/forge/tasks/ring16_acquisition.json']
    data = dict(
        schema='particlegan_ka2_k3p_serving_cohort_proposal_v1', status='DECLARED_NOT_EXECUTED',
        recommendation='PROPOSED', source_commit=commit,
        protected_scientific_base='4749b2780add539df4bd8d2dd1d3cc9f002f77ad',
        source_files_sha256={str(path.relative_to(SOURCE)): sha(path) for path in sorted(set(sources))},
        shared_overrides_by_family={f: KNOBS for f in FAMILIES}, ordinary_knob_motivation=motivation,
        denominator={'families': 2, 'cases_each': 8, 'cells': 16, 'numeric_capacity_UNRESOLVED': 16,
                     'scientific_NOT_EXECUTED': 16, 'qualified_whole_configs': 0},
        records=records,
        prerequisite_stopping='Each family separately needs eight fresh candidate-bound SUPPORTED capacity cells. Intensity2 then two-broad are the original two smoke prerequisites; first nonpass stops that family, keeping all unknown cells in the denominator.',
        study_gate='First five consecutive passing post-update primary observations, then every later observation passing with at least five later checks. Original terminal-five/full-budget verdict remains separate.',
        binding_boundaries=[
            'Old API study restricts family names to Atlas/E22 and its capacity schema/rate admission does not authorize these candidates.',
            'Old host_recipe_options/resolved_recipe omit the existing KA2/K3P fixture total_steps adaptation. A new test-only resolver must mirror it exactly before recipe verification.',
            'A named family serving cohort must bind no DV12, fast-only selection, scheduled fixed noise and original native noisy primary explicitly; no old Atlas/E22 sampling-law credit.',
            'New committed reproducer and complete source/dependency/config snapshot, exact three-field overrides and actual full resolved recipes, fresh owners/streams/optimizers and zero clocks must be attested.',
            'Fresh full-count samples, primary gates, targets/views, observer/RNG purity and bitwise public replay are required. Retained fast parameter inputs are construction inputs only.',
            'No learned log-sigma or old controller/average/clock may be transplanted into these fixed-noise family owners.'
        ],
        native_construction_proposal={
            'status': 'ANALYTIC_PROPOSAL_NOT_CONSTRUCTED_NOT_MEASURED',
            'host': 'Public learned ParticlePrior 20000x2 and original learned linear 2x2 G; identity G is permitted.',
            'clock0': 'Place 200 rows per target mode as 100 antipodal pairs: u_j=(j+1/2)/100; r_j=.03*sqrt(-2*log(1-u_j)); theta_j=2*pi*j/golden_ratio; row=center +/- r_j*(cos(theta_j),sin(theta_j)). Uniform index sampling retains equal mass; no fitting, sampler change or jitter.',
            'purpose': 'Finite within-mode width and radial quantiles address the old near-center-input mismatch when fixed output_sigma(0)=0 and no DV12 perturbation is present.',
            'limits': 'Finite row quadrature is not an exact continuous Gaussian law; stochastic repeated-index clouds and every per-mode/accuracy gate remain unmeasured. A failed candidate construction is not incapacity.',
            'terminal_population_residual_std': math.sqrt(.03**2 - .029**2),
            'terminal_argument': 'At full warmup, a population isotropic residual N(0,(.03^2-.029^2)I) plus the fixed .029 output noise has target sigma .03. This is an analytic convolution statement, not a finite-row public sampler certificate or evidence training contracts rows to that state.',
            'reachability': 'A legitimate zero-clock snapshot need not pass unchanged at later noise clocks. Do not fabricate completed clock7000 for a terminal witness or infer convergence from the population argument.'},
        other_construction_proposals={
            'images': 'Original retained fitted fast raw-host parameters and balanced row assignments may be explicit construction inputs only. Reinitialize owners and replay no-DV12 output-noise-off public serving, without fitting or source/gate changes; passing is unmeasured.',
            'vectors': 'Original LeakyReLU(.2) MLP can exactly realize coordinate identity using +/- coordinate pairs and phi(x)-phi(-x)=(1+.2)*x. Place deterministic mass-allocated within-component quadrature rows through that public map; no continuous latent kernel is needed to propose a finite sample-law witness. Exact finite CDF/mass/covariance gates must be measured afresh.'},
        accounting={'unchanged_ceiling_seconds': 15360., 'retained_prior_debit_seconds': 113.99425188452005,
                    'new_allowance_or_reservation_created': False, 'science_seconds': 0., 'costs_reset': False},
        operations={'metadata_recipe_constructions': 16, 'model_constructions': 0, 'model_restores': 0,
                    'optimizer_constructions': 0, 'optimizer_updates': 0, 'new_draws': 0, 'capacity_exports': 0,
                    'scorer_calls': 0, 'cuda_contexts': 0},
        inspection_runtime={'python': platform.python_version(), 'torch': str(torch.__version__), 'cpu_threads': 1},
        reproducer={'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)})
    assert not torch.cuda.is_initialized()
    out = OUTPUT / 'ka2-k3p-next-cohort.json'
    out.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'path': str(out), 'sha256': sha(out), 'records': len(records),
                      'source': commit, 'models_updates_samples': [0, 0, 0]}, sort_keys=True))


if __name__ == '__main__':
    main()
