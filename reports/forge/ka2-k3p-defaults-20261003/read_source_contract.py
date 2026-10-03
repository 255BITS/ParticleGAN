"""Read frozen metadata/Recipe declarations only; never instantiate a fixture."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/ml2/hypergan/ParticleGAN-ka2-k3p-defaults-20261003')
OUT = Path(__file__).resolve().parent
COMMIT = '26ff278c3796d775969391adc0bde52e3af11149'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == COMMIT
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
from benchmarks.toy_audit import api_contract as contract, api_family_search as search


def import_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = import_file('source_contract_protocol', 'reports/forge/ka2-k3p-defaults-20261003/protocol.py')
cases = contract.discover()

EXPLANATIONS = {
    'image-develop-img_intensity2-source-transpose12': {
        'question': 'Can the original convolutional GAN recover both equal-mass patch intensities with accurate pixels and sustained finite-template fidelity?',
        'target_description': 'Two 8x8 grayscale images: central rows/columns 2:6 at .35 or .85; all other pixels zero; target probabilities .5/.5. Training pixels add independent sigma=.01 noise and are clipped to [0,1]; evaluation uses the exact clean ordered bank.',
        'why_distinct': 'Photometric recovery: matching patch geometry or nearest-mode coverage cannot hide brightness errors, rejected images, or unequal template frequencies.',
        'gif_should_show': 'Fixed ordered .35/.85 goals and real served images at actual update boundaries; keep the default FAIL/PASS and HQ/finite-template TV visible. A two-patch-looking endpoint alone is insufficient.',
        'diagnostic_fields': ['Saved image arrays: central-patch intensity and background leakage', 'Exact distinct clean outputs and draw multiplicities; nearest/quality counts per target', 'Recorded hq, rejected_mass, distribution_tv and finite_template_tv', 'Final prior row geometry, G/D and EMA parameter differences; optimizer row histories and critic call/anchor records'],
        'limits': ['Mean RMSE is diagnostic; .06 is a per-image quality cutoff, not an aggregate RMSE gate.', 'Deterministic 32-row population counting requires all32 distinct outputs to be present in the retained draw; row-to-output IDs and intermediate weights are not saved.']},
    'api-vector-two-broad': {
        'question': 'Can the same shared family configuration learn two broad Gaussian components with correct mass and within-mode spread, rather than collapse onto two centers?',
        'target_description': 'Two stationary means (-1,0)/(1,0), mass .5/.5, isotropic sigma=.25 (covariance .0625I).',
        'why_distinct': 'Broad two-mode smoke condition tests mass and continuous width before the expensive narrow100 modes. Center-only coverage is explicitly rejected by full covariance/eigenvalue and analytic CDF gates.',
        'gif_should_show': 'Global held-out target/generated scatter, nearest-mode mass bars, and the fixed mode0 zoom at every media boundary; preserve the analytic projection-KS bound even if both centers look occupied.',
        'diagnostic_fields': ['Recorded mass_tv, hq, component_covariance_error, component_min_eigen_ratio, sw1_normalized and projection_ks', 'Retained scatter arrays permit per-mode means/covariance, spill/outlier and fixed-projection CDF attribution without new draws', 'Final learned prior/G weights and public optimizer state; data/latent/noise streams'],
        'limits': ['The public receipt keeps scalar metrics only; per-component arrays from the underlying vector scorer are not retained as scalar fields.', 'A sampled CDF failure does not identify a critic or optimizer mechanism.']},
    'api-grid100': {
        'question': 'Can the original linear-generator/public-critic host learn all100 equally weighted narrow Gaussian modes, their local widths and independent density-fidelity bounds?',
        'target_description': '10x10 axis-aligned centers from coordinates -4.5 through4.5; equal .01 weights; isotropic sigma=.03.',
        'why_distinct': 'High mode count and narrow precision jointly test population coverage, mass, local density and long-horizon stability. Exact center occupancy is not enough.',
        'gif_should_show': 'Global100-mode cloud and mass display plus fixed mode0 sigma-scale zoom; noisy-primary and output-noise-off diagnostic panels must be explicitly distinguished.',
        'diagnostic_fields': ['Primary native precision/mass/covariance/radial metrics and independent accuracy_ metrics', 'Saved primary noisy cloud and clean_ diagnostic cloud; output_sigma and clean_gate_passed', 'Long-horizon scheduled G/D/prior rates and optimizer counters', 'Actual fast serving flag; no feature-cell or DV12 controller exists under these named families'],
        'limits': ['Native primary is sampled with the current scheduled fixed output noise; clean diagnostics never replace its gates.', 'Zero-clock sigma0 capacity is not a certificate for later .029 additive-noise reachability.']},
    'api-rotated100': {
        'question': 'Do the original100-mode coverage, width and accuracy requirements still hold after a fixed25-degree rotation?',
        'target_description': 'The same 10x10 equal100 Gaussian mixture rotated by25 degrees; isotropic sigma=.03 remains unchanged.',
        'why_distinct': 'A useful orientation control: it preserves mode count/width/masses and varies alignment with the generator/critic features. This is related to grid100, not an independent natural-data generalization law.',
        'gif_should_show': 'Show the actual rotated global target and generated cloud with the identical local-width/mass panels and unchanged numerical annotations.',
        'diagnostic_fields': ['Same native/accuracy and noisy-vs-clean fields as grid100', 'Per-mode residual direction/covariance from retained arrays can expose axis-sensitive errors', 'Final parameter/rate/optimizer records'],
        'limits': ['The rotation is stationary, not a moving-target adaptation test.', 'Failure/success differences need matched evidence before being attributed to feature alignment.']},
    'api-staggered100': {
        'question': 'Can the same100-mode host preserve precision, mass and local density on a row-offset lattice with compressed horizontal spacing?',
        'target_description': 'Original grid centers: coordinate0 scaled by .85; alternating rows shift coordinate1 by -.25/+.25; equal .01 masses and isotropic sigma=.03.',
        'why_distinct': 'Tests structured but nonrectangular geometry and spacing. It is neither another seed of grid100 nor a moving law.',
        'gif_should_show': 'Actual staggered target/served cloud globally, plus fixed mode0 local width and mass panels; show all100 masses rather than relying on dot visibility.',
        'diagnostic_fields': ['Same original native/accuracy and clean-diagnostic fields', 'Per-row/per-mode deficits, residual direction and spill attribution using the saved clouds', 'Actual scheduled rates and population optimizer state'],
        'limits': ['Unequal spacing does not change target component sigma or uniform mass.', 'No separate feature-cell mechanism is active in this noncontinuous cohort.']},
    'api-vector-unequal-mass': {
        'question': 'Can the shared configuration reproduce strongly unequal mode probabilities while preserving the shape of components resolved by the256-row population?',
        'target_description': 'Four corners at (+/-1.5,+/-1.5), source order (-,-),(-,+),(+,-),(+,+), masses .55/.30/.13/.02; all sigma=.18.',
        'why_distinct': 'Rare-mode and dominant-mode balancing differs from equal mixtures. The .02 component is explicitly under the32-row expected-population shape floor, but its mass and distributional CDF remain tested.',
        'gif_should_show': 'Global scatter and unequal target-mass bars, with the fixed first-mode width zoom. Rare-mode visibility is not full rare covariance qualification.',
        'diagnostic_fields': ['min_mass_ratio, mass_tv, hq and analytic projection_ks', 'resolved_core_covariance_error, resolved_core_min_eigen_ratio and resolved_max_component_spill', 'Retained arrays allow component counts and covariance/spill attribution with the frozen resolution mask', 'Prior row ownership/counters and final G/D state'],
        'limits': ['Expected row counts are140.8/76.8/33.28/5.12; the last component is excluded from resolved shape aggregates, not dropped from mass/HQ/CDF tests.', 'Original historical full-shape failures remain unchanged; this named current gate has a different explicit resolution contract.']},
    'api-vector-anisotropic': {
        'question': 'Can the shared configuration recover three differently oriented Gaussian ellipses, including their narrow eigen-directions and resolved local spill?',
        'target_description': 'Equal three modes at (-2,-1),(0,1.5),(2,-1), with source covariances [[.09,.018],[.018,.0081]], [[.0081,-.018],[-.018,.09]], [[.04,.03],[.03,.04]].',
        'why_distinct': 'Correct global variance or round blobs can hide local anisotropy; signed covariance and the minimum whitened eigenvalue test distinct shape recovery.',
        'gif_should_show': 'Three target ellipses and generated cloud globally; target mass bars and fixed mode0 zoom reveal narrow-axis shrinkage and spill.',
        'diagnostic_fields': ['Resolved core covariance error/minimum eigenvalue/spill plus mass_tv,hq,sw1_normalized,projection_ks', 'Retained samples allow per-component rotated eigensystems and remote-outlier attribution', 'Final generator/prior parameter shape and optimizer row counters'],
        'limits': ['All three components resolve at about85.33 expected rows; broad-axis error cannot be traded against a missing narrow axis.', 'The gate does not identify which optimizer owner caused a local-shape failure.']},
    'image-develop-img_bars4-source-transpose12': {
        'question': 'Can the same original convolutional host recover all four equal-mass bar locations/orientations with accurate pixels and sustained frequency fidelity?',
        'target_description': 'Four8x8 binary templates: vertical bars in columns1:3 and5:7, followed by horizontal bars in rows1:3 and5:7; .25 mass each. Training pixels add sigma=.01 noise and clip to[0,1].',
        'why_distinct': 'Spatial/orientation recovery in a convolutional host; it distinguishes correct locations from mean intensity alone and tests four-way mass after earlier prerequisites.',
        'gif_should_show': 'Ordered four reference bars beside actual served images at real updates, retaining mode count, HQ and both mass-fidelity annotations.',
        'diagnostic_fields': ['Saved image arrays: nearest-template location/orientation counts and rejected-image attribution', 'Recorded modes,hq,distribution_tv,finite_template_tv,rejected_mass', 'G/prior parameter diversity and critic penalty/optimizer state'],
        'limits': ['.10 is the per-image RMSE cutoff; both aggregate TVs separately require<=.10.', 'An unreached case remains UNKNOWN even if its zero-update capacity witness is SUPPORTED.']},
}

records = []
for name, tier in search.DEFAULT_CASES:
    case = cases[name]
    definitions = EXPLANATIONS[name]
    records.append(dict(case_id=name, stage=tier, case_sha256=search.digest(case),
                        case=case, **definitions,
                        original_metric_steps=contract.evaluation_steps(case['default_steps'], 25),
                        original_media_steps=contract.evaluation_steps(case['default_steps'], 9),
                        families={f: dict(resolved_recipe=protocol.resolved_recipe(case, f),
                                          family_law=protocol.family_law(case, f))
                                  for f in protocol.FAMILIES}))

paths = list((ROOT/'particlegan').glob('*.py'))
paths += [ROOT/p for p in (
    'benchmarks/toy_audit/api_contract.py', 'benchmarks/toy_audit/api_family_search.py',
    'benchmarks/toy_audit/api_vectors.py', 'benchmarks/toy_audit/api_images.py',
    'benchmarks/toy_audit/api_run.py', 'benchmarks/toy_audit/definition_quality.py',
    'benchmarks/toy_audit/vector_quality.py', 'benchmarks/transfer_suite/vector_tasks.py',
    'benchmarks/transfer_suite/image_tasks.py', 'benchmarks/toy100/problems.py',
    'benchmarks/toy100/metrics.py', 'benchmarks/toy100/accuracy.py', 'lib/toy_models.py',
    'reports/forge/ka2-k3p-defaults-20261003/protocol.py',
    'reports/forge/ka2-k3p-defaults-20261003/bind_capacity.py',
    'reports/forge/ka2-k3p-defaults-20261003/run_family_defaults.py')]
hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(set(paths))}
packet = dict(schema='ka2_k3p_source_question_checklist_v1', source_commit=COMMIT,
              source_root=str(ROOT), source_files_sha256=hashes,
              definition_count=8, family_case_denominator=16, shared_overrides=protocol.OVERRIDES,
              training_seed=24002, evaluation_seed=34002,
              executed_by_this_reader=dict(models=0, optimizer_updates=0, draws=0, scoring_calls=0,
                                           cuda_initialized=torch.cuda.is_initialized()),
              retrospective_source_checklist=True,
              original_gate='Exact full default updates/draw count; all final5 post-update metric checks pass.',
              study_gate='First5 consecutive passing metric checks; every later check must pass and at least5 later checks must exist.',
              stage_stop='First non-PASS original/study result stops that whole shared configuration; unreached questions stay UNKNOWN.',
              common_law_notes=[
                  'This is a named KA2/K3P family-owned cohort, not an Atlas/E22 replay or old ordinary-board qualification.',
                  'Both use fast serving, continuous_policy=None, no DV12 perturbation, no learned noise parameter, AMSGradFalse; standardize=True metadata does not standardize ParticlePrior reads.',
                  'AMSGradFalse is not a disabling of KA2/K3P mechanisms: public critic formulations/spike guard and conditional latent-row damping remain owned by the recipe.',
                  'Training fixed output sigma warms from0 to.029 over.2*full horizon; image/vector primary evaluation removes additive output noise, native primary retains it.',
                  'Generator/critic cosines start at.6*min(full horizon,1600) and end at network floor.01. Prior cosine uses full horizon and floor.05. Input noise.5 decays over first.1*horizon.',
                  'KA2 begins blended penalty at call800 (first799 are pureA); K3P handover follows the recorded critic LR ratio. Both image protocols end at600 penalty calls, so KA2 does not reach its blended phase.',
                  'EMA G/prior are checkpointed, but serve_average0 means they do not supply the primary sample law.',
                  'api_run discards each step return: loss/gradient/controller event traces and intermediate model weights are not retained. Only retained observations and final state may support later attribution.',
                  'Family/host comparison and source changes do not import a positive grade. Zero-update capacity is necessary representability evidence, not convergence or a family winner.'
              ], records=records)
assert len(records) == 8 and not torch.cuda.is_initialized()
(OUT/'source-questions-and-observables.json').write_text(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False)+'\n')

md = [f'# Eight source-bound questions and retained observables\n\nSource `{COMMIT}`; metadata/Recipe reads only. This checklist is retrospective source documentation, not a newly preregistered scientific run. No fixture/model, restore, sample, scorer or CUDA call occurs in this reader.\n',
      'Each family uses one shared `lr=.006375 / prior_lr_mult=1 / d_lr_mult=1` configuration over all eight definitions. All16 outcomes stay in the denominator. The original full-budget/final-five gate and additional first-five-plus-five-later hold gate are separate; unreached cases remain UNKNOWN.\n',
      'Primary serving is fast-only with no DV12 or latent standardization. Fixed output noise is a training regularizer: images and ordinary vectors evaluate without additive output noise; native100 primary evaluation includes its scheduled sigma. Native clean panels are diagnostics. The generic provider caption mentioning retained latent perturbation is not evidence that this named family has a controller. `amsgrad=False` selects the ordinary Adam denominator while preserving the public KA2/K3P penalty, guard and conditional row-damping owners.\n']
for row in records:
    c=row['case']
    md += [f"## {row['case_id']}\n", row['question']+'\n',
           row['target_description']+'\n', 'Reason to retain: '+row['why_distinct']+'\n',
           f"Original resources: {c['default_steps']} updates, batch{c['batch_size']}, evaluation n={c['eval_samples']}; 24 post-update numeric checks, final5 required. Exact frozen thresholds:\n\n```json\n"+json.dumps(c['thresholds'], indent=2, sort_keys=True)+'\n```\n',
           'GIF purpose: '+row['gif_should_show']+'\n',
           'Retained diagnostic fields: '+'; '.join(row['diagnostic_fields'])+'.\n',
           'Limits: '+' '.join(row['limits'])+'\n']
md += ['## Evidence limits\n',
       'The public runner retains all metric/media-union sample arrays, scalar metrics and a final complete owner checkpoint. It discards step-return losses and does not save intermediate parameters, per-row evaluated output IDs, gradients, or penalty-phase events. Final optimizer counters can establish completed calls or an anchor having started, but cannot reconstruct every earlier transition or explain why a particular row crossed a quality bound. Final G/prior EMA weights can be compared as parameters; their unobserved output law cannot be inferred without a separately declared model evaluation.\n',
       'Exact case metadata, full per-family resolved Recipes, public sampling laws, source file SHA256s and observation/media boundaries are in `source-questions-and-observables.json`. No scientific grades are assigned by this checklist.\n']
(OUT/'SOURCE_QUESTIONS_AND_OBSERVABLES.md').write_text('\n'.join(md))
print(json.dumps({'source':COMMIT,'questions':len(records),'family_cells':16,'cuda_initialized':torch.cuda.is_initialized()}))
