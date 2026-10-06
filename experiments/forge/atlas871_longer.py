"""One fixed from-init 2000/48 diagnostic; original Gaussian task and gates are protected."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math

CANDIDATE_ID = 'atlas-existing-mog-longer871-v1'
VIEW_ID = 'atlas_existing_mog_longer871_v1'
STUDY_ID = 'atlas-existing-mog-longer871-study-v1'
TRACK_ID = 'atlas871_existing_mog_longer'
TASK_ID = 'gaussian1d_acquisition_longer871_v1'
COHORT = 'atlas_existing_mog_longer871_v1'
TASK_PATH = 'configs/forge/task-variants/atlas871_longer/gaussian1d_acquisition_longer871_v1.json'
TASK_PIN = {'sha256': '3b438187b658afadf98d49d42e2433613530d9d8587ab6a0b49db04741427242', 'bytes': 4175}
TASK_DIGEST = '09025c182ec7670f3346238ae2e95c2b92c5312cae8ae19a5f8bb4d52701e256'
PARENT_PATH = 'configs/forge/tasks/gaussian1d_acquisition.json'
PARENT_PIN = {'sha256': 'b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13', 'bytes': 3668}
PARENT_DIGEST = '2ee559cdf9589c0051d463b20d03b00c28801442b9c903bcebd63bf38200ec0c'
PROTOCOL_PATH = 'configs/forge/protocols/screening.json'
PROTOCOL_PIN = {'sha256':'3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803','bytes':972}
EXPECTED_TASK = {'schema_version': 1, 'id': 'gaussian1d_acquisition_longer871_v1', 'adapter': 'transfer_vector', 'execution': {'initializer': 'deterministic_orthogonal', 'host': 'gaussian1d_acquisition', 'steps': 2000, 'prior': {'kind': 'mog', 'sigma': 0.025, 'standardize': False, 'learnable': True}, 'protocol': 'screening', 'produces_state': False, 'host_source': 'benchmarks/toy_audit/gaussian1d_quality.py', 'host_definition': {'hidden': 32, 'layers': 2, 'fourier': 2, 'z_dim': 2, 'particles': 256, 'batch': 128, 'steps': 2000, 'lr': 0.00425, 'd_lr_mult': 1.0, 'prior_lr_mult': 2.0, 'prior_reg': 0.0, 'betas': [0.0, 0.999], 'ema_decay': 0.995, 'reg_arm': 'k3p', 'reg_coeff': 1.0, 'reg_kappa': 1.0, 'd_every': 1, 'g_every': 1, 'family': 'gaussian1d_acquisition', 'split': 'development', 'kind': 'gaussian_mixture', 'means': [[2.0]], 'covariances': [[[0.25]]], 'masses': [1.0], 'identifiable': True}}, 'evaluation': {'sampling_contract_version': 1, 'eval_output_noise': 'clean', 'kind': 'transfer_sustained', 'evaluator': 'experiments.forge.atlas871_longer:test_verdict', 'gate_policy': {'id': 'gaussian1d-acquisition-v1', 'finite_atom_exemptions': False, 'rationale': 'Provisional scalar acquisition gate fixed before training. Exact analytic CDF, location and width reject point collapse, shift, wrong spread and discrete same-moment impostors; independent oracle controls do not calibrate the tier.'}, 'sampling_law': 'public_prior_without_output_noise', 'thresholds': [['sample_count', '>=', 4096], ['finite_fraction', '==', 1.0], ['mean_error_sigma', '<=', 0.2], ['std_ratio', '>=', 0.8], ['std_ratio', '<=', 1.2], ['cdf_ks', '<=', 0.05]], 'observations': 48, 'minimum_stable_checks': 5, 'scoring_weights': 'live', 'sources': {'benchmarks/transfer_suite/protocol.py': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89', 'benchmarks/locked_shared/observation.py': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b', 'benchmarks/transfer_suite/vector_tasks.py': '3ee4eb27759f61a430029c80b7772cac3dca2fb2ac84919ace0db94efca1e0b0', 'benchmarks/toy_audit/gaussian1d_quality.py': '7ec72e07c1aea87e77c85401b7e822f23d5a248ba45d718180045a1ad64ccd8b'}, 'sample_evaluator': 'benchmarks.toy_audit.gaussian1d_quality:score_samples'}, 'resources': {'gpus': 1, 'gpu_memory_mb': 2048, 'cpu_threads': 1, 'timeout_seconds': 240}, 'requires_capabilities': ['checkpoint', 'named_rng', 'live_sampling', 'learned_locations', 'mog_prior'], 'dependencies': [], 'description': 'One from-initialization 2000-update Gaussian duration diagnostic, with the original 1000/24 prefix and an appended 24-read lattice. Original network, prior, data, seed, live sampling and every numerical gate are fixed. No early stop or original-task qualification credit.', 'research_artifacts': {'api_publication': 'reports/toy_audit/api_contract/gaussian1d/results.json', 'readout': 'reports/toy_audit/api_contract/gaussian1d/README.md'}, 'retained_question_ids': ['develop-gaussian1d_acquisition'], 'task_cohort': 'atlas_existing_mog_longer871_v1', 'duration_diagnostic_parent': {'task': 'gaussian1d_acquisition', 'raw_sha256': 'b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13', 'raw_bytes': 3668, 'original_steps': 1000, 'original_observations': 24, 'clock_rule': 'ceil(k*1000/24),k=1..48', 'from_initialization': True, 'qualification_reuse': False}}
EXPECTED_PARENT = {'schema_version': 1, 'id': 'gaussian1d_acquisition', 'adapter': 'transfer_vector', 'execution': {'initializer': 'deterministic_orthogonal', 'host': 'gaussian1d_acquisition', 'steps': 1000, 'prior': {'kind': 'mog', 'sigma': 0.025, 'standardize': False, 'learnable': True}, 'protocol': 'screening', 'produces_state': False, 'host_source': 'benchmarks/toy_audit/gaussian1d_quality.py', 'host_definition': {'hidden': 32, 'layers': 2, 'fourier': 2, 'z_dim': 2, 'particles': 256, 'batch': 128, 'steps': 1000, 'lr': 0.00425, 'd_lr_mult': 1.0, 'prior_lr_mult': 2.0, 'prior_reg': 0.0, 'betas': [0.0, 0.999], 'ema_decay': 0.995, 'reg_arm': 'k3p', 'reg_coeff': 1.0, 'reg_kappa': 1.0, 'd_every': 1, 'g_every': 1, 'family': 'gaussian1d_acquisition', 'split': 'development', 'kind': 'gaussian_mixture', 'means': [[2.0]], 'covariances': [[[0.25]]], 'masses': [1.0], 'identifiable': True}}, 'evaluation': {'sampling_contract_version': 1, 'eval_output_noise': 'clean', 'kind': 'transfer_sustained', 'evaluator': 'benchmarks.transfer_suite.protocol:test_verdict', 'gate_policy': {'id': 'gaussian1d-acquisition-v1', 'finite_atom_exemptions': False, 'rationale': 'Provisional scalar acquisition gate fixed before training. Exact analytic CDF, location and width reject point collapse, shift, wrong spread and discrete same-moment impostors; independent oracle controls do not calibrate the tier.'}, 'sampling_law': 'public_prior_without_output_noise', 'thresholds': [['sample_count', '>=', 4096], ['finite_fraction', '==', 1.0], ['mean_error_sigma', '<=', 0.2], ['std_ratio', '>=', 0.8], ['std_ratio', '<=', 1.2], ['cdf_ks', '<=', 0.05]], 'observations': 24, 'minimum_stable_checks': 5, 'scoring_weights': 'live', 'sources': {'benchmarks/transfer_suite/protocol.py': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89', 'benchmarks/locked_shared/observation.py': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b', 'benchmarks/transfer_suite/vector_tasks.py': '3ee4eb27759f61a430029c80b7772cac3dca2fb2ac84919ace0db94efca1e0b0', 'benchmarks/toy_audit/gaussian1d_quality.py': '7ec72e07c1aea87e77c85401b7e822f23d5a248ba45d718180045a1ad64ccd8b'}, 'sample_evaluator': 'benchmarks.toy_audit.gaussian1d_quality:score_samples'}, 'resources': {'gpus': 1, 'gpu_memory_mb': 2048, 'cpu_threads': 1, 'timeout_seconds': 120}, 'requires_capabilities': ['checkpoint', 'named_rng', 'live_sampling', 'learned_locations', 'mog_prior'], 'dependencies': [], 'description': 'Can the public ParticleGAN trainer acquire the scalar law N(2, 0.5^2) from random initialization within 1,000 updates, with correct location, width and CDF shape at five terminal checks?', 'research_artifacts': {'api_publication': 'reports/toy_audit/api_contract/gaussian1d/results.json', 'readout': 'reports/toy_audit/api_contract/gaussian1d/README.md'}, 'retained_question_ids': ['develop-gaussian1d_acquisition']}
SCORER_PINS = {'benchmarks/transfer_suite/protocol.py': {'sha256': '99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89', 'bytes': 9964}, 'benchmarks/locked_shared/observation.py': {'sha256': 'bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b', 'bytes': 4351}, 'benchmarks/transfer_suite/vector_tasks.py': {'sha256': '3ee4eb27759f61a430029c80b7772cac3dca2fb2ac84919ace0db94efca1e0b0', 'bytes': 22296}, 'benchmarks/toy_audit/gaussian1d_quality.py': {'sha256': '7ec72e07c1aea87e77c85401b7e822f23d5a248ba45d718180045a1ad64ccd8b', 'bytes': 2175}}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def declaration(task):
    if not isinstance(task, dict):
        raise ValueError('diagnostic task must be an object')
    result = deepcopy(task)
    if 'preflight_blockers' in result and not isinstance(result.pop('preflight_blockers'), list):
        raise ValueError('malformed preflight cache')
    if 'field_ownership' in result and not isinstance(result.pop('field_ownership'), dict):
        raise ValueError('malformed ownership annotation')
    return result

def is_task(task):
    return isinstance(task, dict) and task.get('id') == TASK_ID

def checkpoints():
    # Integer ceil, extending the original lattice; no 2000/24 retiming.
    return [(k * 1000 + 23) // 24 for k in range(1, 49)]

def _pin(root, relative, expected):
    base = Path(root).resolve()
    path = base / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError('missing or aliased diagnostic Source: ' + relative)
    raw = path.read_bytes()
    if {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)} != expected:
        raise ValueError('diagnostic Source drift: ' + relative)
    return raw

def validate(task, root=None):
    actual = declaration(task)
    if actual != EXPECTED_TASK or digest(actual) != TASK_DIGEST:
        raise ValueError('one exact 2000/48 Gaussian duration declaration required')
    # Every original scientific field is retained except these explicit duration declarations.
    restored = deepcopy(actual)
    restored['id'] = EXPECTED_PARENT['id']
    restored['execution']['steps'] = 1000
    restored['execution']['host_definition']['steps'] = 1000
    restored['evaluation']['observations'] = 24
    restored['evaluation']['evaluator'] = EXPECTED_PARENT['evaluation']['evaluator']
    restored['resources']['timeout_seconds'] = 120
    restored['description'] = EXPECTED_PARENT['description']
    restored.pop('task_cohort')
    restored.pop('duration_diagnostic_parent')
    if restored != EXPECTED_PARENT or digest(restored) != PARENT_DIGEST:
        raise ValueError('duration diagnostic changed an original scientific field')
    if root is not None:
        _pin(root, TASK_PATH, TASK_PIN)
        _pin(root, PARENT_PATH, PARENT_PIN)
        _pin(root, PROTOCOL_PATH, PROTOCOL_PIN)
        for relative, expected in SCORER_PINS.items():
            _pin(root, relative, expected)
    return dict(task_id=TASK_ID, task_sha256=TASK_PIN['sha256'], task_payload_sha256=TASK_DIGEST,
                prior=deepcopy(actual['execution']['prior']))

def load_variants(root, parents):
    path = Path(root) / TASK_PATH
    if not path.exists():
        return {}
    if 'gaussian1d_acquisition' not in parents or declaration(parents['gaussian1d_acquisition']) != EXPECTED_PARENT:
        raise ValueError('duration diagnostic needs the exact original Gaussian parent')
    task = json.loads(path.read_bytes())
    validate(task)
    if TASK_ID in parents:
        raise ValueError('duplicate duration task identity')
    return {TASK_ID:task}

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.observation import sustained
from benchmarks.transfer_suite.protocol import requirements


def test_verdict(spec, result):
    """Recompute sustained success from complete live curves, never trust a stamp."""
    if result is None:
        return dict(status="MISSING", attempted=False, passed=False, confirmation_fraction=2., shortfall=2.)
    if result.get("error"):
        return dict(status="ERROR", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    observations = result.get("observations", result.get("curve", []))
    if spec != {"steps": 2000, "thresholds": EXPECTED_PARENT["evaluation"]["thresholds"]}:
        raise ValueError("exact duration and unchanged Gaussian gates required")
    steps = checkpoints()
    try:
        convergence = sustained(observations, requirements(spec), expected_steps=steps)
    except (KeyError, TypeError, ValueError):
        return dict(status="INVALID", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    cells = baseline.score_metrics(result.get("live", {}), requirements(spec))
    passed = (convergence["confirmed_step"] is not None
              and all(c["status"] == "PASS" for c in cells))
    status = "PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE"
    deficits = [2. if c["margin"] is None else min(2., max(0., -c["margin"]) / (abs(c["threshold"]) or 1.))
                for c in cells]
    return dict(status=status, attempted=True, passed=passed, metrics=cells,
                convergence=convergence,
                shortfall=sum(deficits) / len(deficits) if convergence["complete"] else 2.,
                confirmation_fraction=convergence["confirmed_step"] / spec["steps"] if passed else 2.)


def grade_transfer(task, evidence):
    from .views import _curve, _finite, _verdict
    try:
        validate(task)
    except (ValueError, TypeError, KeyError) as error:
        return _verdict('INVALID', str(error))
    spec = {'steps':2000, 'thresholds':task['evaluation']['thresholds']}
    points = evidence.get('observations', evidence.get('curve'))
    try:
        _curve(points, spec['thresholds'])
    except ValueError as error:
        return _verdict('INVALID' if points else 'INCOMPLETE', str(error))
    if [p['step'] for p in points] != checkpoints():
        return _verdict('INCOMPLETE', 'all 48 original-grid diagnostic observations are required')
    live = evidence.get('live')
    if not isinstance(live, dict) or any(not _finite(live.get(key)) for key, _, _ in spec['thresholds']):
        return _verdict('INCOMPLETE', 'missing finite final live metrics')
    grade = test_verdict(spec, {'observations':points, 'live':live})
    return _verdict(grade['status'], 'recomputed 48-read duration diagnostic with original terminal five-suffix gates',
                    metrics=live, evaluator_result=grade, original_task_credit_transferred=False,
                    evidence_use='research_diagnostic')
