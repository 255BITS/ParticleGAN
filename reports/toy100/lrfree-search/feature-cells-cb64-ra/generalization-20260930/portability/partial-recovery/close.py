"""Seal the RA15 source bridge and completed CPU contracts before GPU use."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
BASE = STUDY / 'pkg-RA14-replay'
PACKAGE = STUDY / 'pkg-RA15-partial-recovery'
CONFIG = STUDY / 'configs/RA15-partial-recovery.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_sha(package):
    h = hashlib.sha256()
    for path in sorted((package / 'particlegan').rglob('*.py')):
        h.update(str(path.relative_to(package / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


assert not (ROOT / 'SOURCE-BRIDGE.json').exists()
assert not (ROOT / 'SOURCE-FREEZE.json').exists()
assert package_sha(BASE) == '68f5706590683a44348cb04b3798917411bfaa54e5d45b17d57c345a4da33c15'
assert CONFIG.read_bytes() == (STUDY / 'configs/RA14-replay.json').read_bytes()
assert sha(CONFIG) == 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
original_freeze = json.loads((STUDY / 'validation-ra14-r2/SOURCE-FREEZE.json').read_text())
for path, expected in original_freeze['hashes'].items():
    assert sha(path) == expected, path
base_sources = {str(path.relative_to(BASE / 'particlegan')): sha(path)
                for path in (BASE / 'particlegan').rglob('*.py')}
candidate_sources = {str(path.relative_to(PACKAGE / 'particlegan')): sha(path)
                     for path in (PACKAGE / 'particlegan').rglob('*.py')}
assert base_sources.keys() == candidate_sources.keys()
changed = sorted(name for name in base_sources if base_sources[name] != candidate_sources[name])
assert changed == ['feature_cells.py', 'feature_policy.py', 'mean_transport.py', 'output_moments.py']
patch = []
for name in changed:
    old, new = BASE / 'particlegan' / name, PACKAGE / 'particlegan' / name
    patch.extend(difflib.unified_diff(old.read_text().splitlines(True), new.read_text().splitlines(True),
                 fromfile=str(old), tofile=str(new)))
(ROOT / 'candidate.patch').write_text(''.join(patch))
results = {'existing_source_contracts': (ROOT / 'all-58-contracts.log', '58 passed'),
           'focused_recovery_contracts': (ROOT / 'focused-contracts-pass.log', '17 passed')}
for path, marker in results.values():
    content = path.read_text()
    assert marker in content and 'cuda_initialized False' in content
    assert str(PACKAGE / 'particlegan/__init__.py') in content
bridge = dict(status='CPU_PASS', candidate='RA15-partial-recovery',
    base_package=str(BASE), base_package_sha256=package_sha(BASE),
    package=str(PACKAGE), package_sha256=package_sha(PACKAGE),
    config=str(CONFIG), config_sha256=sha(CONFIG), config_byte_identical=True,
    changed_modules=changed, unchanged_modules=len(base_sources)-len(changed),
    source_sha256=candidate_sources, base_source_sha256=base_sources,
    trigger='current feature facade: policy.surprise exists, type(fires) is int, fires > 0',
    scope='preaction even-real and EMA occupied-group moment subset only after an R1 fire',
    active_mask='ephemeral positive frozen weights; no serialized field or schema change',
    original_mass_weights_retained=True, remaining_weights_renormalized=False,
    inactive_weights_and_directions_zero=True, all_odd_observations_retained=True,
    original_alpha_radius_raw_axes_budget_retained=True,
    inactive_proposals_and_previews_filtered_before_division=True,
    initially_active_current_empty_group_veto=True,
    static_no_fire_fixture_exact_checkpoint_and_RNG_parity=True,
    timing_fixture='Shared diagnostic perf_counter for eval_seconds; independent elapsed measurements are not learner state',
    checkpoint_schema_unchanged=True, recipe_fields_unchanged=True,
    scorer_changed=False, schedule_changed=False, thresholds_changed=False,
    serving_guard_changed=False, original_R1_guard_detector_or_optimizer_changed=False,
    tests={key: dict(path=str(path), sha256=sha(path), result=marker, cuda_initialized=False)
           for key, (path, marker) in results.items()},
    total_CPU_contracts_passed=75, CUDA_visible_during_contracts=True, CUDA_initialized=False,
    original_diagnosis_sha256=sha(STUDY / 'diagnostics/moving-rotated-recovery/receipt.json'),
    original_frozen_lane_unchanged=True,
    retained_harness_failure=dict(path=str(ROOT / 'focused-contracts.log'), sha256=sha(ROOT / 'focused-contracts.log'),
        reason='16 passed; independent eval_seconds caused strict metadata comparison to fail; source unchanged during correction'),
    candidate_patch_sha256=sha(ROOT / 'candidate.patch'))
(ROOT / 'SOURCE-BRIDGE.json').write_text(json.dumps(bridge, indent=2, sort_keys=True) + '\n')
files = list((PACKAGE / 'particlegan').rglob('*.py')) + list((BASE / 'particlegan').rglob('*.py'))
files += [CONFIG, STUDY / 'configs/RA14-replay.json']
files += [path for path in ROOT.iterdir() if path.is_file() and path.name not in ('SOURCE-FREEZE.json', 'SHA256SUMS')]
files += [STUDY / 'portability/test_ra12_contracts.py',
          STUDY / 'portability/settled-guard/test_settled_guard.py',
          STUDY / 'portability/cpu-fix/test_cpu_optimizer_scope.py',
          STUDY / 'portability/replay-alias/test_state_transfer_alias.py',
          STUDY / 'diagnostics/moving-rotated-recovery/isolation-arrays.npz',
          STUDY / 'diagnostics/moving-rotated-recovery/receipt.json']
frozen = dict(status='FROZEN_CPU_PASS_NOT_GPU_TRAINED', candidate='RA15-partial-recovery',
    package_sha256=package_sha(PACKAGE), config_sha256=sha(CONFIG),
    hashes={str(path): sha(path) for path in sorted(set(files))})
(ROOT / 'SOURCE-FREEZE.json').write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
for path, expected in frozen['hashes'].items():
    assert sha(path) == expected, path
paths = sorted(path for path in ROOT.iterdir() if path.is_file() and path.name != 'SHA256SUMS')
(ROOT / 'SHA256SUMS').write_text('\n'.join(f'{sha(path)}  {path.name}' for path in paths) + '\n')
print(json.dumps(dict(status='CPU_PASS', package_sha256=package_sha(PACKAGE), config_sha256=sha(CONFIG),
    source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'), source_bridge_sha256=sha(ROOT / 'SOURCE-BRIDGE.json'),
    guarded_files=len(frozen['hashes']), total_contracts_passed=75), sort_keys=True))
