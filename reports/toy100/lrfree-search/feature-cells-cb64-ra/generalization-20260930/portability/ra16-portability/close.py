"""Seal reviewed RA16 source and completed CPU qualification."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
BASE = STUDY / 'pkg-RA15-partial-recovery'
PACKAGE = STUDY / 'pkg-RA16-portability'
CONFIG = STUDY / 'configs/RA16-portability.json'
PREP = STUDY / 'integration-prep/ra16-portability'
TEST = PREP / 'tests/test_feature_portability.py'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_sha(package):
    h = hashlib.sha256()
    for path in sorted((package / 'particlegan').rglob('*.py')):
        h.update(str(path.relative_to(package / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


assert not (ROOT / 'SOURCE-BRIDGE.json').exists()
assert not (ROOT / 'SOURCE-FREEZE.json').exists()
assert package_sha(BASE) == '741fc3933654b985a74314b730b93be814d408ad46f6f7f6547b561f64619228'
assert CONFIG.read_bytes() == (STUDY / 'configs/RA15-partial-recovery.json').read_bytes()
assert sha(CONFIG) == 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
original = json.loads((STUDY / 'portability/partial-recovery/SOURCE-FREEZE.json').read_text())
for path, expected in original['hashes'].items():assert sha(path) == expected, path
old = {p.name: sha(p) for p in (BASE / 'particlegan').glob('*.py')}
new = {p.name: sha(p) for p in (PACKAGE / 'particlegan').glob('*.py')}
assert old.keys() == new.keys()
changed = sorted(name for name in old if old[name] != new[name])
assert changed == ['feature_policy.py', 'feature_reference.py', 'mean_transport.py',
                   'output_moments.py', 'population_continuity.py']
patch = []
for name in changed:
    a, b = BASE / 'particlegan' / name, PACKAGE / 'particlegan' / name
    patch.extend(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True),
                 fromfile=str(a), tofile=str(b)))
(ROOT / 'candidate.patch').write_text(''.join(patch))
proof = json.loads((ROOT / 'default-cpu-bridge.json').read_text())
assert proof['status'] == 'CPU_PASS' and proof['CUDA_initialized'] is False
assert proof['source_AST_only_7_CPU_factory_keywords_plus_pre_mutation_shape_validator']
assert len(proof['CPU_allocation_sites']) == 7
for name, marker in [('portable-cpu-attempt-1.log', '19 passed'),
                     ('all-cpu-contracts-attempt-1.log', '94 passed')]:
    log = (ROOT / name).read_text()
    assert marker in log and 'cuda_initialized False' in log and str(PACKAGE / 'particlegan/__init__.py') in log
review = json.loads((ROOT / 'PEER-REVIEW.json').read_text())
assert review['status'] == 'APPROVE'
report = '''# RA16 portability closure

RA16 is cloned from immutable RA15. Five modules differ: seven CPU planning
factory calls now explicitly allocate on CPU, and backend restore validates
the exact FIFO sample shape before creating/installing controls or mutating
models, optimizers, RNG streams, parameter versions, or live controls.
An unresolved output shape cannot carry an initialized FIFO. Resolved output
shapes may still carry an untouched FIFO, and legacy KNN shape metadata stays
compatible. Checkpoint schemas, config bytes, arithmetic, ordering, budgets,
R1 trigger, serving guard, and quality thresholds are unchanged.

## Executed CPU evidence

- 94 contracts pass: 58 existing source contracts, 17 portable partial
  recovery contracts, and 19 new portability contracts. CUDA is visible and
  remains uninitialized.
- `meta` default contexts exercise the actual streamed projection, complete
  and partial frozen moments/odd queries, absent-moment proposals, an accepted
  paired packet including real chart queries/epoch guards/hash, feature
  control installation, and empty/nonempty population scalar queries.
- Trainer and policy restore reject product-preserving shape mismatches
  atomically in feature and KNN fallback routes. Unresolved initialized-FIFO
  cases are rejected, while valid pending/resolved untouched-FIFO states load.
- The original RA15 mixed-device projection failure and accepted malformed
  shape followed by generated-observation failure are preserved in the CPU
  bridge receipt.
- Default CPU RA15/RA16 loss and complete checkpoint state match at every
  update for 18 feature updates through two reactions (2048 mean-forward
  rows) and two KNN fallback updates. Samples, bidirectional valid restore,
  and the next update match exactly. Only the diagnostic elapsed clock is
  shared. AST normalization proves the only source changes are the seven CPU
  keyword additions and the pre-mutation shape validator.

## Prepared GPU contract

`test_cuda_default_preserves_feature_reactions_and_checkpoint_replay` runs the
same seed1234 and explicit GPU models/table/data under CPU and CUDA defaults,
through 18 updates, valid checkpoint restore, exact next update, and samples.
It skips only if CUDA is unavailable. The root agent's serial GPU queue owns
execution; it is not part of this completed CPU count. The CPU fixture permits
an already initialized CUDA runtime and rejects any new lazy initialization.

Old RA15 packages, lanes, failure logs, and qualification remain unchanged.
This source bridge preserves the ordinary CPU-default helper allocation law;
it does not claim that the deferred actual CUDA-default test has passed.
'''
(ROOT / 'REPORT.md').write_text(report)
bridge = dict(status='CPU_PASS_GPU_DEFAULT_PENDING', candidate='RA16-portability',
    base_package=str(BASE), base_package_sha256=package_sha(BASE),
    package=str(PACKAGE), package_sha256=package_sha(PACKAGE),
    config=str(CONFIG), config_sha256=sha(CONFIG), config_byte_identical=True,
    changed_modules=changed, unchanged_modules=len(new)-len(changed),
    source_sha256=new, base_source_sha256=old,
    changes=['seven explicit CPU allocations in CPU float64 planning/calibration',
             'exact FIFO/output shape and unresolved initialized-FIFO validation before mutation'],
    CPU_allocation_sites=proof['CPU_allocation_sites'],
    default_CPU_source_AST_forward_law=True, default_CPU_trace_forward_law=proof['default_CPU_forward_law'],
    default_CPU_bridge_path=str(ROOT / 'default-cpu-bridge.json'),
    default_CPU_bridge_sha256=sha(ROOT / 'default-cpu-bridge.json'),
    checkpoint_schema_unchanged=True, valid_checkpoint_math_unchanged=True,
    checkpoint_validation_more_strict_for_invalid_shapes=True,
    recipe_fields_unchanged=True, scorer_changed=False, schedule_changed=False,
    thresholds_changed=False, serving_guard_changed=False, training_RNG_law_changed=False,
    model_optimizer_detector_or_R1_guard_changed=False,
    CPU_contracts_passed=94, new_portable_CPU_contracts_passed=19,
    CUDA_visible_during_CPU_contracts=True, CUDA_initialized=False,
    portable_test_path=str(TEST), portable_test_sha256=sha(TEST),
    portable_fixture_path=str(PREP / 'tests/fixtures/feature-auto-base.json'),
    portable_fixture_sha256=sha(PREP / 'tests/fixtures/feature-auto-base.json'),
    prepared_GPU_test_node=str(TEST) + '::test_cuda_default_preserves_feature_reactions_and_checkpoint_replay',
    actual_GPU_test_executed=False, peer_review_sha256=sha(ROOT / 'PEER-REVIEW.json'),
    original_RA15_source_freeze_valid=True, candidate_patch_sha256=sha(ROOT / 'candidate.patch'))
(ROOT / 'SOURCE-BRIDGE.json').write_text(json.dumps(bridge, indent=2, sort_keys=True) + '\n')
files = list((PACKAGE / 'particlegan').glob('*.py')) + list((BASE / 'particlegan').glob('*.py'))
files += [CONFIG, STUDY / 'configs/RA15-partial-recovery.json']
files += list((PREP / 'tests').rglob('*.py')) + list((PREP / 'tests').rglob('*.json'))
files += [STUDY / 'integration-prep/ra15-partial-recovery/tests/test_feature_partial_recovery.py',
          STUDY / 'portability/test_ra12_contracts.py',
          STUDY / 'portability/settled-guard/test_settled_guard.py',
          STUDY / 'portability/cpu-fix/test_cpu_optimizer_scope.py',
          STUDY / 'portability/replay-alias/test_state_transfer_alias.py']
files += [p for p in ROOT.iterdir() if p.is_file() and p.name not in ('SOURCE-FREEZE.json', 'SHA256SUMS', 'close.log')]
frozen = dict(status='FROZEN_CPU_PASS_GPU_DEFAULT_PENDING', candidate='RA16-portability',
    package_sha256=package_sha(PACKAGE), config_sha256=sha(CONFIG),
    hashes={str(p): sha(p) for p in sorted(set(files))})
(ROOT / 'SOURCE-FREEZE.json').write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
for path, expected in frozen['hashes'].items():assert sha(path) == expected, path
(ROOT / 'SHA256SUMS').write_text('\n'.join(f'{sha(p)}  {p.name}' for p in sorted(ROOT.iterdir())
    if p.is_file() and p.name not in ('SHA256SUMS', 'close.log')) + '\n')
print(json.dumps(dict(status=bridge['status'], package_sha256=package_sha(PACKAGE),
    config_sha256=sha(CONFIG), source_bridge_sha256=sha(ROOT / 'SOURCE-BRIDGE.json'),
    source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'), guarded_files=len(frozen['hashes']),
    portable_test_sha256=sha(TEST), CPU_contracts_passed=94, GPU_test_executed=False), sort_keys=True), flush=True)
