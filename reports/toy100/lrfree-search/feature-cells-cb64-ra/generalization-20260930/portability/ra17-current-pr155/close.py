"""Freeze the reviewed current-PR155 source and scoped historical bridges."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
BASE = STUDY / 'pkg-RA16-portability'
PACKAGE = STUDY / 'pkg-RA17-current-pr155'
CONFIG = STUDY / 'configs/RA17-current-pr155.json'
REPO = Path('/ml2/hypergan/ParticleGAN-ra11-pr155')
NOISE = STUDY / 'diagnostics/upstream-noise-floor-applicability'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):h.update(block)
    return h.hexdigest()


def package_sha(package):
    h = hashlib.sha256()
    for path in sorted((package / 'particlegan').rglob('*.py')):
        h.update(str(path.relative_to(package / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


assert not (ROOT / 'SOURCE-BRIDGE.json').exists()
assert not (ROOT / 'SOURCE-FREEZE.json').exists()
assert package_sha(BASE) == 'ba20f8e6353b2aca4725258835a68c1dfaa056a0484b2d683a5bb74a1eca0aa0'
assert package_sha(PACKAGE) == '500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61'
assert sha(CONFIG) == 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
assert CONFIG.read_bytes() == (STUDY / 'configs/RA16-portability.json').read_bytes()
for freeze_path in (STUDY / 'portability/ra16-portability/SOURCE-FREEZE.json', NOISE / 'SOURCE-FREEZE.json'):
    for path, expected in json.loads(freeze_path.read_text())['hashes'].items():assert sha(path) == expected, path
old = {p.name: sha(p) for p in (BASE / 'particlegan').glob('*.py')}
new = {p.name: sha(p) for p in (PACKAGE / 'particlegan').glob('*.py')}
assert len(old) == len(new) == 28 and old.keys() == new.keys()
assert all(new[name] == sha(REPO / 'particlegan' / name) for name in new)
changed = sorted(name for name in old if old[name] != new[name])
assert changed == ['continuous.py', 'output_moments.py', 'policy.py', 'routing.py']
assert (PACKAGE / 'particlegan/output_moments.py').read_bytes() + b'\n' == (BASE / 'particlegan/output_moments.py').read_bytes()
subprocess.run(['git', 'diff', '--check'], cwd=REPO, check=True)
git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
assert git_head == 'dd58d843ea44eff7411bfb7dda8e75ca1cca33eb'
patch = []
for name in changed:
    a, b = BASE / 'particlegan' / name, PACKAGE / 'particlegan' / name
    patch.extend(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True),
                 fromfile=str(a), tofile=str(b)))
(ROOT / 'candidate.patch').write_text(''.join(patch))
cpu = json.loads((ROOT / 'CPU-BRIDGE.json').read_text())
assert cpu['status'] == 'CPU_PASS_LATEST_COMBINED_SOURCE_GPU_GATES_PENDING'
assert cpu['CUDA_initialized'] is False and cpu['candidate_byte_exact_current_repository']
log = (ROOT / 'all-cpu-contracts.log').read_text()
assert '104 passed' in log and 'cuda_initialized False' in log
assert str(PACKAGE / 'particlegan/__init__.py') in log
review = json.loads((ROOT / 'PEER-REVIEW.json').read_text())
assert review['status'] == 'APPROVE'
noise = json.loads((NOISE / 'receipt.json').read_text())
assert noise['original_quality_tasks_proven_unaffected'] == 19
assert noise['learned_fixture_prefixes_proven_unaffected'] == 2
assert noise['minimum_all_step_table_s_bound'] == .25
report = '''# RA17 current-PR155 source closure

RA17 copies all28 current repository modules after the clean merge of PR155
cabe208 into the byte-exact RA16 integration commit. The one root-authorized
EOF newline removal is reflected byte-exactly. Config bytes remain unchanged.
RA16 packages, freezes, completed suites, replay, and qualification stay intact.

## Changes relative to RA16

- `continuous.py`: DV12 diagnostic reductions are deferred and materialized
  as ordinary float dictionaries. The public `latent_applications` checkpoint
  key and load law remain compatible; only the internal cache representation
  changes. The actual perturbation and private draw law are unchanged.
- `routing.py`: validation predicates are detached and aggregated at complete
  CUDA forward finish. Valid logits, weights, mixed codes, token/site means,
  and gradients retain the original arithmetic. CPU invalid-value checks stay
  immediate. A closed execution now also rejects finish.
- `policy.py`: learned output-noise floor settlement excludes the noise
  tester. This intentionally changes behavior once model/table testers settle,
  and is not a universal numerical-parity claim.
- `output_moments.py`: exactly one trailing newline removed; no AST change.

## Completed CPU evidence

104 contracts pass with CUDA visible and uninitialized: the94 prior focused
contracts plus relevant new upstream noise-floor and71-site CPU regressions.
The independent RA16/RA17 trace matches every loss and complete public
checkpoint state across18 feature updates/two mean reactions and2 KNN updates,
with emitted samples, valid restore in both directions, and exact next update.
Only the diagnostic elapsed clock is shared in this synthetic fixed fixture.

Direct DV12 witnesses match outputs, gradients, private stream state, last-two
float diagnostics and serialized public state for float32/float64, with and
without CPU bf16 autocast. Direct71-site witnesses cover dependent sites,
different correlated token counts, inactive mass rows, codes/usage and table,
log-mass and logit gradients; functional callback weights match too.

## Scope of the historical noise-floor bridge

The upstream floor correction can change sigma and its derivative at the
1/64 boundary, as the actual old/new method witness demonstrates. For each of
the19 original quality sources and the Toy/MNIST2000-update learned prefixes,
the retained table tester has at most2 cumulative stationary decisions.
Stationary alone halves s; its lifetime counter persists across restart, while
reopen/drift/population revocation only raise s. Hence every earlier table
scale is at least2**(-C_stationary(T))>=1/4>1/64. Both floor laws use settle=1
throughout each original finite horizon. This is a cumulative action/source
proof, not endpoint-scale interpolation. Original40 replay windows1001..1010
lie inside those proved learned prefixes. No new post-budget horizon or manual
tester mutation is covered.

The independent reviewer approved these source/math and finite-horizon claims.
Historical evidence retains its original source labels. Current-source GPU
replay and full suite are separate fresh gates and are still pending in this
closure; no agent GPU launch occurred here.
'''
(ROOT / 'REPORT.md').write_text(report)
bridge = dict(status='CPU_PASS_LATEST_BASE_GPU_GATES_PENDING', candidate='RA17-current-pr155',
    package=str(PACKAGE), package_sha256=package_sha(PACKAGE),
    base_package=str(BASE), base_package_sha256=package_sha(BASE),
    config=str(CONFIG), config_sha256=sha(CONFIG), config_byte_identical=True,
    source_sha256=new, base_source_sha256=old, changed_modules=changed, unchanged_modules=24,
    merged_repository_head=git_head, current_PR155_base='cabe2084284db923d525918cbf3e18de6f20faac',
    current_repo_28_core_modules_byte_exact=True, repository_diff_check_pass=True,
    root_authorized_EOF_trim_exactly_one_newline=True,
    public_DV12_diagnostics_and_checkpoint_key_preserved=True,
    lazy_diagnostics_training_outputs_gradients_RNG_math_preserved=True,
    valid_many_site_routing_codes_usage_gradient_math_preserved=True,
    output_noise_floor_intentionally_corrected=True,
    universal_noise_floor_training_parity_claimed=False,
    original_21_fixture_noise_floor_finite_horizon_proof=str(NOISE / 'receipt.json'),
    original_21_fixture_noise_floor_proof_sha256=sha(NOISE / 'receipt.json'),
    original_19_quality_and_2_learned_noise_math_preserved=True,
    minimum_original_all_step_table_s_bound=.25, original_40_replay_ranges_covered=True,
    fresh_training_required_for_original_noise_floor_fixtures=[],
    original_noise_floor_proof_checkpoint_hashes=noise['inputs'],
    CPU_bridge_path=str(ROOT / 'CPU-BRIDGE.json'), CPU_bridge_sha256=sha(ROOT / 'CPU-BRIDGE.json'),
    CPU_contracts_passed=104, CUDA_visible_during_CPU_contracts=True, CUDA_initialized=False,
    actual_latest_GPU_replay_executed=False, actual_latest_full_suite_executed=False,
    current_source_GPU_replay_and_full_suite_required=True,
    historical_evidence_relabelled=False, original_RA16_freeze_valid=True,
    peer_review_sha256=sha(ROOT / 'PEER-REVIEW.json'), candidate_patch_sha256=sha(ROOT / 'candidate.patch'))
(ROOT / 'SOURCE-BRIDGE.json').write_text(json.dumps(bridge, indent=2, sort_keys=True) + '\n')
files = list((PACKAGE / 'particlegan').glob('*.py')) + list((BASE / 'particlegan').glob('*.py'))
files += list((REPO / 'particlegan').glob('*.py'))
files += [CONFIG, STUDY / 'configs/RA16-portability.json']
files += [STUDY / 'integration-prep/ra16-portability/tests/test_feature_portability.py',
          STUDY / 'integration-prep/ra16-portability/tests/fixtures/feature-auto-base.json',
          STUDY / 'integration-prep/ra15-partial-recovery/tests/test_feature_partial_recovery.py',
          STUDY / 'portability/test_ra12_contracts.py',
          STUDY / 'portability/settled-guard/test_settled_guard.py',
          STUDY / 'portability/cpu-fix/test_cpu_optimizer_scope.py',
          STUDY / 'portability/replay-alias/test_state_transfer_alias.py',
          REPO / 'tests/test_e22_noise_floor.py', REPO / 'tests/test_e22_routed_readbacks.py',
          REPO / 'examples/e22_routed_readbacks.py', REPO / 'examples/e22_routed_sites.py',
          REPO / 'examples/e22_external_loop.py']
files += [p for p in ROOT.iterdir() if p.is_file() and p.name not in ('SOURCE-FREEZE.json', 'SHA256SUMS', 'close.log')]
files += [p for p in NOISE.iterdir() if p.is_file()]
hashes = {str(p): sha(p) for p in sorted(set(files))}
hashes.update(noise['inputs'])
frozen = dict(status='FROZEN_CPU_PASS_LATEST_BASE_GPU_GATES_PENDING', candidate='RA17-current-pr155',
    package_sha256=package_sha(PACKAGE), config_sha256=sha(CONFIG), hashes=hashes)
(ROOT / 'SOURCE-FREEZE.json').write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
for path, expected in frozen['hashes'].items():assert sha(path) == expected, path
(ROOT / 'SHA256SUMS').write_text('\n'.join(f'{sha(p)}  {p.name}' for p in sorted(ROOT.iterdir())
    if p.is_file() and p.name not in ('SHA256SUMS', 'close.log')) + '\n')
print(json.dumps(dict(status=bridge['status'], package_sha256=package_sha(PACKAGE),
    config_sha256=sha(CONFIG), source_bridge_sha256=sha(ROOT / 'SOURCE-BRIDGE.json'),
    source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'), guarded_files=len(hashes),
    CPU_contracts_passed=104, GPU_replay_and_full_suite_executed=False), sort_keys=True), flush=True)
