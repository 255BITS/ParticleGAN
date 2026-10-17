"""Seal a compact additive archive of the closed dense comparison; no Torch import."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
CONV = HERE.parent
STUDY = CONV.parent
REPO = Path('/ml2/hypergan/ParticleGAN-ra11-pr155')
REPORT = Path('reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930')
PREFIX = Path('visualization-pr223-convergence')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def validate_closures():
    source = read(CONV / 'SOURCE-FREEZE.json')
    data = read(CONV / 'closure-v1/DATA-RECEIPT.json')
    assert data['status'] == 'CLOSED_VALID_BOTH_ORIGINAL_QUALITY_GATES_PASS'
    checks = {}
    for name, key in [('SOURCE-FREEZE.json', 'hashes'),
                      ('BASELINE-SOURCE-FREEZE.json', 'hashes'),
                      ('closure-v1/DATA-RECEIPT.json', 'inputs_sha256'),
                      ('closure-v1/EVIDENCE-FREEZE.json', 'files')]:
        values = read(CONV / name)[key]
        for original, digest in values.items():
            original = str(Path(original) if Path(original).is_absolute() else (CONV / name).parent / original)
            if isinstance(digest, dict):
                assert Path(original).stat().st_size == digest['bytes']
                digest = digest['sha256']
            assert sha(original) == digest, original
            if original in checks:
                assert checks[original] == digest
            checks[original] = digest
    for name, status in [('parity-attempt-1/PARITY-COMPLETION.json', 'PASS'),
                         ('full-attempt-1/COMPLETION.json', 'COMPLETE')]:
        value = read(CONV / name)
        assert value['status'] == status and value['evidence_validity'] == 'VALID'
    assert len(source['hashes']) == 104
    return checks, source, data


def verify():
    manifest = read(HERE / 'ARCHIVE-APPEND.json')
    freeze = read(HERE / 'FROZEN.json')
    for item in manifest['files']:
        for path in [Path(item['source']), HERE / 'payload' / item['path']]:
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
    for path, digest in freeze['files'].items():
        assert sha(path) == digest, path
    checks, _, _ = validate_closures()
    print(json.dumps(dict(status='VALID', files=manifest['file_count'],
                         total_bytes=manifest['total_bytes'],
                         closure_inputs=len(checks), manifest_sha256=sha(HERE / 'ARCHIVE-APPEND.json'))))


def prepare():
    assert not (HERE / 'ARCHIVE-APPEND.json').exists(), 'Use a fresh archive attempt.'
    checks, source, data = validate_closures()
    selected = {}

    def add(original, relative, role):
        original = Path(original)
        target = PREFIX / relative
        assert original.is_file() and original.suffix not in ('.pt', '.npz', '.pyc')
        assert target not in selected, str(target)
        selected[target] = dict(source=str(original), role=role)

    for name in ['run_capture.py', 'instrumentation.py', 'check_source.py', 'e22_runner.py',
                 'atlas_runner.py', 'PROTOCOL.md', 'SOURCE-FREEZE.json',
                 'BASELINE-SOURCE-FREEZE.json', 'E22-RECIPE-IDENTITY.json',
                 'HOST-ADAPTER-CHECK.json', 'CPU-CHECK-E22.json', 'CPU-CHECK-Atlas.json',
                 'cpu-check-e22.log', 'cpu-check-atlas.log', 'cpu-check-e22-attempt-1.log']:
        add(CONV / name, name, 'observer_source_or_preflight')
    for name in ['configs', 'source-reference', 'pkg-pr155-e22-cabe2084']:
        for path in sorted((CONV / name).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                add(path, path.relative_to(CONV), 'baseline_source_or_config')
    for path in sorted((STUDY / 'pkg-RA17-current-pr155/particlegan').rglob('*.py')):
        add(path, Path('pkg-RA17-current-pr155/particlegan') / path.name, 'atlas_source')
    for path in sorted(Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100').glob('*.py')):
        add(path, Path('source-reference/hosts/native100') / path.name, 'original_host_and_scorer')
    add('/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py',
        'source-reference/original-rotate-gate.py', 'original_runner')
    add(STUDY / 'validation-ra15/moving/rotated100/adapted_runner.py',
        'source-reference/original-ra15-adapted-runner.py', 'original_atlas_adapter')
    for name in ['COMPLETION.json', 'LAUNCH.json', 'frames.npz.verdict.json']:
        add(STUDY / 'validation-ra15/moving/rotated100' / name,
            Path('source-reference/historical-atlas') / name, 'historical_semantic_parity_receipt')
    for name in ['portability/ra17-current-pr155/SOURCE-FREEZE.json', 'validation-ra17/SOURCE-FREEZE.json']:
        add(STUDY / name, Path('source-reference/study-guards') / name, 'prior_source_guard')
    for name in ['parity-attempt-1', 'full-attempt-1', 'closure-v1',
                 'cpu-observer-e22', 'cpu-observer-atlas']:
        for path in sorted((CONV / name).rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.py') and '__pycache__' not in path.parts:
                add(path, path.relative_to(CONV), 'closed_parity_full_or_data_evidence')

    # All raw data stays local, including the old checkpoints used in semantic parity.
    raw_paths = set()
    for path in CONV.rglob('*'):
        relative = path.relative_to(CONV)
        if path.is_file() and path.suffix in ('.pt', '.npz') and relative.parts[0] not in ('render', 'archive-v1'):
            raw_paths.add(path)
    raw_paths.update(Path(path) for path in checks if Path(path).suffix in ('.pt', '.npz'))
    local = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                  role='local_checkpoint' if path.suffix == '.pt' else 'local_sample_cloud')
             for path in sorted(raw_paths)]
    by_source = {item['source']: str(target) for target, item in selected.items()}
    closure_inputs = []
    for path, digest in sorted(checks.items()):
        archived = by_source.get(path)
        # Public core guards have identical bytes to the archived Atlas package.
        if archived is None and path.startswith(str(REPO / 'particlegan') + '/'):
            candidate = PREFIX / 'pkg-RA17-current-pr155/particlegan' / Path(path).name
            assert sha(selected[candidate]['source']) == digest
            archived = str(candidate)
        closure_inputs.append(dict(path=path, sha256=digest, bytes=Path(path).stat().st_size,
                                   archived_path=archived, storage='archive' if archived else 'local_reference'))
    write(HERE / 'LOCAL-REFERENCES.json', dict(schema_version=1, raw_file_count=len(local),
          raw_total_bytes=sum(item['bytes'] for item in local), raw_files=local,
          media_append='Separate renderer closure; no media, individual rendered frames or layout drafts in this append.'))
    write(HERE / 'CLOSURE-INPUTS.json', dict(schema_version=1, closure_input_count=len(closure_inputs),
          inputs=closure_inputs, original_absolute_paths_retained=True,
          relocation_note='archived_path locates byte-identical source/evidence copies; raw state and clouds remain at original local paths.'))
    (HERE / 'README.md').write_text('''# Dense E22 versus ParticleGAN Atlas evidence append

This additive archive records the actual fresh rotated100 runs: current PR155 E22
at `cabe2084284db923d525918cbf3e18de6f20faac` versus ParticleGAN Atlas RA17.
Both used the original fixture, seed 1234, initial models, real streams, 1,500
updates, scorer and 20,000-point acceptance draws. The two original target turns
remain at updates 500 and 1,000. Each formulation retains its own control recipe.

| Original acceptance draw | E22 HQ / modes | Atlas HQ / modes |
| --- | --- | --- |
| 500, before first turn | 88.10% / 100 | 96.09% / 100 |
| 1,000, after first turn | 83.99% / 100 | 96.09% / 100 |
| 1,500, after second turn | 94.73% / 100 | 92.59% / 98 |

Both pass both original turn gates: at least 95 modes and HQ at least 90% of
their own update-500 baseline. Atlas reaches 90% on the separate visualization
sample earlier during initial learning and first-turn recovery; E22 has higher
final original HQ and more final modes. These runs support a scoped comparison,
not a universal final-score advantage.

Each dense recording contains 151 observed 4,096-point float32 clouds at updates
0, 10, ..., 1,500. Two additional target-transition frames reuse the preceding
cloud exactly and rotate only the target centers. There is no interpolation.
The visualization metrics use these 4,096 points, including a mode threshold of
at least 10 HQ points; they are separate from the original 20,000-point gates.
All 302 observation state guards passed. The paired GPU observer diagnostic
passed for E22 fresh updates 1–20 and Atlas restored updates 1,001–1,010. Atlas
full-run checkpoint state and original gate metrics match the prior frozen
Atlas run, excluding only recorded evaluation elapsed seconds from state parity.

## Archive format

`../ARCHIVE-APPEND.json` is schema version 1. Paths in its `files` are relative
to the generalization report root; each entry gives original `source`, `path`,
`bytes`, `sha256` and `role`. Copy absent paths, accept byte-identical existing
paths and reject every conflict. `prepare_archive.py --verify` validates this
local staging tree and the unchanged original closures without importing Torch.

The manifest covers copied payload files. The manifest itself and `FROZEN.json`
are explicitly listed as closure controls outside its own file table to avoid
recursive hashes. `FROZEN.json` hashes the manifest and all payload members.
`PREPARATION-RECEIPT.json` records existing byte-identical report files and the
exact ignored paths needing `git add -f`; trace JSONL filenames are checked
individually. Raw `.pt` checkpoints and `.npz` point clouds stay local and are
listed in `LOCAL-REFERENCES.json` with size and SHA-256. `CLOSURE-INPUTS.json`
retains every source/data/closure input hash and maps it to an archived copy
where available. Original receipt bytes retain their absolute study paths.

## Separate media append

The renderer owns a separate closure for the GIF, MP4, poster, renderer and
render receipt. It must be added with its own manifest after media review.
Rendered per-frame PNGs and all draft/layout attempts remain in the local study.
Earlier four-frame RA14/RA15 artifacts are separate historical evidence and are
not the current E22 versus Atlas comparison.
''')
    for name in ['prepare_archive.py', 'README.md', 'LOCAL-REFERENCES.json', 'CLOSURE-INPUTS.json']:
        add(HERE / name, Path('archive-v1') / name, 'archive_control')

    existing = []
    for target, item in sorted(selected.items()):
        candidate = REPO / REPORT / target
        if candidate.exists():
            assert candidate.is_file() and sha(candidate) == sha(item['source']), str(candidate)
            existing.append(str(target))
    report_paths = [str(REPORT / target) for target in sorted(selected)]
    ignore = subprocess.run(['git', 'check-ignore', '--no-index', '-z', '--stdin'],
                            input=('\0'.join(report_paths) + '\0').encode(),
                            cwd=REPO, capture_output=True, check=False)
    assert ignore.returncode in (0, 1), ignore.stderr.decode()
    ignored = sorted(path for path in ignore.stdout.decode().split('\0') if path)
    write(HERE / 'PREPARATION-RECEIPT.json', dict(schema_version=1,
          status='VALID_ADDITIVE_PAYLOAD_PREPARED_NO_REPOSITORY_WRITES',
          closure_inputs_checked=len(checks), original_source_guards=104,
          existing_byte_identical_paths=existing, ignored_paths_force_add=ignored,
          jsonl_paths_checked=[path for path in report_paths if path.endswith('.jsonl')],
          ignored_jsonl_paths=[path for path in ignored if path.endswith('.jsonl')],
          observation_guards=data['all_observation_state_guards_pass'],
          raw_file_count=len(local), raw_total_bytes=sum(item['bytes'] for item in local)))
    add(HERE / 'PREPARATION-RECEIPT.json', 'archive-v1/PREPARATION-RECEIPT.json', 'archive_control')
    files = []
    for target, item in sorted(selected.items()):
        original = Path(item['source'])
        entry = dict(item, path=str(target), bytes=original.stat().st_size, sha256=sha(original))
        files.append(entry)
        staged = HERE / 'payload' / target
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, staged)
    manifest = dict(schema_version=1, status='SEALED_ADDITIVE_DENSE_CURRENT_E22_ATLAS_COMPARISON',
        created_utc=datetime.now(timezone.utc).isoformat(), report_root=str(REPORT),
        root=str(STUDY), file_count=len(files), total_bytes=sum(item['bytes'] for item in files), files=files,
        role_counts=dict(Counter(item['role'] for item in files)),
        copy_policy='Add absent relative paths; accept existing byte-identical targets; reject all conflicts.',
        package_raw_sha256={key: value['sha256'] for key, value in source['packages'].items()},
        config_sha256=source['config_sha256'], source_freeze_sha256=sha(CONV / 'SOURCE-FREEZE.json'),
        data_receipt_sha256=sha(CONV / 'closure-v1/DATA-RECEIPT.json'),
        parity_completion_sha256=sha(CONV / 'parity-attempt-1/PARITY-COMPLETION.json'),
        full_completion_sha256=sha(CONV / 'full-attempt-1/COMPLETION.json'),
        closure_controls_outside_file_table=[str(PREFIX / 'ARCHIVE-APPEND.json'), str(PREFIX / 'archive-v1/FROZEN.json')],
        excludes=['raw checkpoints and runtime states (.pt)', 'raw sample clouds (.npz)',
                  'media and individual rendered frames (separate renderer append)',
                  'layout drafts', 'bytecode caches', 'lockfiles'],
        media_append='Separate renderer manifest and review; this append makes no media approval claim.')
    write(HERE / 'ARCHIVE-APPEND.json', manifest)
    staged_manifest = HERE / 'payload' / PREFIX / 'ARCHIVE-APPEND.json'
    shutil.copyfile(HERE / 'ARCHIVE-APPEND.json', staged_manifest)
    assert validate_closures()[0] == checks
    freeze = dict(schema_version=1, status='CLOSED_VALID_ADDITIVE_PAYLOAD',
        files={str(HERE / 'ARCHIVE-APPEND.json'): sha(HERE / 'ARCHIVE-APPEND.json'),
               **{item['source']: item['sha256'] for item in files}},
        payload_file_count=len(files), payload_bytes=manifest['total_bytes'],
        copy_file_count=len(files) + 2,
        copy_bytes_excluding_this_freeze=manifest['total_bytes'] + staged_manifest.stat().st_size,
        original_closures_unchanged=True, no_repository_writes=True, no_GPU_operations=True)
    write(HERE / 'FROZEN.json', freeze)
    shutil.copyfile(HERE / 'FROZEN.json', HERE / 'payload' / PREFIX / 'archive-v1/FROZEN.json')
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else prepare()
