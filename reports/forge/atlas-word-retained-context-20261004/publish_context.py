"""One fixed saved-byte word illustration; no numerical reclassification."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path('/ml2/hypergan/pg-atlas-named-native-v4b-publication-20261003')
CARD_SHA = 'f287d51757c4e9397afb5435e7b3848877915b6326ba59d0659b32fab20c6be2'
RESULT_SHA = '0a175519b59b5180078d9669519060b79dc30d596daca3c8be1aca2dd9ce1d5f'
AUDIT_SHA = '4c68a3ca0061efc6d7b44c95cc43299baead0ad3e0b91a39f3c23ca6fd29b2e5'
COMMIT = 'fb7acc775b3a1a6184d36b55e035b9da04531492'
SOURCE = 'f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed'
FAMILY = 'atlas_word_joint_min11'
TASK = 'five_word_joint_acquisition_word_joint_policy_min11_v1'
MEDIA_SHA = 'c51e08932cbd4e3aab0ec6bee2e0cc9a70bcfa55b77243262af453b5fd7687ef'
GIF_SHA = '6e70906af5df22fb369de292955fe9c5a077a963f8174f99cf5645fb3ea42b4e'
PRIVATE = {'token', 'tokens', 'nonce', 'credential', 'credentials', 'password', 'secret', 'authorization',
           'access_token', 'refresh_token', 'api_key', 'lease_fd', 'lease_fds', 'lease_path'}
FLAGS = {key: False for key in ('qualification_input', 'ordinary_tier_credit', 'default_adoption',
                              'speed_ranking', 'cross_cohort_pooling', 'numeric_regrade', 'new_training_credit')}
CAPTION = ('Retained illustration only: execution INVALID; certified numerical gate UNAVAILABLE. '
           'The producer recorded 20,001 updates and 24 pure reads. The original GIF shows FAIL from '
           'a rejected health grade; that badge is not an accepted numerical verdict. No status is changed.')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path).absolute()
    if not p.is_file() or p.is_symlink() or any(q.is_symlink() for q in p.parents):
        raise ValueError('missing/unsafe retained input')
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON field')
            out[k] = v
        return out
    def bad(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)


def public(value, secrets):
    if isinstance(value, dict):
        if any(str(k).lower() in PRIVATE for k in value):
            raise ValueError('private field in companion')
        for v in value.values():
            public(v, secrets)
    elif isinstance(value, list):
        for v in value:
            public(v, secrets)
    elif isinstance(value, str):
        if any(s and s in value for s in secrets):
            raise ValueError('plaintext private nonce in companion')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite public scalar')


def publish(output):
    inputs = {}
    def checked(record):
        if pin(record['path']) != record:
            raise ValueError('changed retained input')
        inputs[record['path']] = record
        return Path(record['path'])
    def load(record):
        return read(checked(record))
    card_pin = pin(ROOT / 'final-inputs.json')
    result_pin = pin(ROOT / 'publication-final/results.json')
    audit_pin = pin(ROOT / 'actual-publication-final-audit.json')
    if (card_pin['sha256'], result_pin['sha256'], audit_pin['sha256']) != (CARD_SHA, RESULT_SHA, AUDIT_SHA):
        raise ValueError('exact immutable original cut required')
    card, results, audit = (load(p) for p in (card_pin, result_pin, audit_pin))
    ref = card['studies'][FAMILY]
    aux = ref['auxiliary'][TASK]
    study = load(ref['study'])
    raw = load(aux['raw'])
    grade = load(aux['grading'])
    resolved = load(aux['resolved'])
    supervisor = load(aux['supervisor'])
    manifest = load(results['source_manifest_input'])
    row = study['jobs'][0]
    terminal = load(row['terminal'])
    public_row = results['families'][FAMILY]['attempts'][0]
    token = row['token']
    secrets = {token}
    if (not isinstance(token, str) or not token or terminal['token'] != token or supervisor['token'] != token
            or resolved['worker']['token'] != token
            or hashlib.sha256(token.encode()).hexdigest() != public_row['token_sha256']):
        raise ValueError('foreign durable attempt')
    if (study['status'] != 'INVALID' or row['status'] != 'INVALID' or public_row['status'] != 'INVALID'
            or public_row['numerical_gate'] != 'UNAVAILABLE' or public_row['goal_gif'] is not None
            or row['task_ids'] != [TASK] or raw['task_id'] != TASK
            or row['attempt_key'] != public_row['attempt_key'] or terminal['attempt_status'] != 'completed'
            or row['unmeasured_interrupt_reserved_seconds'] != 0 or terminal['child_returncode'] != 0
            or row['paid_wall_seconds'] != terminal['paid_wall_seconds']
            or row['paid_wall_seconds'] != public_row['paid_seconds']):
        raise ValueError('unchanged INVALID/cost/status attribution required')
    if (study['source']['origin_commit'] != COMMIT or study['source']['digest'] != SOURCE
            or study['source'] != supervisor['source'] or study['source'] != resolved['packet']['source']
            or manifest != {k: v for k, v in study['source'].items() if k != 'snapshot_path'}
            or hashlib.sha256(canonical(manifest['files']).encode()).hexdigest() != SOURCE
            or results['source_file_count'] != 1390 or len(manifest['files']) != 1390
            or grade['source_digest'] != SOURCE
            or grade['raw_hash'] != hashlib.sha256(canonical(raw).encode()).hexdigest()):
        raise ValueError('raw/grade/source provenance changed')
    task = study['request']['tasks'][TASK]
    evidence = raw['evidence']
    observations = evidence['observations']
    cadence = [math.ceil(i * 20001 / 24) for i in range(1, 25)]
    if (task['execution']['steps'] != 20001 or raw['cost']['completed_steps'] != 20001
            or raw['cost']['optimizer_updates'] != {k: 20001 for k in ('generator', 'encoder', 'prior', 'discriminator')}
            or [o['step'] for o in observations] != cadence or evidence['live'] != observations[-1]
            or raw['execution_path'] != 'public_components' or raw['device'] != 'cuda:0'
            or raw['applied']['recipe']['lr'] != .0053125 or raw['applied']['recipe']['prior_lr_mult'] != 1.5
            or len(evidence['policy_purity']) != 24 or len(evidence['policy_observations']) != 24
            or len(evidence['rng_audits']) != 24):
        raise ValueError('recorded full clock/owner/cadence changed')
    if (not all(type(v) in (int, float) and math.isfinite(v) for o in observations for v in o.values())
            or not all(o['pure'] is True and o['before_sha256'] == o['after_sha256']
                       and o['global_rng_before_sha256'] == o['global_rng_after_sha256']
                       for o in evidence['policy_purity'])
            or not all(o['unintended_rng_deviations'] == 0 and not o['unintended_streams'] for o in evidence['rng_audits'])):
        raise ValueError('finite scalar/pure recorded-read receipts changed')
    expected_rejection = {'gate_status': 'FAIL', 'status': 'FAIL', 'reasons': ['observed public policy state is nonfinite']}
    if grade['grades'] != {TASK: expected_rejection} or evidence['guards']['all_finite'] is not False:
        raise ValueError('preserve exact rejected health grade')
    media_pin = pin(Path(aux['raw']['path']).with_name(TASK + '-media.json'))
    if media_pin['sha256'] != MEDIA_SHA:
        raise ValueError('original retained illustration receipt changed')
    media = load(media_pin)
    selected = [round(i * 23 / 8) for i in range(9)]
    selected_steps = [cadence[i] for i in selected]
    if (media['schema'] != 'particlegan_atlas_named_gpu_diagnostics_v1_media'
            or media['renderer_sha256'] != manifest['files']['reports/forge/atlas-named-gpu-diagnostics-v1/run_diagnostics.py']
            or media['actual_steps'] != selected_steps or media['task'] != TASK or media['family'] != FAMILY
            or media['source_digest'] != SOURCE or media['draws'] != 0 or media['optimizer_updates'] != 0
            or media['qualification_input'] is not False or media['original_gate'] != 'FAIL'
            or media['gif']['sha256'] != GIF_SHA or len(media['inputs']) != 9):
        raise ValueError('retained illustration identity changed')
    artifact_root = Path(evidence['artifact_root'])
    artifact_manifest = evidence['artifact_manifest']
    if hashlib.sha256(canonical(artifact_manifest['files']).encode()).hexdigest() != artifact_manifest['sha256']:
        raise ValueError('recorded artifact manifest digest differs')
    for record, step in zip(media['inputs'], selected_steps):
        path = checked(record)
        relative = f'observations/step_{step:06d}.npz'
        original = artifact_manifest['files'][relative]
        if path != artifact_root / relative or original != {'sha256': record['sha256'], 'size': record['bytes']}:
            raise ValueError('media arrays not bound to this raw clock/artifact manifest')
    gif = checked(media['gif'])
    from PIL import Image
    with Image.open(gif) as image:
        if image.n_frames != 9:
            raise ValueError('nine actual retained frames required')
        dimensions = [image.width, image.height]
        for frame in range(9):
            image.seek(frame)
            image.load()
    diagnosis_root = Path('/ml2/hypergan/pg-word-dimension-health-review-20261004')
    diagnosis_inputs = {}
    for name, expected in (('DIAGNOSIS.md', '7f74f3bdc73ed8a55cc0840991adc87c6e12a2efb8caa7018717f2dac2d99694'),
                           ('diagnosis.json', 'e914ace15d8d65d587ea5518804c1e9dd8417f114ac166ef8ef34a69f7eea3cc')):
        record = pin(diagnosis_root / name)
        if record['sha256'] != expected:
            raise ValueError('separate author diagnosis changed')
        checked(record)
        diagnosis_inputs[name] = record
    result = {'schema': 'pg_atlas_word_retained_context_v1', 'task_id': TASK, 'family': FAMILY,
              'execution_status': 'INVALID', 'certified_numerical_gate': 'UNAVAILABLE',
              'scope': 'Source-bound recorded clock, observation metadata and saved illustration; no new verdict or qualification.',
              'source': {'origin_commit': COMMIT, 'digest': SOURCE, 'source_files': 1390}, 'runtime': study['lane_runtime'],
              'original_cut': {'results': result_pin, 'trusted_card': card_pin, 'audit': audit_pin},
              'attempt_key': row['attempt_key'], 'token_sha256': public_row['token_sha256'],
              'durable_terminal': row['terminal'], 'producer_terminal_status': 'completed', 'producer_child_returncode': 0,
              'paid_seconds': row['paid_wall_seconds'], 'reserved_seconds': 0,
              'cost_scope': 'Already counted once in the original v4b ledger; companion adds zero training charge.',
              'raw_reported_clock': {'completed_steps': 20001, 'optimizer_updates': raw['cost']['optimizer_updates'],
                                     'execution_path': 'public_components', 'device': 'cuda:0'},
              'recorded_observation_receipts': {'count': 24, 'steps': cadence, 'finite_scalar_metadata': True,
                                               'typed_state_and_global_rng_pure': True, 'unintended_rng_deviations': 0,
                                               'purity_receipts': evidence['policy_purity']},
              'original_guard_all_finite': False, 'raw_rejected_grade': expected_rejection,
              'original_wrapper_rejection': row['reason'], 'recorded_endpoint': observations[-1],
              'original_thresholds_reference_only': task['evaluation']['thresholds'],
              'actual_prior_rows': 11, 'canonical_target_words': 5,
              'original_n5_status': 'BLOCKED, no min11-law credit',
              'retained_illustration': {'path': 'word-retained-goal.gif', 'sha256': GIF_SHA, 'bytes': media['gif']['bytes'],
                                        'frames': 9, 'actual_steps': selected_steps, 'dimensions': dimensions,
                                        'caption_required': True, 'caption': CAPTION, 'raw_renderer_badge': 'FAIL',
                                        'accepted_numeric_verdict': 'UNAVAILABLE', 'metadata_path': 'word-retained-media.json'},
              'tensor_or_checkpoint_health': 'Not inspected by this companion; separate source-defined sentinel diagnosis.',
              'separate_sentinel_diagnosis': {'public_readme': '../atlas-word-dimension-health-20261004/DIAGNOSIS.md',
                                              'inputs': diagnosis_inputs, 'checkpoint_analysis_repeated': False,
                                              'changes_original_grade': False},
              'local_raw_inputs_required_for_revalidation': True, 'automatic_hydration': False, **FLAGS}
    public(result, secrets)
    for record in inputs.values():
        checked(record)
    output = Path(output).absolute()
    if output.exists():
        raise ValueError('new companion directory required')
    output.mkdir(parents=True)
    for original, name in ((gif, 'word-retained-goal.gif'), (Path(media_pin['path']), 'word-retained-media.json')):
        if original.suffix == '.json':
            public(read(original), secrets)
        elif any(s.encode() in original.read_bytes() for s in secrets):
            raise ValueError('private nonce in retained media')
        shutil.copyfile(original, output / name)
        if sha(output / name) != sha(original):
            raise ValueError('copied original bytes differ')
    index = {'schema': 'pg_atlas_word_retained_context_input_index_v1', 'files': list(inputs.values()),
             'file_count': len(inputs), 'input_bytes_changed': False, 'no_checkpoint_loaded': True}
    public(index, secrets)
    (output / 'input-index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    result['input_index'] = {'path': 'input-index.json', 'sha256': sha(output / 'input-index.json'),
                             'bytes': (output / 'input-index.json').stat().st_size}
    result['metadata_publisher'] = {'sha256': sha(__file__), 'bytes': Path(__file__).stat().st_size,
                                    'models': 0, 'restores': 0, 'draws': 0, 'rescoring': 0, 'training_updates': 0,
                                    'gpu_calls': 0, 'queue_calls': 0}
    text = '\n'.join([
        '# Retained min11 word context', '',
        '**Execution: INVALID. Certified numerical gate: UNAVAILABLE.** The producer nevertheless recorded all '
        '**20,001 updates**, with generator, free encoder, prior and discriminator clocks each at 20,001, and '
        '**24 finite scalar observations with pure typed-state/global-RNG receipts**. These are recorded facts, '
        'not a replacement qualification verdict.', '',
        CAPTION, '',
        '![Retained nine-frame word illustration: INVALID; numerical gate UNAVAILABLE](word-retained-goal.gif)', '',
        '*' + CAPTION + '*', '',
        'The byte-original GIF retains its raw FAIL badge. That badge came from the rejected health grade '
        '`observed public policy state is nonfinite`. The diagnostic wrapper classified the attempt INVALID '
        'and withheld numerical credit. This companion preserves that decision and performs no scoring, '
        'health relaxation or model inspection.', '',
        'The supervised producer completed with child return code 0. The parent rejected its health-grade '
        'evidence and halted the lane as INVALID. This was not a producer exception or an absent training clock.', '',
        'The frozen publisher set the clock to UNAVAILABLE for every INVALID attempt and looked for an '
        '`error` field in the producer JSON. Here the producer returned complete `public_components` '
        'evidence without such a field; the rejection lives in the independent grade and study job reason. '
        'This page supplies the missing reporting context while the original portable report remains unchanged.', '',
        '**This companion supersedes only the frozen cut’s clock, error and GIF-availability wording for this '
        'attempt. Execution INVALID and numerical gate UNAVAILABLE are unchanged.**', '',
        '| Recorded endpoint at update 20,001 | Value |', '|---|---:|',
        '| Sample count | 1,024 |', '| Recovered modes | 2 of 5 |', '| Quality fraction | 0.9248046875 |',
        '| Mass TV | 0.6 |', '| Exact reconstruction | 0 |', '| Minimum reconstruction token probability | 0 |',
        '| Reconstruction NLL | 11.05240844637142 |', '',
        'The selected effective-code law has eleven actual prior rows, five original target words and a free '
        'encoder. Original N5 remains separately BLOCKED. The original six thresholds are retained as '
        'reference metadata in [context.json](context.json); no threshold verdict is recomputed here.', '',
        f'Source `{COMMIT}` / `{SOURCE}`, physical GPU1 / logical `cuda:0`. Recorded paid cost '
        '**558.5739127129782 seconds**, reserve zero, was already charged in the original v4b ledger; '
        'this companion introduces no new training charge, default, speed, ordinary-tier or cross-cohort credit.', '',
        '[Original unchanged v4b report](../atlas-named-gpu-diagnostics-native-v4b-20261004/README.md). '
        '[Byte-original retained media receipt](word-retained-media.json), [source-bound context](context.json), '
        '[input index](input-index.json). [Separate sentinel diagnosis](../atlas-word-dimension-health-20261004/DIAGNOSIS.md) '
        'is pinned separately and does not change the old INVALID status. This companion does not inspect '
        'checkpoints or repeat that tensor-health analysis.', '',
        'Share the illustration with this caption/page. A detached raw GIF badge is not an accepted numerical '
        'verdict. The compact files are portable; full hash revalidation requires the local immutable inputs '
        'in the index. No raw checkpoint, array, log, internal nonce or lease descriptor is copied.', ''])
    public(text, secrets)
    public(result, secrets)
    (output / 'context.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'README.md').write_text(text)
    for record in inputs.values():
        checked(record)
    audit = {'schema': 'pg_atlas_word_retained_context_audit_v1', 'status': 'PASS',
             'inputs': len(inputs), 'raw_clock': 20001, 'pure_reads': 24, 'gif_frames': 9,
             'all_input_hashes_unchanged': True, 'copied_gif_and_media_byte_original': True,
             'private_fields_or_plaintext_nonces': 0, 'model_checkpoint_loads_draws_rescoring_gpu_queue_calls': 0,
             'source': SOURCE, 'execution_status': 'INVALID', 'numeric_gate': 'UNAVAILABLE',
             'files': {f.name: {'sha256': sha(f), 'bytes': f.stat().st_size} for f in output.iterdir() if f.is_file()}, **FLAGS}
    public(audit, secrets)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    return {'status': 'PUBLISHED_RETAINED_INVALID_CONTEXT', 'inputs': len(inputs), 'output': str(output),
            'context_sha256': sha(output / 'context.json'), 'audit_sha256': sha(output / 'audit.json'),
            'gif_sha256': GIF_SHA, 'clock': 20001, 'observations': 24, 'numerical_gate': 'UNAVAILABLE'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(publish(parser.parse_args().output), sort_keys=True))
