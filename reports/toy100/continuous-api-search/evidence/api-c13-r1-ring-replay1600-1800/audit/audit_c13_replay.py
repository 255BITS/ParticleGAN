"""Own replay receipt plus independent raw whole-checkpoint content equality."""
import hashlib
import io
import json
import math
import sys
import zipfile

sys.dont_write_bytecode = True
import audit_c13 as a


def content_digest(path):
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = name[:-len('data.pkl')]
        value = a.h.StorageOnlyUnpickler(io.BytesIO(archive.read(name))).load()
        def visit(item):
            if isinstance(item, dict) and set(item) == {'storage', 'offset', 'size', 'stride'}:
                storage = item['storage']
                width, dtype = {'FloatStorage': (4, 'torch.float32'), 'ByteStorage': (1, 'torch.uint8')}[storage['dtype']]
                expected = 1
                for length, stride in reversed(list(zip(item['size'], item['stride']))):
                    assert length <= 1 or stride == expected
                    expected *= length
                raw = archive.read(prefix + 'data/' + storage['key'])
                start = item['offset'] * width
                raw = raw[start:start + math.prod(item['size']) * width]
                digest.update(json.dumps(['tensor', dtype, list(item['size'])]).encode())
                digest.update(raw)
            elif isinstance(item, dict):
                digest.update(b'dict[')
                for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                    visit(key)
                    visit(item[key])
                digest.update(b']')
            elif isinstance(item, (list, tuple)):
                digest.update(type(item).__name__.encode() + b'[')
                for child in item:
                    visit(child)
                digest.update(b']')
            else:
                digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
        visit(value)
    return digest.hexdigest()


def main():
    folder, original = a.RUNS / 'C13-R1-checkpoint', a.RUNS / 'C13-R1-single'
    result = a.h.read_json(folder / 'continuation.json')
    assert result['exact_receipt_match'] and result['actual'] == result['expected']
    assert result['solver_telemetry_matches'] == result['solver_telemetry_checks'] == 200
    assert result['cuda_failures'] == []
    source = a.h.source_summary('C13-R1-checkpoint')
    reference = a.h.source_summary('C13-R1-single')
    assert not source['declaration_hash_mismatches'] and not source['missing_declared_files']
    assert source['package_sha256'] == reference['package_sha256']
    declaration = a.h.read_json(folder / 'declaration.json')
    assert declaration['checkpoint_sha256'] == a.h.sha((original / 'state-1600.pt').read_bytes())
    states = {}
    for step in (1740, 1750, 1800):
        candidate = folder / ('state-' + str(step) + '.pt')
        baseline = original / candidate.name
        actual, expected = content_digest(candidate), content_digest(baseline)
        assert actual == expected
        metadata = a.h.checkpoint_summary(candidate)
        assert metadata['adam_step_values'] == [[float(step)], [float(step)]]
        assert metadata['adam_step_devices'] == [['cpu'], ['cpu']] and metadata['serial_backward']
        assert metadata['critic_record']['ema_updates'] == step - 799 and metadata['critic_reference_equal']
        states[str(step)] = dict(whole_checkpoint_content_equal=True, content_sha256=actual,
            original_sha256=a.h.sha(baseline.read_bytes()), replay=metadata)
    assert content_digest(folder / 'resumed-1800.pt') == states['1800']['content_sha256']
    out = dict(scope='Stdlib restricted raw-storage inspection; no Torch/GPU/model/training execution.',
        candidate='API-C13-R1', status='VERIFIED_OWN_EXACT_REPLAY', source=source,
        continuation_sha256=a.h.sha((folder / 'continuation.json').read_bytes()), checkpoint_input_sha256=declaration['checkpoint_sha256'],
        own_steps=[1600, 1800], producer_receipt=result, independently_checked_states=states,
        limits=['200 solver telemetry matches are producer receipts; all serialized learner/data/target content at1740/1750/1800 independently compared.',
                'No stationary prefix, later-life stability or broader quality pass inferred.'])
    target = a.HERE / 'c13-r1-replay-audit.json'
    target.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(audit=str(target), source_zip_sha256=source['source_zip_sha256'],
        continuation_sha256=out['continuation_sha256'], exact_states=list(states), telemetry_checks=200), indent=2))


if __name__ == '__main__':
    main()
