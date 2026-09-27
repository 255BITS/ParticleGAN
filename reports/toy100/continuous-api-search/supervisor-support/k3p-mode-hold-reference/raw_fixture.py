"""Restricted raw checkpoint metadata reader; no Torch import or tensor execution."""
from collections import OrderedDict
import hashlib
import io
import math
import pickle
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rebuild(storage, offset, size, stride, *unused):
    return dict(storage=storage, offset=offset, size=size, stride=stride)


class StorageOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return rebuild
        if module == 'torch' and name in ('FloatStorage', 'ByteStorage'):
            return name
        raise ValueError('unapproved pickle global: ' + module + '.' + name)

    def persistent_load(self, value):
        assert value[0] == 'storage'
        return dict(dtype=value[1], key=value[2], device=value[3], length=value[4])


def inspect_checkpoint(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('/data.pkl'))
        prefix = name[:-len('data.pkl')]
        envelope = StorageOnlyUnpickler(io.BytesIO(z.read(name))).load()
        state = envelope['trainer']

        def tensor(t):
            width, dtype = {'FloatStorage': (4, 'torch.float32'), 'ByteStorage': (1, 'torch.uint8')}[t['storage']['dtype']]
            count, expected = math.prod(t['size']), 1
            for length, stride in reversed(list(zip(t['size'], t['stride']))):
                assert length <= 1 or stride == expected
                expected *= length
            raw = z.read(prefix + 'data/' + t['storage']['key'])
            start = t['offset'] * width
            return dict(shape=list(t['size']), dtype=dtype, sha256=sha(raw[start:start + count * width]))

        return dict(completed_steps=state['completed_steps'],
                    models={m: {k: tensor(t) for k, t in values.items()} for m, values in state['models'].items()},
                    streams={k: tensor(v) for k, v in state['streams'].items()},
                    cpu_rng=tensor(state['cpu_rng']), cuda_rng=tensor(state['cuda_rng']),
                    data_rng=tensor(envelope['data_rng']),
                    native_state_counts=[len(o['state']) for o in state['optimizers']])
