#!/usr/bin/env python
"""Fetch only selected Anima tensors, with a pinned revision and byte hashes.

No pickle is downloaded or executed. The remote file is safetensors; the local
torch bundle contains tensors and primitive metadata and loads weights_only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

import torch

REPO = 'circlestone-labs/Anima'
REVISION = 'f973fc41ec7545364ac9776c2440285f43ff2a30'
FILENAME = 'split_files/diffusion_models/anima-base-v1.0.safetensors'


def select_tensors(header, blocks):
    """Normalize Base and Turbo export namespaces without changing tensors."""
    groups = [f'blocks.{i}.' for i in blocks] + ['t_embedder.', 't_embedding_norm.']
    roots = [root for root in ('net.', 'model.diffusion_model.')
             if all(any(k.startswith(root + group) for k in header) for group in groups)]
    if len(roots) != 1:
        raise ValueError('Missing or ambiguous selected tensor groups')
    root = roots[0]
    selected = {k: v for k, v in header.items()
                if any(k.startswith(root + group) for group in groups)}
    return root, selected


def read_range(url, start, length):
    request = urllib.request.Request(url + f'?range={start}-{length}',
                                    headers={'Range': f'bytes={start}-{start+length-1}'})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError('Server must support byte ranges; refusing full download')
        expected = f'bytes {start}-{start+length-1}/'
        if not response.headers.get('Content-Range', '').startswith(expected):
            raise RuntimeError('Unexpected HTTP byte range')
        data = response.read(length + 1)
    if len(data) != length:
        raise RuntimeError('Incomplete HTTP byte range')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--out', default='data/anima/blocks_0_1.pt')
    parser.add_argument('--revision', default=REVISION, help='Pinned Hugging Face commit')
    parser.add_argument('--filename', default=FILENAME, help='Safetensors path within the repository')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Revision must be a full lowercase commit SHA')
    url = f'https://huggingface.co/{REPO}/resolve/{args.revision}/{args.filename}'
    size = struct.unpack('<Q', read_range(url, 0, 8))[0]
    if size > 10_000_000:
        raise ValueError('Unexpected header size')
    raw = read_range(url, 8, size)
    header = json.loads(raw)
    root, selected = select_tensors(header, args.blocks)
    # Merge adjacent selected tensors; do not transfer the text adapter or other blocks.
    spans = []
    for k, v in sorted(selected.items(), key=lambda item: item[1]['data_offsets'][0]):
        a, b = v['data_offsets']
        if spans and a == spans[-1][1]:
            spans[-1][1] = b
            spans[-1][2].append(k)
        else:
            spans.append([a, b, [k]])
    tensors, hashes = {}, {}
    dtypes = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
    for lo, hi, names in spans:
        print(f'FETCH {len(names)} tensors, {(hi-lo)/1e6:.1f} MB', flush=True)
        payload = read_range(url, 8 + size + lo, hi - lo)
        for k in names:
            spec = selected[k]
            a, b = spec['data_offsets']
            data = bytearray(payload[a-lo:b-lo])
            hashes[k] = hashlib.sha256(data).hexdigest()
            tensor = torch.frombuffer(data, dtype=dtypes[spec['dtype']]).clone()
            if tensor.numel() != math.prod(spec['shape']):
                raise ValueError(f'Invalid shape for {k}')
            tensors[k.removeprefix(root)] = tensor.reshape(spec['shape'])
    metadata = {'repository': REPO, 'revision': args.revision, 'filename': args.filename,
                'blocks': args.blocks, 'source_prefix': root,
                'header_sha256': hashlib.sha256(raw).hexdigest(),
                'tensor_sha256': hashes}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'tensors': tensors, 'metadata': metadata}, out.with_suffix('.tmp'))
    out.with_suffix('.tmp').replace(out)
    digest = hashlib.file_digest(out.open('rb'), 'sha256').hexdigest()
    out.with_suffix('.json').write_text(json.dumps({**metadata, 'bundle_sha256': digest}, indent=2) + '\n')
    print(f'COMPLETE {out} sha256={digest}', flush=True)


if __name__ == '__main__':
    main()
