#!/usr/bin/env python3
"""Bind the fixed, independently reviewed DV2/DV3 ring batch; never launch it."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--candidates', nargs='+', choices=('api-dv2','api-dv3'), required=True)
    parser.add_argument('--lane', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert len(set(args.candidates)) == len(args.candidates)
    index = read(args.preparation / 'prepared-index.json')
    sources, proofs, cases = {}, [], []
    for candidate in args.candidates:
        row = next(r for r in index['rows'] if Path(r['directory']).name == candidate)
        bundle = Path(row['directory'])
        manifest = read(bundle / 'bundle-sha256.json')
        assert sha(bundle / 'bundle-sha256.json') == row['manifest_sha256']
        files = manifest['files'] | {'bundle-sha256.json': row['manifest_sha256']}
        for name, wanted in files.items():
            assert sha(bundle / name) == wanted, name
        proof_path = Path(row['required_review'])
        review_path = proof_path.parent / 'source-review.json'
        proof, review = read(proof_path), read(review_path)
        assert proof['status'] == 'PASS' and proof['cuda_initialized'] is False
        assert proof['learner_steps'] == proof['forward_calls'] == proof['backward_calls'] == proof['optimizer_steps'] == 0
        assert all(proof['checks'].values())
        assert proof['manifest_sha256'] == review['manifest_sha256'] == row['manifest_sha256']
        assert proof['declaration_sha256'] == review['declaration_sha256'] == sha(bundle / 'declaration.json')
        assert review['status'].startswith('PASS') and all(review['checks'].values())
        assert review['cpu_receipt_sha256'] == sha(proof_path)
        assert review['worker_sha256'] == sha(bundle / 'worker.py')
        sources[candidate] = dict(root=str(bundle), files=files)
        review_group = candidate + '-review'
        sources[review_group] = dict(root=str(proof_path.parent), files={p.name: sha(p) for p in (proof_path, review_path, proof_path.parent / 'source-review.md')})
        proofs.append(dict(path=str(proof_path), sha256=sha(proof_path), required_status='PASS'))
        proofs.append(dict(path=str(review_path), sha256=sha(review_path), required_status=review['status']))
        cases.append(dict(id=candidate + '-new-init-single-shift', argv=[
            '/tmp/pr38-default-env/bin/python', '{inputs}/' + candidate + '/worker.py',
            '--reviewed-cpu-proof', '{inputs}/' + review_group + '/cpu-constructor-proof.json',
            '--output', '{output}/' + candidate + '-single-shift']))
    spec = dict(status='REVIEWED_READY_FOR_EXTERNAL_RUN', lane=args.lane, sources=sources,
        independent_reviews=proofs, cases=cases,
        instructions='Fixed existing-candidate followups only: each unchanged public candidate receives its own original ring recipe with fresh develop initialization. Preserve native lazy CPU clocks or candidate-owned eager parameter-device clocks exactly as declared. All initial non-RNG state and buffers must equal own independent CPU proof before update1. Learner total_steps=None; evaluation4600, target shift only in host after2400, no reset/target hints. Retain all460 observations, all4600 rate/controller records, full initial/change/final/error states and frozen2400 control; report first arrival, every departure and final suffix without81/81 deadline. COMPLETE means measurement completeness, not qualification. No new mechanism, retry, seed, schedule, source mutation, continuation or unlisted experiment. Exit after the listed cases, consistent with the requested bounded completion and stop.')
    args.output.write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(dict(spec=str(args.output), sha256=sha(args.output), cases=[c['id'] for c in cases], launched=False)))


if __name__ == '__main__':
    main()
