"""Read certified saved Sinkhorn counters; no model construction, updates or draws."""
import argparse
import importlib.util
import json
from pathlib import Path


def load_module(path):
    spec = importlib.util.spec_from_file_location('sinkhorn_saved_publisher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    publication = json.loads((args.publication / 'phase3-results.json').read_text())
    pub = load_module(root / 'reports/forge/bcap-develop-integration/publish.py')
    rows = []
    for item in publication['task_results']:
        if item['role'] != 'candidate' or item['gate_status'] not in {'PASS', 'FAIL'}:
            continue
        certified = pub.certified_attempt(root, item['attempt_id'])
        candidates = [r for r in certified['result']['task_results'] if r['task_id'] == item['task_id']]
        pub.require(len(candidates) == 1, 'Ambiguous certified task row')
        saved, proof = pub.checkpoint(candidates[0])
        pub.require(saved is not None, 'Complete measured task is missing its provenance state')
        stats = pub.mechanism_stats(saved)
        counters = {}
        for path, values in stats.items():
            counter = values.get('sinkhorn') if values.get('consumer') == 'output_marginal_v1' else values
            if isinstance(counter, dict) and counter.get('consumer') == 'sinkhorn_finite_dual_v1':
                counters[path] = counter
        pub.require(counters, 'Active complete candidate is missing actual-call counters')
        canonical = next(iter(counters.values()))
        pub.require(all(c == canonical for c in counters.values()), 'Different duplicate counter copies')
        pub.require(canonical['iterations'] == 72 * canonical['calls'], 'Finite iteration budget differs')
        rows.append(dict(task_id=item['task_id'], gate_status=item['gate_status'], attempt_id=item['attempt_id'],
            checkpoint=proof, duplicate_counter_paths=list(counters), counters=canonical,
            mean_finite_dual_loss=canonical['loss_sum'] / canonical['calls'] if canonical['calls'] else None))
    receipt = dict(schema_version=1, scope='certified_saved_sinkhorn_counters', qualification_input=False,
        source_commit=publication['source_commit'], source_digest=publication['source_digest'],
        publication_sha256=pub.file_hash(args.publication / 'phase3-results.json'),
        producer_holds_include_their_restored_prefix=True, duplicate_copies_counted_once=True,
        rows=rows, optimizer_updates_added=0, sampling_draws_added=0)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(event='sinkhorn_saved_counters', tasks=len(rows), output=str(args.output))))


if __name__ == '__main__':
    main()
