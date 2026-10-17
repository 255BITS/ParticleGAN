"""Archive full concluded readouts and publish certified final summaries only."""
from copy import deepcopy
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from reports.forge.regenerate_technique_inventory import project_receipt


def compact(root, archive):
    archive.mkdir(parents=True, exist_ok=True)
    for path in sorted((root / 'reports/forge/records').glob('*.json')):
        current = read_json(path)
        if not str(current.get('study_id', '')).startswith('bispf-throughput-'):
            continue
        original_path = archive / path.name
        if current.get('full_readout_archive'):
            if file_hash(original_path) != current['full_readout_archive']['sha256']:
                raise ValueError('Original concluded readout changed')
        elif original_path.exists():
            if file_hash(path) != file_hash(original_path):
                raise ValueError('Refuse to replace a different archived readout')
        else:
            shutil.copyfile(path, original_path)
        original = read_json(original_path)
        record = deepcopy(original)
        originals = {(r['compatibility_key'], r['task_id']): r for r in original['task_results']}
        rows = []
        for attempt in original['attempt_ids']:
            summary = project_receipt(root, attempt)
            for row in summary['task_results']:
                actual = originals[row['compatibility_key'], row['task_id']]
                if row['gate_status'] != actual['gate_status']:
                    raise ValueError('Certified verdict differs from concluded readout')
                rows.append({**row, 'attempt_id': attempt, 'metrics': actual['metrics'],
                             'final_metrics_sha256': stable_hash(actual['metrics'])})
        if len(rows) != len(original['task_results']):
            raise ValueError('Compaction lost a task outcome')
        record['task_results'] = sorted(rows, key=lambda r:r['task_id'])
        record['original_source'] = record['source']
        record['source'] = {'path':'reports/forge/bispf-throughput/README.md'}
        record.update(qualification_input=False,
                      full_readout_sha256=file_hash(original_path),
                      full_readout_archive=dict(path=str(original_path), sha256=file_hash(original_path),
                                               git_blob_oid=subprocess.check_output(['git','hash-object',str(original_path)],text=True,cwd=root).strip(),
                                               bytes=original_path.stat().st_size),
                      summary_curation='Certified final metrics, verdicts, decision outcomes and source/receipt identities retained. Per-update evidence is archived byte-exactly outside Git. This display summary cannot replace original qualification receipts.')
        atomic_json(path,record)
        print(path.name,len(rows),path.stat().st_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    args=parser.parse_args()
    compact(ROOT,args.artifacts.resolve()/'archive/readouts')
