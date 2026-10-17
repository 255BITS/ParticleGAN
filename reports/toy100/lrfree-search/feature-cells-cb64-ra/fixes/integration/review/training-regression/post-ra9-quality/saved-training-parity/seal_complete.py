"""Seal the completed watcher only after its numerical-free parse process exits."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    assert not (HERE/'FROZEN.json').exists()
    pid=json.loads((HERE/'WATCH-PID.json').read_text());proc=Path(f'/proc/{pid["pid"]}/stat')
    if proc.exists():
        fields=proc.read_text().rsplit(')',1)[1].split()
        if fields[19]==pid['start_ticks'] and fields[0]!='Z':raise RuntimeError('Watcher still active')
    source=json.loads((HERE/'SOURCE-FROZEN.json').read_text())
    for path,value in source['files'].items():assert sha(path)==value
    area=HERE/'accepted-attempt1';summary=json.loads((area/'summary.json').read_text())
    assert len(summary['records'])==10
    files={};checkpoints={}
    for row in summary['records']:
        receipt=Path(row['receipt_path']);assert sha(receipt)==row['receipt_sha256']
        packet=json.loads(receipt.read_text());checkpoints.update(packet['checkpoint_sha256'])
        assert packet['new_training_steps']==packet['new_emissions']==0 and not packet['numerical_replay']
    for path,value in checkpoints.items():assert sha(path)==value
    completed=datetime.now(timezone.utc).isoformat()
    total=sum(row['unexpected_difference_count'] for row in summary['records'])
    rows='\n'.join(f'| {r["step"]} | {r["actual_cells"]} | {r["metric_rank"]} | {r["unexpected_difference_count"]} | {r["coherent_rows"]} | {r["paired_average_lease_live"]} |' for r in summary['records'])
    report=f'''# Saved RA9 versus RA8 training neutrality

Status: **{summary['status']}** across all ten original saved toy endpoints.
Unexpected serialized training differences: **{total}**.

| Update | Actual K | Rank | Differences | Coherent rows | Serving lease |
|---:|---:|---:|---:|---:|:---:|
{rows}

Every serialized training leaf is compared exactly, including FAST/EMA model and
buffer values, both optimizers and row history, controller/settlers/evidence,
dedicated streams and CPU/CUDA RNG bytes, FIFO/lineage/actions/count and birth/copy
state, serving stamps, work and action counters. Only backend8→7, requested cells
128→64 in recipe/settings, the verified new resolution-policy setting, and the
original last.eval_seconds diagnostic are normalized.

The helper was frozen before any RA9 numerical checkpoint was opened. Both
variants' original post-save log events seal inputs; checkpoint and guarded source
hashes are verified before and after each CPU parse and again after watcher exit.
Actual fitted chart/count metadata is recognized by the frozen production scalar
validator. Existing typed bit-comparison functions are AST-identical to the
previous frozen audit. Original checkpoint objects and global CPU RNG are unchanged.

This is descriptive CPU parsing of saved endpoints, **not numerical replay,
intermediate-step equivalence, emissions, training or quality acceptance**.
No model construction/forward, RNG restore/consumption, CUDA context, optimizer
step or new seed is performed. The original quality gates remain separate.
Outer quality/log/provenance records are outside serialized training parity.
Closed post-exit seal: {completed}.
'''
    with (HERE/'REPORT.md').open('x') as target:target.write(report)
    value=dict(status=summary['status'],closed_utc=completed,endpoints=10,unexpected_difference_count=total,
               records=summary['records'],helper_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),checkpoint_sha256=checkpoints,
               source_sha256=source['files'],root_identities=source['root_identities'],watcher_exited=True,
               CPU_only=True,numerical_replay=False,new_training_steps=0,new_emissions=0,quality_verdict=None)
    with (HERE/'receipt.json').open('x') as target:target.write(json.dumps(value,indent=2)+'\n')
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:files[str(path)]=sha(path)
    frozen=dict(status='POST_EXIT_FROZEN_SAVED_ENDPOINT_CPU_PARSE',closed_utc=completed,files=files,
                checkpoint_sha256=checkpoints,source_sha256=source['files'],quality_verdict=None)
    with (HERE/'FROZEN.json').open('x') as target:target.write(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],unexpected_difference_count=total,
                         receipt_sha256=sha(HERE/'receipt.json'),frozen_sha256=sha(HERE/'FROZEN.json'))),flush=True)


if __name__=='__main__':main()
