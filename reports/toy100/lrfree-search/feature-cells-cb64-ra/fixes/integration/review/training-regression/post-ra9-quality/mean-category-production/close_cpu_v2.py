"""Post-exit mechanics receipt using JSON/raw hashes only; no numerical imports."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    result_path=HERE/'cpu-contract-attempt2/result.json'
    result=json.loads(result_path.read_text())
    assert result['status']=='PASS' and result['device']=='cpu' and not result['cuda_initialized']
    assert [r['case'] for r in result['records']]==['grid','toy']
    assert all(r['status']=='PASS' and all(r['checks'].values()) for r in result['records'])
    assert result['new_training_steps']==result['new_optimizer_steps']==result['new_quality_emissions']==0
    source=json.loads((HERE/'SOURCE-FROZEN-v2.json').read_text())
    for p,d in source['source_and_input_sha256'].items():assert sha(p)==d,p
    outputs=[result_path,HERE/'cpu-contract-attempt2.log',Path(__file__)]
    for case in ('grid','toy'):
        outputs.extend(p for p in (HERE/'cpu-contract-attempt2'/case).iterdir() if p.is_file())
    for record in result['records']:
        for p,d in record['output_sha256'].items():assert sha(p)==d,p
    records=result['records'];g,t=records
    report=f'''# RA10 CPU reaction mechanics

The corrected fixture v2 passed one actual fresh backend9 reaction per frozen
RA9 grid/toy raw input, plus the unchanged caller reset/rebase hook. The exact
production package/config/source law stayed unchanged throughout both helper
attempts. The failed v1 fixture and metadata read remain sealed separately.

| Fixed input | Witness lower bound | Ordinary moves | Mean copies | Mean attempts |
|---|---:|---:|---:|---:|
| Grid final raw state | {g['witness']['lower_bound']:.9f} | {g['ordinary']} | {g['mean']} | {g['witness']['attempts']} |
| Toy final raw state | {t['witness']['lower_bound']:.9f} | {t['ordinary']} | {t['mean']} | {t['witness']['attempts']} |

Grid's current EMA feature objective decreased from
{g['witness']['objective_before_mean']} to {g['witness']['objective_after_mean']}.
Both-view actual inside/category/group/support predicates, exact prepared
coordinates and optimizer/history inheritance, unique sources/children,
complete prefix reservations, the shared5% ordinary cap, all moved reset/rebase
rows and final fresh paired lease passed. All model weights, dedicated training
streams and global CPU RNG were unchanged. The real FIFO and named raw inputs
were asserted exact after the native constructor.

These are fixed-input CPU mechanics, not historical GPU replay, emitted quality,
distribution equivalence or population certification. Source/helper/input guards
were frozen and independently reviewed before v2 execution. Fresh backend9
before/after fixtures and the grid packet sidecar support separate ownership and
cold-load/continuation checks; those controls provide their own receipts.

No CUDA, GAN gradient/optimizer step, new quality sample or seed experiment was
run here. Root alone launches the same frozen helper's CUDA mechanics branch,
then the unchanged full toy25/grid100 quality gates if all reviews qualify.
'''
    report_path=HERE/'CPU-REPORT.md';assert not report_path.exists();report_path.write_text(report)
    outputs.append(report_path)
    receipt=dict(status='PASS',post_exit=True,closed_UTC=datetime.now(timezone.utc).isoformat(),
        fixture_version=2,backend_schema=9,trainer_schema=5,records=records,
        source_and_input_sha256=source['source_and_input_sha256'],
        protected_preseal_sha256={str(HERE/'SOURCE-FROZEN.json'):sha(HERE/'SOURCE-FROZEN.json'),
            str(HERE/'SOURCE-FROZEN-v2.json'):sha(HERE/'SOURCE-FROZEN-v2.json')},
        output_and_reporting_sha256={str(p):sha(p) for p in outputs},
        numerical_imports_in_closer=0,quality_verdict=None,
        authoritative_fixture_directory=str(HERE/'cpu-contract-attempt2'),
        failed_helper_evidence_retained=str(HERE/'FAILED-V1-FROZEN.json'))
    path=HERE/'CPU-RECEIPT.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',receipt=str(path),sha256=sha(path))))

if __name__=='__main__':main()
