"""Conclude frozen studies and archive every attempt, including execution timeout."""
from pathlib import Path
import json
from experiments.forge.contracts import read_json,atomic_json,file_hash,stable_hash
from experiments.forge import knowledge
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/queue')

def main():
    results=read_json(OUT/'results.json');state=read_json(QUEUE/'queue/state.json');history=[]
    assert len(results['task_results'])==15
    assert all(j['status']=='terminal' for j in state['jobs'].values())
    for job in state['jobs'].values():
        for attempt in job['attempts']:
            aid=attempt['attempt_id'];durable=ROOT/'reports/forge/attempts'/aid
            result=read_json(durable/'result.json');certificate=read_json(durable/'evidence.json')
            assert certificate['result_hash']==stable_hash(result)
            history.append(dict(attempt_id=aid,task_id=job['definition']['task_id'],final_attempt=aid==job['result']['attempt_id'],
                gate_statuses={r['task_id']:r['gate_status'] for r in result['task_results']},
                paid_seconds=next(c['seconds'] for c in state['charges'] if c['attempt_id']==aid),
                full_reservation=job['definition']['budget_seconds'],local_artifact_root=str(Path(attempt['path'])),
                receipt_files={n:dict(path=str((durable/(n+'.json')).relative_to(ROOT)),sha256=file_hash(durable/(n+'.json'))) for n in ('request','result','evidence')}))
    atomic_json(OUT/'execution-history.json',dict(schema_version=1,qualification_input=False,attempts=history,
        original_timeout_preserved=True,final_attempts=15,total_paid_attempts=len(history),full_reservations=sum(a['full_reservation'] for a in history),
        paid_seconds=sum(a['paid_seconds'] for a in history),execution_repair='Same frozen science; physicalGPU1/logicalcuda:0, poll5s, pre-import BLAS/OpenMP threads1; scientific Torch threadbudget1.'))
    reports={}
    for role in ('local','direction','finite'):
        subset=[r for r in results['task_results'] if r['role']==role]
        stability=next(r for r in subset if r['task_id']=='gaussian1d_stability')
        conclusion=f"{role}: {results['outcomes'][role]}; complete Gaussian stability {stability['gate_status']}; stationary/reacquisition/hold remain authoritative."
        report=knowledge.readout(ROOT,f'gaussian_regression-round5-{role}-v1',conclusion,
            'Same-source three-arm local-v2, exact PR367 direction-blend+local-v2 and exact PR368 strict-finite+local-v2; archived controls contextual and verified by saved-state parity.',
            'Stop this bounded comparison. Preserve rare/broad repairs and original failures; removing finite acceptance does not restore complete Gaussian retention. No sweep, continuation, seed experiment, merge, ordinary qualification or promotion.',study_id=f'gaussian_regression-round5-{role}-study-v1')
        reports[role]=report['record_id'];print(role,report['record_id'],flush=True)
    atomic_json(OUT/'readouts.json',reports)
if __name__=='__main__':main()
