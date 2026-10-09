"""Close the exact frozen diagnostic subscriptions with summaries-only readout."""
from pathlib import Path
import argparse,os,sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.knowledge import readout
from experiments.forge.contracts import read_json,atomic_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,required=True)
    args=parser.parse_args();os.environ['PARTICLEGAN_FORGE_QUEUE']=str(args.queue_root)
    directory=Path(__file__).parent;results=read_json(directory/'results.json')
    assert results['complete'] and results['campaign']['reserved_seconds']==0
    arms=results['comparison']
    native={label:next(t for t in arm['tasks'] if t['task_id']=='grid100') for label,arm in arms.items()}
    comparison=(f"Role-motion candidate versus exact winner in identical source/runtime: "
        f"candidate {arms['candidate']['outcomes']}, control {arms['control']['outcomes']}. "
        f"Native precision {native['control']['metrics']['precision']} -> {native['candidate']['metrics']['precision']}. "
        'Archived winner and transport-v2 saved probes are context, not another matched arm. Full sustained gates and own-smoke dependencies are retained.')
    rows=[]
    for label in ['candidate','control']:
        study=f'role-motion-{label}-round4-v1';arm=arms[label]
        conclusion=(f"All admitted runnable full-budget tasks complete; {arm['outcomes']}. "
            f"Native precision {native[label]['metrics']['precision']} with frozen forecast>=.48/falsifier<.30. "
            'Controller counters and independent saved-center Jacobians are separate from actual served-law gates. Exact blocked two-pole applicability remains declared.')
        action=('Stop this exact generator/prior RMS-balance revision; retain original winner/default qualification. '
                'Inspect saved early role gradients and deformation before any separately frozen substantive successor. No seed repeat, sweep, extra arm, continuation or default adoption.'
                if label=='candidate' else 'Close the matched control subscription, preserve exact receipts and original qualification; no unchanged rerun.')
        record=readout(ROOT,arm['candidate_id'],conclusion,comparison,action,study_id=study)
        rows.append(dict(label=label,study_id=study,record_id=record['record_id'],lifecycle=record['lifecycle'],decision_outcomes=record['decision_outcomes']))
    atomic_json(directory/'study-readouts.json',dict(schema_version=1,items=rows,qualification_input=False,optimizer_updates_added=0,sampling_draws_added=0))
    print([(r['label'],r['decision_outcomes'][0]['outcome']) for r in rows],flush=True)

if __name__=='__main__':main()
