"""Close the two bounded round-three studies through summaries-only readout."""
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
    width=read_json(directory/'native-width.json')
    controller=read_json(directory/'controller-diagnostics.json')
    native={label:next(t for t in arm['tasks'] if t['task_id']=='grid100') for label,arm in results['comparison'].items()}
    comparison=(f"Combined successor versus exact winner in identical source/runtime: candidate {results['comparison']['candidate']['outcomes']}, control {results['comparison']['control']['outcomes']}. "
        f"Native precision {native['candidate']['metrics']['precision']} versus {native['control']['metrics']['precision']}. "
        'The predecessor is unsupported as an admission control; archived v1/v2 results are contextual, not incremental causal evidence. '
        'The control study parent is an untrained admission reference, not a third trained arm or qualification input.')
    rows=[]
    for label in ['candidate','control']:
        study=f'hydraulic-local-shape-{label}-round3-v1';arm=results['comparison'][label]
        conclusion=(f"Completed all admitted full-budget runnable tasks. Exact outcomes {arm['outcomes']}; native precision {native[label]['metrics']['precision']} against frozen prediction.60/falsifier.48. "
            +('Finite sampled shape/travel bounds and rounded mean projection are checked in controller-diagnostics.json; complete task gates remain decisive. Own smoke-dependent stability is retained under its actual dependency status.' if label=='candidate' else 'This is the matched exact-winner diagnostic measurement, not a repair or comparison against its untrained parent.'))
        action=('Stop this exact local graph-capacity/mean-separation revision; no sweep, seed repeat, extra arm, continuation or adoption. '
            'Retain the winner/default and original qualification. Inspect neighborhood resolution and useful mean/mass transport using saved evidence before any separately frozen substantive successor.' if label=='candidate' else
            'Close the matched control subscription and retain exact receipts. Preserve original selection and qualification; no unchanged rerun follows.')
        record=readout(ROOT,arm['candidate_id'],conclusion,comparison,action,study_id=study)
        rows.append(dict(label=label,study_id=study,record_id=record['record_id'],lifecycle=record['lifecycle'],decision_outcomes=record['decision_outcomes']))
    atomic_json(directory/'study-readouts.json',dict(schema_version=1,items=rows,qualification_input=False,optimizer_updates_added=0,random_sampling_draws_added=0))
    print([(r['label'],r['decision_outcomes'][0]['outcome']) for r in rows],flush=True)


if __name__=='__main__':main()
