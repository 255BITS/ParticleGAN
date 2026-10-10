"""Conclude exactly the two completed round-two studies using safe readout API."""
from pathlib import Path
import argparse, os, sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.knowledge import readout
from experiments.forge.contracts import read_json, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,required=True)
    args=parser.parse_args();os.environ['PARTICLEGAN_FORGE_QUEUE']=str(args.queue_root)
    directory=Path(__file__).parent;results=read_json(directory/'results.json')
    assert results['complete'] and results['campaign']['reserved_seconds']==0
    comparison=('Four matched executed tasks yield2PASS2FAIL in each arm. Combined successor versus exact winner: '
                'Gaussian retention69/72 versus2/72 and shifted hold23/24 versus0/24; native precision.50133 versus.24072, '
                'covariance trace bias.901832611 versus.371384942 and median local variance ratio4.456226400 versus2.976158908. '
                'No comparison with the archived hydraulic-v1 source can isolate the added penalty effect. '
                'The control study parent is an untrained admission reference with no comparable measured evidence or qualification credit.')
    rows=[]
    for label,study in [('candidate','hydraulic-deformation-candidate-round2-v1'),('control','hydraulic-deformation-control-round2-v1')]:
        candidate=results['comparison'][label]['candidate_id']
        conclusion=('Combined travel and secant deformation candidate retains broad-vector PASS but strict Gaussian and native FAIL; '
                    'native precision.50133 misses.55 prediction, covariance bias.90183 misses.70 explanatory forecast, '
                    'and local variance ratio4.45623 exceeds the matched winner. Two-pole remains unsupported/unmeasured.'
                    if label=='candidate' else
                    'The exact winner diagnostic control completes all five frozen tasks: three PASS and two FAIL. '
                    'It reproduces original Gaussian/native endpoints; this is no new repair, qualification or parent comparison.')
        action=('Stop this exact combined revision as a global repair; retain winner/default and all original qualifications. '
                'Before a separately frozen substantive idea, distinguish local training density from global variance normalization '
                'and measure deformation gradient work. No coefficient sweep, seed repeats, transfer or extra training follows.'
                if label=='candidate' else
                'Close this exact matched-control study and retain its receipts. Preserve archived winner selection and qualifications; '
                'no parent training or further control rerun follows.')
        record=readout(ROOT,candidate,conclusion,comparison,action,study_id=study)
        rows.append(dict(label=label,study_id=study,record_id=record['record_id'],
                         lifecycle=record['lifecycle'],decision_outcomes=record['decision_outcomes']))
    atomic_json(directory/'study-readouts.json',dict(schema_version=1,items=rows,qualification_input=False,
        optimizer_updates_added=0,random_sampling_draws_added=0))
    print([(r['label'],r['record_id'],r['decision_outcomes'][0]['outcome']) for r in rows],flush=True)


if __name__=='__main__':main()
