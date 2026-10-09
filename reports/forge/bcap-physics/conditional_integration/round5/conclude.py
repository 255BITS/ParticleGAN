"""Conclude each frozen study; use summaries-only memory refreshes."""
from pathlib import Path
import json
from experiments.forge.contracts import read_json,atomic_json
from experiments.forge.knowledge import readout
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
PREFIX='conditional-integration-round5'

def main():
    result=read_json(OUT/'results.json');records=[]
    descriptions={
      'winner':'Exact winner reproduces 3PASS4FAIL; both original conditional identities, rare mixture and Gaussian retention fail.',
      'direction':'Exact direction blend retains all three conditional variant PASSes:5PASS2FAIL; rare mixture and Gaussian retention fail.',
      'transport':'Exact local-v2 retains rare/broad and mid-scale PASSes:4PASS3FAIL; both conditional identities and Gaussian retention fail.',
      'both':'One global direction+local-v2 recipe retains all three conditional variants and rare/broad density repairs:6PASS1FAIL. Gaussian retention fails (6/72 stationary,2/24 shift hold,deadlineFAIL), despite passing finalKS.'}
    for role in ('winner','direction','transport','both'):
        record=readout(ROOT,f'{PREFIX}-{role}-v1',descriptions[role],
            'Four matched arms on one frozen source/runtime and explicit marginal-consumer variants; no new trajectory paired supervision. Exact singleton/inactive numerical parity and all consumed streams verified.',
            'Stop this finite campaign. Retain BOTH as scoped measured conditional/density union, preserve Gaussian failure; no global replacement, ordinary qualification, default adoption, seed study, tuning or continuation.',
            study_id=f'{PREFIX}-{role}-study-v1')
        records.append(dict(role=role,record_id=record['record_id'],path=f'reports/forge/records/{record["record_id"]}.json'))
        print(json.dumps(dict(role=role,record_id=record['record_id'],phase='concluded')),flush=True)
    atomic_json(OUT/'readouts.json',dict(schema_version=1,qualification_input=False,records=records))
if __name__=='__main__':main()
