"""Recover the compact receipt solely from the completed saved CUDA draw.

The original diagnostic stopped during receipt serialization. This reducer adds
no training, sampling, gradient calculations or fresh model evaluation.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.gaussian_tasks import bounds
from experiments.forge.state import state_digest
from benchmarks.toy_audit.gaussian1d_quality import score_samples

DEST=Path(__file__).resolve().parent
RAW=ROOT/'runs/api/gaussian-smoke-tier-split-v1/confirmation'


def training_state(state):
    result=deepcopy(state)
    result.pop('streams');result['trainer'].pop('streams',None)
    return result


def main():
    protocol=json.loads((DEST/'protocol.json').read_text())
    baseline=Path(protocol['baseline']['raw_directory'])
    for name,expected in protocol['baseline']['artifacts'].items():
        assert file_hash(baseline/name)==expected,name
    prior=torch.load(baseline/'state.pt',weights_only=True,map_location='cpu')
    after=torch.load(RAW/'confirmation-state.pt',weights_only=True,map_location='cpu')
    original_digest=state_digest(training_state(prior));confirmed_digest=state_digest(training_state(after))
    assert original_digest==confirmed_digest
    assert prior['trainer']['completed_steps']==after['trainer']['completed_steps']==1000
    samples=torch.load(RAW/'confirmation-samples.pt',weights_only=True,map_location='cpu')
    task=json.loads((ROOT/'configs/forge/tasks/gaussian1d_smoke.json').read_text())
    metrics=score_samples(samples,task['execution']['host_definition'],1000)
    primary=json.loads((baseline/'curve.json').read_text())[-1]['metrics']
    original=json.loads((baseline/'receipt.json').read_text())
    streams=after['streams']['manifest']['bindings']
    confirms=[v for v in streams.values() if v['family']=='eval' and v['purpose']=='smoke_confirmation']
    assert len(confirms)==1 and confirms[0]['component']=='live' and confirms[0]['device']=='cuda:0'
    result=dict(schema_version=1,id=protocol['id'],scope=protocol['scope'],qualification_input=False,
                original_acquisition_verdict=original['full_verdict'],original_verdict_unchanged=True,
                new_training_updates=0,new_confirmation_draws=1,confirmation_sample_count=len(samples),
                primary=primary,confirmation=metrics,confirmed=not bounds(primary) and not bounds(metrics),
                training_state_unchanged=True,training_state_sha256=original_digest,
                independent_stream='eval/live/smoke_confirmation',protocol_sha256=file_hash(DEST/'protocol.json'),
                original_source=json.loads((baseline/'source.json').read_text()),original_artifacts=protocol['baseline']['artifacts'],
                execution_source=json.loads((RAW/'source.json').read_text()),
                serialization_recovery=dict(original_error="KeyError: 'verdict' after the completed saved CUDA draw",
                    retries=0,new_training_updates=0,new_sampling_draws=0,scored_saved_tensors=True,
                    original_stdout_sha256=file_hash(RAW.parent/'baseline.log'),publisher_sha256=file_hash(Path(__file__))),
                artifacts={name:file_hash(RAW/name) for name in ['confirmation-samples.pt','confirmation-state.pt','source.json']},
                actual_training_gif='../tier1-prior-smoke/mog100-n256-gaussian1d_acquisition.gif')
    atomic_json(RAW/'results.json',result)
    # Source manifest is retained compactly as origin/digest; bulk per-file manifest stays raw.
    compact=deepcopy(result)
    compact['execution_source']={k:result['execution_source'][k] for k in ['origin_commit','digest']}
    compact['original_source']={k:result['original_source'][k] for k in ['origin_commit','digest']}
    atomic_json(DEST/'results.json',compact)
    print(json.dumps(dict(confirmed=result['confirmed'],primary=primary,confirmation=metrics,training_updates=0)),flush=True)


if __name__=='__main__':main()
