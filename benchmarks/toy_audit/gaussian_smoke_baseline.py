"""Zero-update, fixed-checkpoint confirmation of historical Gaussian acquisition.

This source-bound diagnostic does not qualify the new ordinary smoke task and
never regrades the historical five-terminal acquisition verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.gaussian_tasks import bounds, build, training_digest
from experiments.forge.sources import inspect_source, runtime_manifest
from experiments.forge.state import state_digest
from .gaussian1d_quality import score_samples
from .reproducibility import reproducible_execution

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/forge/gaussian-smoke-tier-split/protocol.json'


@reproducible_execution
def execute(output, *, device):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('baseline confirmation requires CUDA; no CPU fallback')
    protocol = json.loads(PROTOCOL.read_text())
    baseline = Path(protocol['baseline']['raw_directory'])
    for name, expected in protocol['baseline']['artifacts'].items():
        if file_hash(baseline / name) != expected:
            raise ValueError('historical artifact differs: ' + name)
    for name, expected in protocol['implementation_sources'].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError('frozen diagnostic source differs: ' + name)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    candidate_path=ROOT/protocol['candidate_path']
    candidate=json.loads(candidate_path.read_text())
    task=json.loads((ROOT/'configs/forge/tasks/gaussian1d_smoke.json').read_text())
    context,trainer,spec=build(dict(candidate=candidate,protocol={'seed':0}),task,device)
    saved=torch.load(baseline/'state.pt',weights_only=True,map_location='cpu')
    context.load_state_dict(saved)
    restored=state_digest(context.state_dict())
    if restored != state_digest(saved) or trainer.completed_steps != 1000:
        raise ValueError('historical fixed1000checkpoint did not restore exactly')
    before=training_digest(context)
    stream=context.streams.generator('eval',component='live',purpose='smoke_confirmation')
    started=time.monotonic()
    samples=trainer.sample(4096,generator=stream,output_noise=False).detach().cpu()
    metrics=score_samples(samples,spec,1000)
    after=training_digest(context)
    if before!=after or trainer.completed_steps!=1000:
        raise ValueError('confirmation changed historical training state')
    primary=json.loads((baseline/'curve.json').read_text())[-1]
    original=json.loads((baseline/'receipt.json').read_text())
    torch.save(samples,output/'confirmation-samples.pt')
    torch.save(context.state_dict(),output/'confirmation-state.pt')
    source=inspect_source(ROOT,extra_paths=(str(PROTOCOL.relative_to(ROOT)),str(candidate_path.relative_to(ROOT))))
    atomic_json(output/'source.json',source)
    result=dict(schema_version=1,id=protocol['id'],scope=protocol['scope'],qualification_input=False,
                protocol_sha256=file_hash(PROTOCOL),new_training_updates=0,completed_steps=1000,
                candidate_path=protocol['candidate_path'],recipe=trainer.recipe.to_dict(),prior=context.prior_config,
                primary=primary,confirmation=metrics,confirmed=not bounds(primary) and not bounds(metrics),
                original_acquisition_verdict=original['verdict'],original_verdict_unchanged=True,
                original_source=json.loads((baseline/'source.json').read_text()),
                original_artifacts=protocol['baseline']['artifacts'],restored_exactly=True,
                restored_state_sha256=restored,training_state_unchanged=before==after,
                independent_stream='eval/live/smoke_confirmation',final_rng=context.streams.manifest(),
                runtime=runtime_manifest(),device=device,gpu=torch.cuda.get_device_name(device),
                diagnostic_loop_seconds=time.monotonic()-started,
                artifacts={name:file_hash(output/name) for name in ['confirmation-samples.pt','confirmation-state.pt','source.json']})
    atomic_json(output/'results.json',result)
    print(json.dumps({key:result[key] for key in ['confirmed','primary','confirmation','new_training_updates','diagnostic_loop_seconds']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();execute(args.output,device=args.device)


if __name__=='__main__':main()
