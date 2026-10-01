"""Run the already-written 30-degree/500-update rotation gate on RA15."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from freeze import CONFIG, PACKAGE, ROOT, sha, verify
from run_screen import ENV, GPU_UUID, OLD, parked_owner

ORIGINAL = Path('/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py')


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old,new,1)


def adapted_source():
    source = ORIGINAL.read_text()
    source = replace_once(source,"REPO = '/ml2/hypergan/ParticleGAN-pr155-merge'",f'REPO = {str(PACKAGE)!r}')
    source = replace_once(source,"options = json.load(open(f'{REPO}/configs/100gaussians/e22-noout.json'))",
                          f'options = json.load(open({str(CONFIG)!r}))')
    source = replace_once(source,'            latent, _ = table.sample(n, generator=latent_stream)',
                          '            latent, indices = table.sample(n, generator=latent_stream)')
    source = replace_once(source,'            clean = trainer._generate(model, latent, 0., latent_stream)',
                          '            clean = trainer._generate(model, latent, 0., latent_stream, indices=indices)')
    source = replace_once(source,'torch.cuda.set_device(0); torch.set_num_threads(1); torch.set_num_interop_threads(1)',
        'torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.2, 0); torch.set_num_threads(1); torch.set_num_interop_threads(1)')
    source = replace_once(source,"print('GATE ' + json.dumps(row), flush=True)",
        "print('GATE ' + json.dumps(row), flush=True)\n"
        "        torch.save(trainer.state_dict(), str(owned_output / f'checkpoint-{step:06d}.pt'))\n"
        "        print('MECHANISM ' + json.dumps(dict(step=step, surprise=None if trainer.policy.surprise is None else trainer.policy.surprise.diagnostics(), backend_selection=trainer.policy._feature_selection.state_dict(), reopen_guard=trainer.policy.reopen_guard.state_dict())), flush=True)")
    source = replace_once(source,"    json.dump(verdict, open(args.out + '.verdict.json', 'w'), indent=1)",
        "    assert len(gate_rows) == 3 and len(ok) == 2, 'rotation gate requires both turns'\n"
        "    json.dump(verdict, open(args.out + '.verdict.json', 'w'), indent=1)")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['grid100','rotated100','staggered100'],required=True)
    args = parser.parse_args()
    os.environ.update(ENV)
    integrity = verify()
    output = ROOT / 'moving' / args.task
    assert not output.exists(), 'retain every numerical attempt'
    output.mkdir(parents=True)
    source = adapted_source()
    script = output / 'adapted_runner.py'
    script.write_text(source)
    receipt = dict(status='WAITING_FOR_GPU',task=args.task,start=time.time(),
        command=[sys.executable,*sys.argv],source_integrity_before=integrity,
        original_runner_sha256=sha(ORIGINAL),adapted_runner_sha256=sha(script),
        protocol=dict(turn_every=500,degrees=30,turns=2,steps=1500,seed=1234,draw=20000),
        scorer_changed=False,schedule_changed=False,thresholds_changed=False)
    (output / 'LAUNCH.json').write_text(json.dumps(receipt,indent=2,sort_keys=True) + '\n')
    print(json.dumps(receipt),flush=True)
    with (OLD / 'quality/.serial-phase.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        parked_owner()
        verify()
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        props = torch.cuda.get_device_properties(0)
        actual_uuid = 'GPU-' + str(props.uuid).removeprefix('GPU-').lower()
        assert actual_uuid == GPU_UUID
        receipt['resources'] = dict(physical_gpu=0,device='cuda:0',gpu_uuid=actual_uuid,
                                    gpu_name=props.name,memory_fraction=.2,total_gpu_bytes=props.total_memory)
        receipt.update(status='RUNNING',numerical_started=time.time())
        (output / 'LAUNCH.json').write_text(json.dumps(receipt,indent=2,sort_keys=True) + '\n')
        sys.argv = [str(script),args.task,str(output / 'frames.npz'),'--every','500','--points','4096',
                    '--steps','1500','--gate','--rotate-every','500','--rotate-deg','30']
        exec(compile(source,str(script),'exec'),{'__name__':'__main__','__file__':str(script),'owned_output':output})
        verdict = json.loads((output / 'frames.npz.verdict.json').read_text())
        assert verdict['turns'] == 2 and [row['period_end'] for row in verdict['periods']] == [500,1000,1500]
        receipt.update(status='COMPLETE',quality_status=verdict['status'],verdict=verdict,
                       peak_allocated_gpu_mib=torch.cuda.max_memory_allocated(0)/2**20,
                       peak_reserved_gpu_mib=torch.cuda.max_memory_reserved(0)/2**20,
                       source_integrity_after=verify(),completed=time.time())
        (output / 'COMPLETION.json').write_text(json.dumps(receipt,indent=2,sort_keys=True) + '\n')
    print(json.dumps(dict(event='complete',task=args.task,quality_status=verdict['status'])),flush=True)


if __name__ == '__main__':
    main()
