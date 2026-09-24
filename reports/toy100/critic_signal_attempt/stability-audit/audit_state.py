"""Read-only evidence comparison for selected H and its baseline continuation."""
from pathlib import Path
import gzip
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parent
SRC = Path('/ml2/hypergan/gan-attempts/batch-20260924T153037Z/critic_signal/20260924T153037Z-1298555/repo')
EVIDENCE = SRC/'reports/toy100/critic_signal_attempt'
CANDIDATE = EVIDENCE/'batch-h/h_n05r06_mixup_c0p01_lr15'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def differences(a, b, prefix=''):
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and torch.equal(a,b) else [prefix]
    if isinstance(a, torch.Generator):
        return differences(a.get_state(), b.get_state(), prefix+'.rng')
    if isinstance(a, dict):
        out = [prefix+'.keys'] if a.keys()!=b.keys() else []
        for key in a.keys() & b.keys():
            out += differences(a[key],b[key],prefix+'.'+str(key))
        return out
    if isinstance(a,(list,tuple)):
        if len(a)!=len(b):
            return [prefix+'.length']
        return [v for i,(x,y) in enumerate(zip(a,b)) for v in differences(x,y,prefix+'.'+str(i))]
    return [] if a==b else [prefix]

def main():
    torch.set_num_threads(1)
    identity=json.loads((EVIDENCE/'source-identity.json').read_text())
    manifest=json.loads((EVIDENCE/'batch-h/manifest.json').read_text())
    mismatches=[name for name,digest in manifest['source_sha256'].items()
                if sha(ROOT/'repo'/name)!=digest]
    saved=torch.load(EVIDENCE/'h-own-hold/final-state.pt',weights_only=False,map_location='cpu')
    replay=torch.load(ROOT/'audit-control-hold/final-state.pt',weights_only=False,map_location='cpu')
    cold=torch.load(CANDIDATE/'mode_hold/final-state.pt',weights_only=False,map_location='cpu')
    metrics_saved=json.loads((EVIDENCE/'h-own-hold/metrics.json').read_text())
    metrics_replay=json.loads((ROOT/'audit-control-hold/metrics.json').read_text())
    receipts=[]
    for path in (CANDIDATE/'mode_hold/signal-policy.json.gz',
                 EVIDENCE/'h-own-hold/optimizer-receipt.json.gz'):
        r=json.loads(gzip.decompress(path.read_bytes()))
        rates={role:sorted({g['lr'] for u in r['updates'] for g in u['groups'] if g['role']==role})
               for role in ('g','d','prior')}
        receipts.append(dict(path=str(path),updates=len(r['updates']),rates=rates,
                             options=r['signal_options'],noise=r['noise'],
                             signal_work=r['signal_work'],objective=r['generator_objective']))
    report=dict(
        archive_sha256=sha(EVIDENCE/'batch-h/source.tar.gz'),
        archive_hash_matches=sha(EVIDENCE/'batch-h/source.tar.gz')==identity['best_cold_source_archive_sha256'],
        archived_source_file_count=len(manifest['source_sha256']),archived_source_mismatches=mismatches,
        cold_checkpoint_sha256=sha(CANDIDATE/'mode_hold/final-state.pt'),
        cold_checkpoint_hash_matches=sha(CANDIDATE/'mode_hold/final-state.pt')==identity['best_cold_checkpoint_sha256'],
        hold_state_differences=differences(saved,replay),
        dense_metrics_bitwise_identical=metrics_saved==metrics_replay,
        cold_step=cold['step'],cold_prior_shape=list(cold['models']['prior']['z'].shape),
        cold_optimizer_step_sets={k:sorted({int(v['step']) for v in opt['state'].values()})
                                  for k,opt in cold['optimizers'].items()},
        cold_noise=dict(output_std=cold['noise_policy']['output_std'],
                        output_sigma=cold['noise_policy']['output_sigma'],
                        total_steps=cold['noise_policy']['total_steps'],
                        output_noise_warmup=cold['noise_policy']['output_noise_warmup'],
                        input_sigma_nominal=cold['noise_policy']['input_sigma']),
        receipts=receipts,
        hold_first_failure=next(p for p in metrics_saved if p['modes']!=8 or p['hq']<.9),
        hold_passing_checks=sum(p['modes']==8 and p['hq']>=.9 for p in metrics_saved),
        hold_final=metrics_saved[-1],
    )
    (ROOT/'audit-findings.json').write_text(json.dumps(report,indent=2)+'\n')
    compact={k:v for k,v in report.items() if k not in ('receipts','hold_first_failure','hold_final')}
    compact.update(first_failure_step=report['hold_first_failure']['step'],
                   first_failure_modes=report['hold_first_failure']['modes'],
                   first_failure_hq=report['hold_first_failure']['hq'])
    print(json.dumps(compact,indent=2))
    assert not mismatches and not report['hold_state_differences']
    assert report['dense_metrics_bitwise_identical']

if __name__=='__main__':
    main()
