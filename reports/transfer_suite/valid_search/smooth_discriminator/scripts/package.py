from pathlib import Path
import gzip,hashlib,json
SRC=Path('/tmp/pr36-valid-smooth-d');DST=Path('/tmp/pr36-valid-smooth-d-handoff');REPO=Path('/ml2/hypergan/ParticleGAN-pr36-valid-recipe')
DST.mkdir(exist_ok=True);items=[];sha=lambda b:hashlib.sha256(b).hexdigest()
def add(name,data,raw=None):
 p=DST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);r={'path':name,'sha256':sha(data),'bytes':len(data)}
 if raw is not None:r.update(uncompressed_sha256=sha(raw),uncompressed_bytes=len(raw))
 items.append(r)
for p in sorted(SRC.glob('*.json')):
 raw=p.read_bytes();add(p.name+'.gz',gzip.compress(raw,mtime=0),raw)
for phase in ('screen','cross'):
 for p in sorted((SRC/phase).glob('*.json')):
  raw=p.read_bytes();add(f'{phase}/{p.name}.gz',gzip.compress(raw,mtime=0),raw)
 for p in sorted((SRC/phase/'episodes').glob('*.gz')):add(f'{phase}/episodes/{p.name}',p.read_bytes(),gzip.decompress(p.read_bytes()))
 add(f'{phase}/attempts.md',(SRC/phase/'README.md').read_bytes());add(f'{phase}/run.log',(SRC/f'{phase}.log').read_bytes())
source=(SRC/'screen/source.tar.gz').read_bytes();assert gzip.decompress(source)==gzip.decompress((SRC/'cross/source.tar.gz').read_bytes());add('source.tar.gz',source,gzip.decompress(source))
for p in sorted(SRC.glob('*.py')):add('scripts/'+p.name,p.read_bytes())
for name in ('MATRIX.md','FINDINGS.md'):add(name,(SRC/name).read_text().replace('(index.json)','(index.json.gz)').encode())
add('tests.log',(SRC/'tests.log').read_bytes());add('reusable/test_smooth_critic_research.py',(REPO/'tests/test_smooth_critic_research.py').read_bytes());add('reusable/smooth_critic_research.md',(REPO/'benchmarks/transfer_suite/smooth_critic_research.md').read_bytes())
README='''# Smooth discriminator handoff

**An architecture-only unequal-width PASS for the original recipe.** Axis Fourier + Softplus(beta5) sustains seven final observations at the original1,200 steps. No architecture passes all six data toys; each full finalist profile passes3/6. See [findings](FINDINGS.md) and the [complete matrix](MATRIX.md).

This separate bounded study contains30 GAN episodes,720 fixed live observations and193.454 seconds summed recorded wall time. Eight declared D architectures screen three hard toys, then two finalists receive the other three. Original beta99 optimizer, LRs, loss, penalty, prior regularizer, G, particles, batch, budgets and thresholds are unchanged. EMA is separate. No target-derived features or seed sweeps.

## Evidence

- [Combined index](index.json.gz), [screen plan](screen/plan.json.gz), [cross plan](cross/plan.json.gz), [finalist selection](selection.json.gz). Every episode includes full live/EMA curves, actions, actual updates, candidate architecture, original/effective specs and verdicts.
- [Source archive](source.tar.gz):58 exact numerical files, including the research module. [Screen protocol](screen/protocol.json.gz) and [cross protocol](cross/protocol.json.gz) have identical hashes. Only one identical source copy is retained.
- [Driver](scripts/run.py), [screen log](screen/run.log), [cross log](cross/run.log), [static architecture checks](architecture_checks.json.gz), [validation](validation.json.gz).
- [Reusable module notes](reusable/smooth_critic_research.md), [focused tests](reusable/test_smooth_critic_research.py), [10-test result](tests.log). The numerical module lives in the source archive at `benchmarks/transfer_suite/smooth_critic_research.py`.
- [Inventory](inventory.json) records compressed/file and original uncompressed SHA256 hashes. [Portable verification](verification.json) checks all source hashes and complete numerical verdicts.

## Reproduction

Base checkout: `a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff`, plus the exact archived research module. Original worktree `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`; environment `/home/mikkel/anaconda3/envs/conceptmod/bin/python`; CPU thread1. Research driver temporarily replaces only the vector runner's discriminator constructor, restores it after each serial episode, and records the replacement architecture explicitly. It never patches the GAN loss, targets, metrics or optimizer.

Original commands were `python -u run.py --phase screen` and `python -u run.py --phase cross --names axis_softplus5 axis_tanh`. The driver retains original absolute worktree/output paths; reproduce there or adapt a separate copy, never overwrite retained evidence. Restore exact source bytes from the archive. Runtime/build fingerprint is in each phase protocol.

Run `python verify.py` to verify this bundle without training. `sha256sum -c SHA256SUMS` checks every stored file. Original JSON bytes and all failed attempts are preserved.
'''
add('README.md',README.encode())
(DST/'inventory.json').write_text(json.dumps({'episodes':30,'live_observations':720,'source_commit_base':'a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff','files':items},indent=2,sort_keys=True)+'\n');print(DST)
