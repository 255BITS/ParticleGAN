"""Build a compact, byte-auditable handoff without modifying original evidence."""
from pathlib import Path
import gzip,hashlib,json,shutil
SRC=Path('/tmp/pr36-valid-recipe');DST=Path('/tmp/pr36-valid-recipe-handoff')
DST.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
items=[]
def add(relative,data,raw=None):
 path=DST/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 row={'path':relative,'sha256':sha(data),'bytes':len(data)}
 if raw is not None:row.update(uncompressed_sha256=sha(raw),uncompressed_bytes=len(raw))
 items.append(row)
def gz_copy(path,relative):
 data=path.read_bytes();add(relative,data,gzip.decompress(data))
for p in sorted(SRC.glob('*.json')):
 raw=p.read_bytes();add(p.name+'.gz',gzip.compress(raw,mtime=0),raw)
for phase in ('screen','cross','cross_secondary'):
 for p in sorted((SRC/phase).glob('*.json')):
  raw=p.read_bytes();add(f'{phase}/{p.name}.gz',gzip.compress(raw,mtime=0),raw)
 for p in sorted((SRC/phase/'episodes').glob('*.gz')):gz_copy(p,f'{phase}/episodes/{p.name}')
 add(f'{phase}/attempts.md',(SRC/phase/'README.md').read_bytes())
 add(f'{phase}/run.log',(SRC/f'{phase}.log').read_bytes())
source=(SRC/'screen/source.tar.gz').read_bytes()
assert all(gzip.decompress((SRC/phase/'source.tar.gz').read_bytes())==gzip.decompress(source) for phase in ('cross','cross_secondary'))
add('source.tar.gz',source,gzip.decompress(source))
for p in sorted(SRC.glob('*.py')):add('scripts/'+p.name,p.read_bytes())
add('MATRIX.md',(SRC/'README.md').read_text().replace('(index.json)','(index.json.gz)').encode())
add('FINDINGS.md',(SRC/'FINDINGS.md').read_bytes())
README='''# Shared vector recipe handoff

**No all-six winner.** The two best fully checked shared recipes each pass four data toys, with different failures. One solves rare mass at the original 256 particles and 1,200 steps; another solves unequal width. See [findings](FINDINGS.md) and the [full leaderboard matrix](MATRIX.md).

There are **63 actual GAN episodes**, **18 optimizer cards**, **1,512 live observations**, and **434.235 seconds** summed recorded wall time. Fifty-four episodes screen the three hard data toys; nine complete all-six profiles for three finalists. Original architecture, capacity, budgets and behavioral thresholds are unchanged. No extra-budget experiments, seed sweeps, dynamics perturbations or heldout tasks were run.

The formulation is fixed: Rp logistic, b_cap coefficient 3 / kappa 1.25, prior regularization .05 and no particle L2. All cards use one shared recipe across tasks. Live weights need all bounds passing for at least the last five of 24 observations; EMA stays separate. Parent required/image validation is outside this bundle, so this search alone cannot promote a default.

## Retained evidence

- [Combined index](index.json.gz): candidate, original/effective spec, independently recomputed verdicts, live/EMA summary and full episode artifact paths. Each episode also stores complete curves, actions, runtime and actual D/G update counts.
- [Screen plan](screen_plan.json.gz), [best cross plan](best_cross_plan.json.gz), [secondary cross plan](secondary_cross_plan.json.gz), and [finalist selection](cross_selection.json.gz).
- [Duplicate audit](duplicate_audit.json.gz): 90 existing episodes inspected; no new card duplicates an effective prior configuration.
- [Source archive](source.tar.gz): all 57 exact fingerprinted files. Phase [screen protocol](screen/protocol.json.gz), [cross protocol](cross/protocol.json.gz) and [secondary protocol](cross_secondary/protocol.json.gz) have identical source hashes. One source archive is retained instead of three identical uncompressed copies.
- Logs: [screen](screen/run.log), [best cross](cross/run.log), [secondary cross](cross_secondary/run.log). Per-attempt tables are alongside each phase.
- [Original validation](validation.json.gz), [archive inventory](inventory.json), and [portable verification result](verification.json).

Every original JSON is retained either as its exact original gzip or compressed without changing its uncompressed bytes. The inventory records both stored and uncompressed hashes. All failures remain in the bundle.

## Reproduction

Numerical checkout: `a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff`. Original worktree: `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`. Original environment: `/home/mikkel/anaconda3/envs/conceptmod/bin/python`, single-thread CPU. Runtime/build details are in the phase protocols. Restore the source archive over that checkout for the exact numerical sources.

Each phase used `python -u -m benchmarks.transfer_suite.solvability_search --plan PLAN --output NEW_DIRECTORY`. The plans above are the exact inputs; the only changes between phases are the candidate/task lists. The original utility is in the source archive. [make_plan.py](scripts/make_plan.py), [summarize.py](scripts/summarize.py), [validate.py](scripts/validate.py), and [package.py](scripts/package.py) retain their original absolute paths. Use those paths or adapt a separate copy; never write reruns into retained evidence. No training is needed to verify this archive.

Run `python verify.py` to check all gzip roundtrips, stored hashes, source-member hashes, full episode invariants, observation positions, original thresholds, update counts and live/EMA/sustained scoring. Run `sha256sum -c SHA256SUMS` for the complete stored-file inventory.
'''
add('README.md',README.encode())
metadata={'source_commit':'a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff','episodes':63,'live_observations':1512,'files':items}
(DST/'inventory.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
print(DST)
