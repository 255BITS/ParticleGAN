from pathlib import Path
import gzip,hashlib,json
SRC=Path('/tmp/pr36-valid-local-d');DST=Path('/tmp/pr36-valid-local-d-handoff');DST.mkdir(exist_ok=True)
items=[];sha=lambda b:hashlib.sha256(b).hexdigest()
def add(name,data,raw=None):
 p=DST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);r={'path':name,'sha256':sha(data),'bytes':len(data)}
 if raw is not None:r.update(uncompressed_sha256=sha(raw),uncompressed_bytes=len(raw))
 items.append(r)
for p in sorted(SRC.glob('*.json')):
 raw=p.read_bytes();add(p.name+'.gz',gzip.compress(raw,mtime=0),raw)
for p in sorted((SRC/'screen').glob('*.json')):
 raw=p.read_bytes();add('screen/'+p.name+'.gz',gzip.compress(raw,mtime=0),raw)
for p in sorted((SRC/'screen/episodes').glob('*.gz')):add('screen/episodes/'+p.name,p.read_bytes(),gzip.decompress(p.read_bytes()))
source=(SRC/'screen/source.tar.gz').read_bytes();add('source.tar.gz',source,gzip.decompress(source));add('screen/run.log',(SRC/'screen.log').read_bytes())
for p in sorted(SRC.glob('*.py')):add('scripts/'+p.name,p.read_bytes())
for name in ('MATRIX.md','FINDINGS.md'):add(name,(SRC/name).read_text().replace('(index.json)','(index.json.gz)').replace('(architecture_checks.json)','(architecture_checks.json.gz)').encode())
README='''# Local Gaussian-feature discriminator handoff

**No sustained rare-mode winner in eight declared architectures.** The best final result passes every bound except minimum component eigenvalue ratio (.131176 < .15). No full-six followup was run. See [findings](FINDINGS.md) and the [parameter-count leaderboard](MATRIX.md).

Exactly 8 GAN episodes, 192 fixed live observations and 75.793955 seconds summed recorded wall time. All cards retain original G, 256 particles, batch128, 1,200 steps, Rp logistic, b_cap3/kappa1.25, prior regularization .05, no particle L2, Adam(0,.99), G/D/prior LRs .001/.0015/.01, cosine and 1:1 updates. EMA is separate. Thresholds and final-five-of24 live requirement are unchanged.

The architecture replaces the original Fourier feature map with raw coordinates plus Gaussian responses at generic standard-normal centers. Declared octave widths are fixed or learned along with centers through only the existing D objective. No target data, labels, centers, moments, normalization, auxiliary loss or seed selection enters construction. Every episode explicitly records research_discriminator metadata and D parameter count.

## Retained evidence

- [Combined index](index.json.gz), [frozen plan](screen/plan.json.gz), [stop decision](selection.json.gz), and [log](screen/run.log).
- [Source archive](source.tar.gz): all 59 numerical dependency files, including `benchmarks/transfer_suite/local_critic_research.py`. [Protocol](screen/protocol.json.gz) records source hashes and exact driver/checker hashes.
- [Driver](scripts/run.py), [static architecture checker](scripts/check_architectures.py), [check results](architecture_checks.json.gz), and [independent validation](validation.json.gz).
- [Inventory](inventory.json) contains stored and original uncompressed hashes. [Verification](verification.json) records the portable audit.

Source base is commit `981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45` plus the exact archived research module. Its SHA256 is `18b1a4f5c7c16bd59599cf1a1a3810cc61dbd63059e897a47bfdd43aab2ce0ff`. No reusable-code/default change is proposed from this unsuccessful screen.

## Reproduction

Restore the exact archive over the base checkout. Original worktree `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`; Python `/home/mikkel/anaconda3/envs/conceptmod/bin/python`; CPU thread1. Original command: `python -u run.py --phase screen`. Driver paths are retained exactly; use those paths or adapt a separate copy without overwriting retained evidence. The driver serially replaces only the discriminator constructor and restores it after each episode. It does not alter the generator, targets, loss or metrics.

All JSON preserves its uncompressed original bytes; original episode gzip bytes are also preserved. Run `python verify.py` without training to verify source hashes, gzip roundtrips, exact architecture-only specs, full curves and independent live/EMA verdicts. `sha256sum -c SHA256SUMS` checks the stored bundle.
'''
add('README.md',README.encode());(DST/'inventory.json').write_text(json.dumps({'episodes':8,'live_observations':192,'files':items},indent=2,sort_keys=True)+'\n');print(DST)
