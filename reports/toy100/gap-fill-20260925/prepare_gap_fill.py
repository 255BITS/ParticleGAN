import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/ml2/hypergan/gan-attempts')
OUT = ROOT / 'gap-fill-20260925'
DOC = Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/gap-fill-20260925')
KD = ROOT / 'claude-pool-20260925T063704Z/critic_both_arms/20260925T125747Z-3385271/repo/reports/toy100/critic-both-arms-3385271'
GD = ROOT / 'claude-pool-20260925T063704Z/critic_both_arms/20260925T101301Z-3111873/repo/reports/toy100/critic-both-arms-3111873'
PD = ROOT / 'claude-pool-20260925T063704Z/critic_both_arms/20260925T113805Z-3248837/repo/reports/toy100/critic-both-arms-3248837'
RD = ROOT / 'claude-pool-20260925T062531Z/native_acquisition/20260925T131236Z-3408013/repo/reports/toy100/native-acquisition-3408013'
REPO = ROOT / 'claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda'
FIX = Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures')
EXTRA = ROOT / 'claude-pool-20260925T024114Z/surviving_moment_weight/20260925T042840Z-2529991/repo/reports/toy100/a2-qualify/initialization-fixtures'
PY = '/tmp/pr38-default-env/bin/python'
OUT.mkdir(exist_ok=False)
DOC.mkdir(exist_ok=False)
for d in ['logs', 'out']:
    (OUT / d).mkdir()
sources = []
def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    sources.append(dict(source=str(src), saved=str(dst.relative_to(DOC)), sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))

candidates = {'k3p': KD/'k3p', 'k3g': GD/'k3g', 'k3': PD/'p1', 'rg5-bcap': RD/'cands/rg5-bcap', 'rg5-a2': RD/'cands/rg5-a2'}
for name, src in candidates.items():
    dst = DOC/'sources'/name
    for n in ['config.json', 'mechanism.py', 'latent.py', 'response.py']:
        copy(src/n, dst/n)
    copy(RD/'probe_g.py' if name.startswith('rg5') else KD/'k3p/probe.py', dst/'probe.py')
    copy(KD/'k3p/checkpoint.py', dst/'checkpoint.py')
    ring = GD/'harness/ring' if name == 'k3g' else KD/'ringsrc'
    for n in ['hold.py', 'shift.py', 'shift_frozen.py', 'convergence_gate.py']:
        copy(ring/n, dst/n)
    if name.startswith('rg5'):
        copy(RD/'shiftsrc/shift.py', dst/'shift.py')
    driver = GD/'harness/native100.py' if name == 'k3g' else RD/'native100s.py' if name.startswith('rg5') else ROOT/'session-findings/native100.py'
    copy(driver, dst/'native100.py')

focused = sorted(p.name for p in FIX.iterdir() if (p/'initial-values.pt').exists())
extra = ['vector_anisotropic', 'vector_overlap', 'vector_spiral']
fixtures = {}
for task in focused + extra:
    src = (EXTRA if task in extra else FIX)/task/'initial-values.pt'
    fixtures[task] = dict(path=str(src), sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    if task in extra:
        init = json.loads((src.parent/'result.json').read_text())
        assert init['status'] == 'INITIALIZATION_CAPTURED' and init['proof']['adam_calls'] == 0

jobs = []
def job(cand, kind, task, floors=None):
    tag = f'{cand}-{kind}-{task}' + (f'-{floors[0]}-{floors[1]}' if floors else '')
    src = DOC/'sources'/cand
    repo = RD/'cudarepo' if cand.startswith('rg5') else REPO
    cmd = [PY, '-u', str(src/('native100.py' if kind=='native' else 'probe.py' if kind=='toy' else kind+'.py'))]
    cmd += ['--repo',str(repo),'--task',task,'--output',str(OUT/'out'/tag)]
    if kind == 'native':
        cmd += ['--candidate',str(src)]
        if cand.startswith('rg5'):
            cmd += ['--stop','0']
    else:
        cmd += ['--config',str(src/'config.json'),'--backend','cuda','--initial-state',fixtures[task]['path']]
    if floors:
        cmd += ['--network-floor',str(floors[0]),'--prior-floor',str(floors[1])]
    jobs.append(dict(id=tag,candidate=cand,kind=kind,task=task,command=cmd,cwd=str(repo),output=str(OUT/'out'/tag),log=str(OUT/'logs'/f'{tag}.log'),floors=floors))

# Start the longest and most discriminating missing problems first.
for cand, tasks in [('k3p',['rotated100','staggered100']),('k3g',['grid100','rotated100']),('rg5-bcap',['rotated100'])]:
    for task in tasks: job(cand,'native',task)
for task in ['mode_hold','vector_unequal_mass','img_stripes2']: job('rg5-bcap','toy',task)
for cand in ['k3g','rg5-bcap']:
    for kind in ['hold','shift']: job(cand,kind,'mode_hold',[.01,.05])
job('rg5-a2','shift_frozen','mode_hold',[.1,.1])
for task in focused + extra:
    if task not in ['mode_hold','vector_unequal_mass','img_stripes2']: job('k3p','toy',task)
for cand in ['k3g','k3']:
    for task in extra: job(cand,'toy',task)
for task in focused + extra:
    if task not in ['mode_hold','vector_unequal_mass','img_stripes2']: job('rg5-bcap','toy',task)
assert len(jobs) == len({j['id'] for j in jobs})
sources = list({row['saved']: row for row in sources}.values())
manifest = dict(protocol='Gap filling only; no seed sweeps. Native seed1234, full7000; transfer declared seeds and original CPU initialization fixtures; CUDA training only. Fixed candidate bundles, base floors except existing RG5+A2 .1/.1 matched frozen control.',sources=sources,fixtures=fixtures,jobs=jobs)
(DOC/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(jobs=len(jobs),output=str(OUT),sources=str(DOC))))
