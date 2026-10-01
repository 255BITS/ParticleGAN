"""Prepare, verify or explicitly launch the frozen E22 versus Atlas capture.

Preparation is CPU/source-only. --parity and --full require root launch under
the shared original GPU mutex; all numerical workers retain every attempt.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PUBLIC = Path('/ml2/hypergan/ParticleGAN-ra11-pr155')
ORIGINAL = Path('/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py')
ADAPTED = STUDY / 'validation-ra15/moving/rotated100/adapted_runner.py'
HOSTS = Path('/ml2/hypergan/lrfree-20260926/harness/hosts')
PACKAGES = dict(E22=HERE / 'pkg-pr155-e22-cabe2084', Atlas=STUDY / 'pkg-RA17-current-pr155')
CONFIGS = dict(E22=HERE / 'configs/pr155-e22.json', Atlas=HERE / 'configs/particle-gan-atlas.json')
ORIGINAL_ATLAS = STUDY / 'validation-ra15/moving/rotated100'
LOCK = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/quality/.serial-phase.lock')
GPU_UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
ENV = dict(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(package):
    h = hashlib.sha256()
    for path in sorted((package / 'particlegan').rglob('*.py')):
        h.update(str(path.relative_to(package / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def adapted(variant):
    source = ADAPTED.read_text()
    source = replace_once(source, f"REPO = {str(STUDY / 'pkg-RA15-partial-recovery')!r}",
                          f'REPO = {str(PACKAGES[variant])!r}')
    source = replace_once(source, f"options = json.load(open({str(STUDY / 'configs/RA15-partial-recovery.json')!r}))",
                          f'options = json.load(open({str(CONFIGS[variant])!r}))')
    if variant == 'E22':
        # PR155's independent-row reference sampler has the original API;
        # Atlas's helper-only indices alias is not added to the baseline.
        source = replace_once(source, '            latent, indices = table.sample(n, generator=latent_stream)',
                              '            latent, _ = table.sample(n, generator=latent_stream)')
        source = replace_once(source, '            clean = trainer._generate(model, latent, 0., latent_stream, indices=indices)',
                              '            clean = trainer._generate(model, latent, 0., latent_stream)')
    # Baseline package has no feature selection or settled guard. The extra
    # MECHANISM print was harness instrumentation, not a model/update call.
    mechanism = "        print('MECHANISM ' + json.dumps(dict(step=step, surprise=None if trainer.policy.surprise is None else trainer.policy.surprise.diagnostics(), backend_selection=trainer.policy._feature_selection.state_dict(), reopen_guard=trainer.policy.reopen_guard.state_dict())), flush=True)"
    source = replace_once(source, mechanism, "        print('CHECKPOINT ' + json.dumps(dict(step=step)), flush=True)")
    source = replace_once(source, 'snap(0)\nfor step in range(1, args.steps + 1):',
        'start = support.begin(trainer, stream, real_batch, draw, angle, rotation, CENTERS, gate_rows, capture_settings)\n'
        'if start == 0:\n'
        '    snap(0)\n'
        'support.capture(start)\n'
        'for step in range(start + 1, args.steps + 1):')
    source = replace_once(source, '        trainer.step(real_d, generator_real=generator_real)',
                          '        result = trainer.step(real_d, generator_real=generator_real)')
    source = replace_once(source, '    if step % args.every == 0:\n        snap(step)',
                          '    if step % args.every == 0:\n        snap(step)\n    support.note_update(step, result)')
    beginning = "final = draw(20000, seed + 403, seed + 402)   # the harness's evaluation seeds"
    assert source.count(beginning) == 1
    prefix, tail = source.split(beginning, 1)
    tail = beginning + tail
    source = prefix + "if capture_settings['stage'] == 'full':\n" + ''.join(
        '    ' + line if line.strip() else line for line in tail.splitlines(keepends=True))
    source += '    support.finish_full()\nelse:\n    support.finish_window()\n'
    return source


def source_contract(variant):
    old, new = ast.parse(ADAPTED.read_text()), ast.parse(adapted(variant))
    dump = lambda n: ast.dump(n, include_attributes=False)
    functions = {n.name:n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name:n for n in new.body if isinstance(n, ast.FunctionDef)}
    for name in functions:
        if name != 'snap':
            expected = functions[name]
            if name == 'draw' and variant == 'E22':
                expected = next(n for n in ast.parse(ORIGINAL.read_text()).body
                                if isinstance(n, ast.FunctionDef) and n.name == 'draw')
            assert dump(expected) == dump(new_functions[name]), name
    # Only the passive checkpoint metadata print in snap differs. Original
    # plot draw, 20k draw and scoring/acceptance statements remain identical.
    old_snap, new_snap = functions['snap'], new_functions['snap']
    def strip_prints(node):
        class Strip(ast.NodeTransformer):
            def visit_Expr(self, n):
                if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'print':
                    return None
                return self.generic_visit(n)
        return Strip().visit(node)
    assert dump(strip_prints(old_snap)) == dump(strip_prints(new_snap))
    old_loop = next(n for n in old.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    new_loop = next(n for n in new.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    class StripInstrumentation(ast.NodeTransformer):
        def visit_Expr(self, n):
            if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'support':
                return None
            return self.generic_visit(n)
        def visit_Assign(self, n):
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'result':
                return ast.Expr(value=n.value)
            return self.generic_visit(n)
    new_loop = StripInstrumentation().visit(new_loop)
    assert [dump(n) for n in old_loop.body] == [dump(n) for n in new_loop.body]
    old_tail = old.body[old.body.index(old_loop)+1:]
    new_tail = next(n for n in new.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Subscript)
                    and isinstance(n.test.left.value, ast.Name)
                    and n.test.left.value.id == 'capture_settings')
    assert [dump(n) for n in old_tail] == [dump(n) for n in new_tail.body[:-1]]
    compile(adapted(variant), f'<{variant}-capture>', 'exec')
    return dict(host_update_body_identical=True, original_draw_identical=True,
                original_gate_and_terminal_scorer_AST_identical=True, original_seed=1234)


def prepare():
    assert not (HERE / 'SOURCE-FREEZE.json').exists()
    baseline = json.loads((HERE / 'BASELINE-SOURCE-FREEZE.json').read_text())
    for path, value in baseline['hashes'].items():assert sha(path) == value, path
    assert digest(PACKAGES['E22']) == baseline['package_sha256']
    assert digest(PACKAGES['Atlas']) == '500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61'
    assert digest(PUBLIC) == digest(PACKAGES['Atlas'])
    contracts = {variant:source_contract(variant) for variant in PACKAGES}
    for variant in PACKAGES:
        with (HERE / f'{variant.lower()}_runner.py').open('x') as handle:handle.write(adapted(variant))
    write(HERE / 'HOST-ADAPTER-CHECK.json', dict(status='SOURCE_PASS', variants=contracts,
        extra_observations='state-guarded original draw with isolated derived geometry cache/work',
        original_gate_cadence=500, capture_cadence=10, target_shift_steps=[500,1000],
        acceptance_draw_count=20000, visualization_draw_count=4096))
    files = set(map(Path, baseline['hashes']))
    files.update((PUBLIC / 'particlegan').rglob('*.py'))
    files.update((PACKAGES['Atlas'] / 'particlegan').rglob('*.py'))
    files.update((HOSTS / 'native100').rglob('*.py'))
    files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'));files.update(HERE.glob('*.json'))
    files.update(CONFIGS.values())
    files.update([ORIGINAL, ADAPTED, ORIGINAL_ATLAS/'checkpoint-001000.pt',
        ORIGINAL_ATLAS/'frames.npz.verdict.json', ORIGINAL_ATLAS/'LAUNCH.json', ORIGINAL_ATLAS/'COMPLETION.json',
        STUDY/'portability/ra17-current-pr155/SOURCE-FREEZE.json', STUDY/'validation-ra17/SOURCE-FREEZE.json'])
    write(HERE / 'SOURCE-FREEZE.json', dict(status='FROZEN_NOT_LAUNCHED',
        hashes={str(p):sha(p) for p in sorted(files)}, packages={v:dict(root=str(p),sha256=digest(p)) for v,p in PACKAGES.items()},
        public_core_sha256=digest(PUBLIC), config_sha256={v:sha(p) for v,p in CONFIGS.items()},
        E22_base_commit=baseline['base_commit'], original_seed=1234,
        original_fixture=dict(particles=20000,z_dim=2,batch_size=2048,steps=1500,
            rotate_every=500,degrees_per_turn=30,turns=2,gate_samples=20000,serial_backward=True),
        parity_windows=dict(Atlas=[1001,1010], E22=[1,20]),
        parity_window_note='Atlas restores the original RA15 update1000 checkpoint. No current-base E22 update1000 checkpoint exists; its first two complete reactions are compared from the original initialization.',
        excluded_parity_fields=['birth_death.last.eval_seconds'],
        GPU_operations=0, training_updates=0, original_acceptance_rule_unchanged=True))
    print(json.dumps(verify(), sort_keys=True), flush=True)


def verify():
    frozen = json.loads((HERE/'SOURCE-FREEZE.json').read_text())
    for path, expected in frozen['hashes'].items():assert sha(path)==expected,path
    for variant, record in frozen['packages'].items():assert digest(Path(record['root']))==record['sha256'],variant
    assert digest(PUBLIC)==frozen['public_core_sha256']
    return dict(status='VALID', files=len(frozen['hashes']), source_freeze_sha256=sha(HERE/'SOURCE-FREEZE.json'),
        packages={v:r['sha256'] for v,r in frozen['packages'].items()})


def parked_owner():
    for pid, ticks, stopped in ((384331,'163720702',True),(383348,'163716372',False)):
        text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
        assert fields[19]==ticks,'original process identity changed'
        if stopped:assert fields[0]=='T','original numerical supervisor is no longer parked'


def worker(args):
    assert args.lock_fd is not None and args.lock_fd>2
    actual, expected=os.fstat(args.lock_fd),LOCK.stat()
    assert (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino)
    fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=verify();parked_owner()
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
    props=torch.cuda.get_device_properties(0)
    uuid='GPU-'+str(props.uuid).removeprefix('GPU-').lower();assert uuid==GPU_UUID
    import instrumentation as support
    output=args.output.resolve();assert output.is_dir()
    source=HERE/f'{args.variant.lower()}_runner.py'
    settings=dict(output=output,variant=args.variant,stage=args.stage,observed=args.observed,
        restore_checkpoint=(ORIGINAL_ATLAS/'checkpoint-001000.pt' if args.stage=='window' and args.variant=='Atlas' else None),
        inherited_verdict=(ORIGINAL_ATLAS/'frames.npz.verdict.json' if args.stage=='window' and args.variant=='Atlas' else None))
    steps=1500 if args.stage=='full' else 1010 if args.variant=='Atlas' else 20
    write(output/'LAUNCH.json',dict(status='RUNNING',variant=args.variant,stage=args.stage,observed=args.observed,
        command=[sys.executable,*sys.argv],source_integrity_before=before,
        resources=dict(physical_gpu=0,gpu_uuid=uuid,gpu_name=props.name,memory_fraction=.2,total_gpu_bytes=props.total_memory),
        package_sha256=digest(PACKAGES[args.variant]),config_sha256=sha(CONFIGS[args.variant]),
        task='rotated100',seed=1234,last_update=steps,numerical_started=time.time()))
    sys.argv=[str(source),'rotated100',str(output/'original-frames.npz'),'--every','500','--points','4096',
        '--steps',str(steps),'--gate','--rotate-every','500','--rotate-deg','30']
    exec(compile(source.read_text(),str(source),'exec'),dict(__name__='__main__',__file__=str(source),
        owned_output=output,support=support,capture_settings=settings))
    write(output/'SOURCE-CLOSE.json',dict(status='VALID',source_integrity_after=verify(),
        peak_allocated_gpu_mib=torch.cuda.max_memory_allocated(0)/2**20,
        peak_reserved_gpu_mib=torch.cuda.max_memory_reserved(0)/2**20))


def child(lock, target, variant, stage, observed):
    target.mkdir()
    cmd=[sys.executable,'-u','-B',str(HERE/'run_capture.py'),'--worker','--variant',variant,
         '--stage',stage,'--output',str(target),'--lock-fd',str(lock.fileno())]
    if observed:cmd.append('--observed')
    print(json.dumps(dict(event='worker_start',variant=variant,stage=stage,observed=observed,output=str(target))),flush=True)
    with (target/'run.log').open('x') as log:
        process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),close_fds=True,pass_fds=(lock.fileno(),))
    if process.returncode:
        write(target/'ERROR.json',dict(status='ERROR',exit_code=process.returncode,source_integrity_after=verify()))
        raise SystemExit(process.returncode)
    assert json.loads((target/'SOURCE-CLOSE.json').read_text())['status']=='VALID'


def run(args):
    for name in ('ABSENT','ABSENT_START','ABSENT_END','LRFREE_NATIVE_TEST_STEPS'):os.environ.pop(name,None)
    os.environ.update(ENV)
    before=verify();out=args.output.resolve();assert not out.exists();out.mkdir(parents=True)
    write(out/'LAUNCH.json',dict(status='WAITING_FOR_GPU',command=[sys.executable,*sys.argv],source_integrity_before=before,start=time.time()))
    print(json.dumps(dict(event='waiting_for_gpu',output=str(out))),flush=True)
    if args.full:
        parity=json.loads(args.parity_receipt.read_text())
        assert parity['status']=='PASS' and parity['source_integrity_after']['source_freeze_sha256']==before['source_freeze_sha256']
    with LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);parked_owner();verify()
        results={}
        for variant in ('E22','Atlas'):
            directory=out/variant.lower();directory.mkdir()
            if args.parity:
                child(lock,directory/'control',variant,'window',False)
                child(lock,directory/'observed',variant,'window',True)
                import torch
                from instrumentation import compare
                read=lambda p:torch.load(p,map_location='cpu',weights_only=False)
                state_diff=compare(read(directory/'control/final-state.pt'),read(directory/'observed/final-state.pt'),skip=('birth_death.last.eval_seconds',))
                runtime_diff=compare(read(directory/'control/runtime-state.pt'),read(directory/'observed/runtime-state.pt'))
                loss_equal=json.loads((directory/'control/LOSSES.json').read_text())==json.loads((directory/'observed/LOSSES.json').read_text())
                a=json.loads((directory/'control/WINDOW-COMPLETION.json').read_text())
                b=json.loads((directory/'observed/WINDOW-COMPLETION.json').read_text())
                result=dict(status='PASS' if not state_diff and not runtime_diff and loss_equal and a['original_evaluation_score']==b['original_evaluation_score'] else 'FAIL',
                    semantic_state_differences=state_diff,runtime_state_differences=runtime_diff,
                    exact_losses=loss_equal,exact_original_evaluation=a['original_evaluation_score']==b['original_evaluation_score'],
                    control=a,observed=b,exclusions=['birth_death.last.eval_seconds'])
                write(directory/'PARITY.json',result);assert result['status']=='PASS',result
                results[variant]=result
            else:
                child(lock,directory/'capture',variant,'full',True)
                results[variant]=json.loads((directory/'capture/CAPTURE-COMPLETION.json').read_text())
            print(json.dumps(dict(event='variant_complete',variant=variant,status=results[variant]['status'])),flush=True)
        if args.full:
            import numpy as np
            with np.load(out/'e22/capture/dense-frames.npz',allow_pickle=False) as e22,np.load(out/'atlas/capture/dense-frames.npz',allow_pickle=False) as atlas:
                for key in ('steps','angles','event_kind','centers'):assert np.array_equal(e22[key],atlas[key]),key
                assert np.array_equal(e22['frames'][0],atlas['frames'][0]),'initial observed sample cloud differs'
        write(out/('PARITY-COMPLETION.json' if args.parity else 'COMPLETION.json'),dict(status='PASS' if args.parity else 'COMPLETE',
            evidence_validity='VALID',variants=results,source_integrity_after=verify(),completed=time.time(),
            scope='observer numerical parity' if args.parity else 'fresh original moving rotated100 E22 versus Atlas',
            original_scoring_and_acceptance_unchanged=True))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    action=ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--check-only',action='store_true')
    action.add_argument('--parity',action='store_true');action.add_argument('--full',action='store_true')
    action.add_argument('--worker',action='store_true')
    ap.add_argument('--output',type=Path);ap.add_argument('--parity-receipt',type=Path)
    ap.add_argument('--variant',choices=['E22','Atlas']);ap.add_argument('--stage',choices=['window','full'])
    ap.add_argument('--observed',action='store_true');ap.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.prepare:prepare()
    elif args.check_only:print(json.dumps(verify(),sort_keys=True),flush=True)
    elif args.worker:
        assert args.output is not None and args.variant is not None and args.stage is not None
        worker(args)
    else:
        assert args.output is not None
        assert not args.full or args.parity_receipt is not None
        run(args)


if __name__=='__main__':main()
