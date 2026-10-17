"""Read-only captures of tensors already computed by the original19 observers.

The reversible overlays below add no forward, sample, score or optimizer call.
Imports are intentionally model-free; Torch is accessed only in an actual
already-running child. The renderer consumes retained arrays, never a checkpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

import numpy as np

SELF = 'reports/forge/pr223-original-full-retest-20261004/goal_observer.py'
_RECORDER = None
_VECTOR_PHASE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def typed_digest(value):
    """Typed, ordered, NaN-bit-aware value digest; no pickle/lazy imports."""
    h = hashlib.sha256()
    def put(v):
        kind = type(v)
        if kind in (dict,OrderedDict):
            h.update(kind.__name__.encode()+b'{')
            for key,child in v.items(): put(key); put(child)
            h.update(b'}')
            if kind is OrderedDict:
                # nn.Module.state_dict carries version metadata on this exact
                # source-defined mapping type. Preserve it as typed state too.
                h.update(b'_metadata:')
                put(getattr(v,'_metadata',None))
        elif kind in (tuple,list):
            h.update(kind.__name__.encode()+b'[')
            for child in v: put(child)
            h.update(b']')
        elif kind is float:
            h.update(b'float'+struct.pack('>d',v))
        elif kind in (str,int,bool,type(None)):
            h.update(kind.__name__.encode()+b':'+repr(v).encode()+b'\0')
        elif isinstance(v,np.ndarray):
            if v.dtype.hasobject: raise ValueError('object arrays are not typed state')
            h.update(b'ndarray'+v.dtype.str.encode()+repr(v.shape).encode()+np.ascontiguousarray(v).tobytes())
        elif isinstance(v,np.generic):
            put(np.asarray(v))
        elif 'torch' in sys.modules and isinstance(v,sys.modules['torch'].Tensor):
            t = v.detach().cpu().contiguous()
            h.update(b'tensor'+str(v.dtype).encode()+str(v.device).encode()+repr(tuple(v.shape)).encode())
            h.update(t.reshape(-1).view(sys.modules['torch'].uint8).numpy().tobytes())
        else:
            raise ValueError('unknown typed state leaf '+kind.__name__)
    put(value)
    return h.hexdigest()


def global_rng_state():
    state = {'python':random.getstate(),'numpy':np.random.get_state()}
    torch = sys.modules.get('torch')
    if torch is not None:
        state['torch_cpu'] = torch.get_rng_state()
        # Never initialize CUDA or make another physical device visible.
        if torch.cuda.is_initialized():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def owner_state(trainer):
    state = trainer.state_dict()
    policy = getattr(trainer,'policy',None)
    modules = getattr(policy,'_training_modules',None)
    if callable(modules):
        modules=list(modules().values())+list(policy._average_modules().values())
    elif modules is None:
        modules = [getattr(trainer,n) for n in ('G','D','prior','ema_G','ema_prior')]
    modes = []
    for root in modules:
        for module in root.modules(): modes.append(bool(module.training))
    return {'checkpoint':state,'module_modes':modes}


def cpu_array(value):
    if isinstance(value,np.ndarray):
        result = value.copy()
    else:
        result = value.detach().cpu().numpy().copy()
    if result.dtype.hasobject or not np.issubdtype(result.dtype,np.number) or not np.isfinite(result).all():
        raise ValueError('retained goal array must be finite numeric data')
    return result


class Recorder:
    def __init__(self, output, case, *, source_guard, deadline_guard,
                 state_reader=owner_state, rng_reader=global_rng_state):
        self.output=Path(output);self.case=deepcopy(case)
        self.source_guard=source_guard;self.deadline_guard=deadline_guard
        self.state_reader=state_reader;self.rng_reader=rng_reader
        self.records=[];self.final_owner=None
        self.folder=self.output/'goal-observations';self.folder.mkdir()

    def guard(self):
        self.deadline_guard(); self.source_guard()

    def _pure_read(self, trainer, action):
        self.guard()
        rng_before=typed_digest(self.rng_reader())
        state_before=typed_digest(self.state_reader(trainer))
        result=action()
        state_after=typed_digest(self.state_reader(trainer))
        rng_after=typed_digest(self.rng_reader())
        self.guard()
        if state_before!=state_after or rng_before!=rng_after:
            raise ValueError('goal capture changed complete public state or global/named RNG')
        return result,{'state_before_sha256':state_before,'state_after_sha256':state_after,
                       'global_rng_before_sha256':rng_before,'global_rng_after_sha256':rng_after,
                       'state_digest_kind':'typed_values_v1','pure':True}

    def capture(self, trainer, step, samples, target, *, source, output_sigma,
                target_kind='original_reference', geometry=None):
        if step not in self.case['media_steps']: return
        if step in [r['step'] for r in self.records]:
            raise ValueError('duplicate primary goal capture')
        if getattr(trainer,'completed_steps',None)!=step:
            raise ValueError('goal clock differs from public completed updates')
        def read():
            state=trainer.state_dict()
            if json.loads(json.dumps(state['recipe']))!=self.case['resolved_recipe']:
                raise ValueError('complete original Recipe changed')
            arrays={'samples':cpu_array(samples),'target':cpu_array(target)}
            if geometry is not None:
                arrays.update({k:cpu_array(v) for k,v in geometry.items()})
            selected=state.get('policy',{}).get('served_source')
            if selected not in {'fast','averaged'}:
                raise ValueError('actual selected policy owner unavailable')
            return arrays,selected
        (arrays,selected),purity=self._pure_read(trainer,read)
        sigma=float(output_sigma)
        if not math.isfinite(sigma) or sigma<0: raise ValueError('invalid actual learned output width')
        path=self.folder/f'step_{step:06d}.npz'
        np.savez_compressed(path,**arrays)
        self.guard()
        record={'step':step,'file':path.relative_to(self.output).as_posix(),'sha256':sha(path),
                'bytes':path.stat().st_size,'shapes':{k:list(v.shape) for k,v in arrays.items()},
                'dtypes':{k:str(v.dtype) for k,v in arrays.items()},'primary_source':source,
                'selected_source':selected,'output_sigma':sigma,'target_kind':target_kind,
                'purity':purity,'new_draws':0,'new_forwards':0,'new_scores':0,'new_updates':0}
        self.records.append(record)
        write(self.output/'goal-capture.json',{'schema':'pr223_goal_capture_v1','records':self.records})

    def attest_final(self, trainer):
        expected=self.case['original_definition']['original_host']['steps']
        if trainer.completed_steps!=expected: raise ValueError('final owner clock is not the full original budget')
        def read():
            state=trainer.state_dict()
            if json.loads(json.dumps(state['recipe']))!=self.case['resolved_recipe']:
                raise ValueError('final complete original Recipe changed')
            if state['completed_steps']!=expected: raise ValueError('checkpoint clock changed')
            torch=sys.modules.get('torch')
            if torch is not None:
                def health(v):
                    if isinstance(v,torch.Tensor) and not torch.isfinite(v).all():
                        raise ValueError('nonfinite learned model or optimizer tensor')
                    if isinstance(v,dict):
                        for c in v.values(): health(c)
                    elif isinstance(v,(tuple,list)):
                        for c in v: health(c)
                health(state['models']);health(state['optimizers'])
            return {'completed_steps':expected,'resolved_recipe':state['recipe'],
                    'checkpoint_sha256':typed_digest(state),'state_digest_kind':'typed_values_v1',
                    'policy':state.get('policy'),'backend_selection':state.get('backend_selection'),
                    'streams_sha256':{k:typed_digest(v) for k,v in state['streams'].items()},
                    'optimizer_group_rates':[[float(g['lr']) for g in o.param_groups]
                                             for o in (trainer.opt_g,trainer.opt_d)]}
        receipt,purity=self._pure_read(trainer,read)
        receipt['purity']=purity
        write(self.output/'final-owner-receipt.json',receipt)
        self.final_owner=receipt

    def finish(self):
        if [r['step'] for r in self.records]!=self.case['media_steps'] or self.final_owner is None:
            raise ValueError('full goal/owner capture is unavailable')
        self.guard()


def configure(recorder):
    global _RECORDER
    if _RECORDER is not None: raise ValueError('one original case per fresh child')
    _RECORDER=recorder


def guard():
    if _RECORDER is None: raise ValueError('unconfigured goal observer')
    _RECORDER.guard()


def primary(ctx, trainer, step, samples, target, *, ema=False, phase=True,
            sigma=0., source='original_primary', target_kind='original_reference', geometry=None):
    if not ema and phase:
        _RECORDER.capture(trainer,step,samples,target,source=source,output_sigma=sigma,
                          target_kind=target_kind,geometry=geometry)


@contextmanager
def vector_phase(ctx,trainer,completed,ema,sigma,s):
    global _VECTOR_PHASE
    old=_VECTOR_PHASE
    _VECTOR_PHASE=(ctx,trainer,completed,ema,sigma,s)
    try: yield
    finally: _VECTOR_PHASE=old


def vector_reference(fake,real,completed):
    if _VECTOR_PHASE is None: raise ValueError('vector scorer capture without explicit observer phase')
    ctx,trainer,step,ema,sigma,s=_VECTOR_PHASE
    if step!=completed: raise ValueError('vector reference clock differs')
    primary(ctx,trainer,step,fake,real,ema=ema,phase=s==sigma,sigma=sigma,
            source='original_noisy_vector_score_input',target_kind='original_scorer_seed991_target')


def final(trainer):
    _RECORDER.attest_final(trainer)


def patches(kind):
    if kind=='screen':
        return [
          ('import traceback\n','import traceback\nimport _pr223_goal_observer as _goal\n'),
          ("    (options, detected) = resolve_options(package, declared, given)",
           "    _goal.guard()\n    (options, detected) = resolve_options(package, declared, given)"),
          ('                noisy_x = fake + sigma * torch.randn_like(fake) if sigma else fake\n',
           '                noisy_x = fake + sigma * torch.randn_like(fake) if sigma else fake\n                _goal.primary(ctx, trainer, trainer.completed_steps, noisy_x, means, ema=ema, sigma=sigma, source="original_mode_noisy_input", target_kind="original_ring_centers")\n'),
          ('                images = trainer._generate(*arguments)\n',
           '                images = trainer._generate(*arguments)\n                _goal.primary(ctx, trainer, trainer.completed_steps, images, centers, ema=ema, phase=s == sigma, sigma=sigma, source="original_image_noisy_input", target_kind="original_image_template_bank")\n'),
          ('                    out.append(host.score_samples(fake, cfg, completed))',
           '                    with _goal.vector_phase(ctx, trainer, completed, ema, sigma, s):\n                        out.append(host.score_samples(fake, cfg, completed))'),
          ('            noisy = host.diversity(value.sample(4096, ema=ema, generator=isolated, output_noise=True), means)',
           '            _goal_points = value.sample(4096, ema=ema, generator=isolated, output_noise=True)\n            noisy = host.diversity(_goal_points, means)'),
          ('            sigma = ctx.sigma(value)\n',
           '            sigma = ctx.sigma(value)\n            if value is trainer and not ema:\n                _goal.primary(ctx, value, value.completed_steps, _goal_points, means, sigma=sigma, source="original_ring_noisy_diversity_input", target_kind="current_original_ring_centers")\n'),
          ("            np.savez_compressed(d / 'snapshots' / f'step_{step:06d}.npz', live=arrays[kind]['live'][:snap_n], ema=arrays[kind]['ema'][:snap_n], target=target_snap)",
           "            np.savez_compressed(d / 'snapshots' / f'step_{step:06d}.npz', live=arrays[kind]['live'][:snap_n], ema=arrays[kind]['ema'][:snap_n], target=target_snap)\n            if kind == 'noisy':\n                _goal.primary(ctx, trainer, step, arrays[kind]['live'][:snap_n], target_snap, sigma=sigma, source='original_native_noisy_saved_live', geometry={'centers': geometry_centres, 'reference_sigma': np.asarray(float(geometry_std))})"),
          ("        if options['save_final_state']:\n", "        _goal.final(trainer)\n        if options['save_final_state']:\n"),
          ("    brief = {k: out.get(k)", "    _goal.guard()\n    brief = {k: out.get(k)")]
    if kind=='vector':
        return [('import torch\n','import torch\nimport _pr223_goal_observer as _goal\n'),
          ('    centered = real-real.mean(0)',
           '    _goal.vector_reference(fake, real, completed_steps)\n    centered = real-real.mean(0)')]
    if kind=='moving':
        return [('import numpy as np\n','import numpy as np\nimport _pr223_goal_observer as _goal\n'),
          ('# screen.py deterministic()','_goal.guard()\n\n# screen.py deterministic()'),
          ('    angles.append(angle(step))',
           '    angles.append(angle(step))\n    _goal.primary(None, trainer, step, frames[-1], CENTERS, sigma=float(trainer.output_sigma()), source="original_moving_saved_frame", target_kind="base_centers_plus_recorded_angle", geometry={"angle": np.asarray(angles[-1])})'),
          ('final = draw(20000, seed + 403, seed + 402)',
           '_goal.final(trainer)\nfinal = draw(20000, seed + 403, seed + 402)'),
          ('                    final=json.dumps(check))',
           '                    final=json.dumps(check))\n_goal.guard()')]
    raise ValueError('unknown original observer overlay')


def overlay(source,kind):
    original=source
    applied=[]
    for before,after in patches(kind):
        if source.count(before)!=1: raise ValueError('original capture marker changed: '+repr(before))
        source=source.replace(before,after,1)
        applied.append({'before':before,'after':after})
    if remove_overlay(source,kind)!=original: raise ValueError('irreversible observation overlay')
    return source,{'kind':kind,'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
                   'executed_sha256':hashlib.sha256(source.encode()).hexdigest(),
                   'operations':applied,'scope':'read-only already-computed tensors; no added model/sample/score/update call'}


def remove_overlay(source,kind):
    for before,after in reversed(patches(kind)):
        if source.count(after)!=1: raise ValueError('derived observer wrapper changed')
        source=source.replace(after,before,1)
    return source


def clock_caption(group,point):
    """Display source-recorded flags; native coverage is not fidelity."""
    def flag(key):
        return 'PASS' if point.get(key) is True else 'FAIL' if point.get(key) is False else 'UNAVAILABLE'
    if group=='native':
        return (f"coverage-read {flag('pass')} · fidelity {flag('acc_accuracy_pass')} · "
                f"frozen {flag('acc_frozen_pass')} · combined-read {flag('acc_passed')}")
    if 'pass' in point:
        return 'read '+flag('pass')
    if point:
        label='gate metadata only'
        if group=='moving':
            if isinstance(point.get('rule'),str): label+=' · '+point['rule']
            pre=point.get('pre_turn_hq')
            if type(pre) in (int,float) and math.isfinite(pre): label+=f' · pre-turn HQ {pre:.4g}'
        return label
    return 'ungated movie state'


def render(output,case,verdict,*,guard_callback):
    """Illustrate actual primary tensors; recorded numerical verdict is authority."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    output=Path(output)
    capture=json.loads((output/'goal-capture.json').read_text())
    records=capture['records']
    if [r['step'] for r in records]!=case['media_steps']:
        raise ValueError('goal renderer clock/count differs')
    if verdict not in {'PASS','FAIL'}: raise ValueError('unaccepted partial evidence has no full goal verdict')
    before={r['file']:sha(output/r['file']) for r in records}
    rows={}
    if case['group']=='moving':
        metric_file='frames.npz.verdict.json'
        before[metric_file]=sha(output/metric_file)
        gate=json.loads((output/metric_file).read_text())
        rows={p['period_end']:p for p in gate['periods']}
        scope='4 original4096-point movie states; independent20k gates at500/1000/1500. Step0 is ungated.'
    else:
        metric_file='metrics.jsonl'
        before[metric_file]=sha(output/metric_file)
        rows={p['step']:p for line in (output/metric_file).read_text().splitlines()
              if (p:=json.loads(line))}
        scope=('Original20k coverage/fidelity checks + final5 + separate100k holdout; shown saved4096-point cloud.'
               if case['group']=='native' else 'Original noisy primary read; clean/forcedEMA diagnostics have separate scope.')
    frames=[];payloads=[];windows=[]
    for record in records:
        guard_callback()
        path=output/record['file']
        if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes'] or record['purity']['pure'] is not True:
            raise ValueError('retained pure primary array changed')
        with np.load(path,allow_pickle=False) as arrays:
            samples,target=arrays['samples'].copy(),arrays['target'].copy()
            centers=arrays['centers'].copy() if 'centers' in arrays else None
            ref_sigma=float(arrays['reference_sigma']) if 'reference_sigma' in arrays else None
            angle=float(arrays['angle']) if 'angle' in arrays else None
        if not np.isfinite(samples).all() or not np.isfinite(target).all(): raise ValueError('nonfinite retained goal pixels')
        image=samples.ndim>=3
        if angle is not None:
            c,s=math.cos(angle),math.sin(angle)
            target=target@np.asarray([[c,-s],[s,c]]).T
        if not image:
            combined=np.concatenate([target,samples]);windows.append((combined.min(0),combined.max(0)))
        payloads.append((record,samples,target,centers,ref_sigma,image))
    limits=None
    if windows:
        lo=np.stack([x[0] for x in windows]).min(0);hi=np.stack([x[1] for x in windows]).max(0)
        pad=np.maximum((hi-lo)*.05,.05)
        limits=(lo-pad,hi+pad)
    for record,samples,target,centers,ref_sigma,image in payloads:
        guard_callback()
        fig,axes=plt.subplots(1,3 if centers is not None else 2,figsize=(11,5))
        if image:
            def mosaic(values,columns):
                v=values[:,0] if values.ndim==4 and values.shape[1]==1 else values
                if v.ndim!=3: raise ValueError('original grayscale image shape changed')
                rows=(len(v)+columns-1)//columns
                tiled=np.ones((rows*v.shape[-2],columns*v.shape[-1]),dtype=v.dtype)
                for i,a in enumerate(v):
                    y,x=divmod(i,columns);tiled[y*a.shape[0]:(y+1)*a.shape[0],x*a.shape[1]:(x+1)*a.shape[1]]=a
                return tiled
            axes[0].imshow(mosaic(target,min(4,len(target))),cmap='gray',vmin=0,vmax=1)
            axes[1].imshow(mosaic(samples,8),cmap='gray',vmin=0,vmax=1)
            for ax in axes: ax.set_axis_off()
        else:
            if samples.ndim!=2 or samples.shape[1]!=2 or target.ndim!=2 or target.shape[1]!=2:
                raise ValueError('original goal cloud dimension changed')
            axes[0].scatter(target[:,0],target[:,1],s=3,alpha=.4,color='#537f99')
            axes[1].scatter(samples[:,0],samples[:,1],s=3,alpha=.4,color='#db4e75')
            for ax in axes[:2]:
                ax.set_xlim(limits[0][0],limits[1][0]);ax.set_ylim(limits[0][1],limits[1][1]);ax.set_aspect('equal')
            if centers is not None:
                if ref_sigma is None or not math.isfinite(ref_sigma) or ref_sigma<=0: raise ValueError('native source width unavailable')
                center=centers[0];extent=3*ref_sigma
                mask=np.all(np.abs(samples-center)<=extent,axis=1)
                axes[2].scatter(*(samples[mask]-center).T,s=4,alpha=.5,color='#db4e75')
                axes[2].add_patch(Circle((0,0),ref_sigma,fill=False,color='#537f99',label='Original 1σ'))
                axes[2].add_patch(Circle((0,0),2*ref_sigma,fill=False,color='#537f99',linestyle='--',label='Original 2σ'))
                axes[2].set_xlim(-extent,extent);axes[2].set_ylim(-extent,extent);axes[2].set_aspect('equal')
                axes[2].set_title('Mode0 width (display filter only)');axes[2].legend(fontsize=8)
        axes[0].set_title('Original reference');axes[1].set_title('Actual noisy selected output')
        source=record['selected_source']
        point=rows.get(record['step'],{})
        shown=[f'{k}={point[k]:.4g}' for k in ('modes','hq','mass_tv','sw1_normalized','covariance_error',
               'acc_center_rms_sigma','acc_abs_cov_trace_bias','acc_radial_ks')
               if type(point.get(k)) in (int,float) and math.isfinite(point[k])]
        clock_verdict=clock_caption(case['group'],point)
        fig.suptitle(f"Original PR223 Atlas · {case['group']}/{case['task']} · FINAL full-protocol verdict {verdict}\n"
                     f"Actual step {record['step']}/{case['original_definition']['original_host']['steps']} · selected {source} · kernel σ {record['output_sigma']:.4g}",fontsize=12)
        fig.text(.025,.05,f"{clock_verdict} · "+' · '.join(shown[:5]),fontsize=8)
        fig.text(.025,.025,scope+' No new draws or metric credit.',fontsize=8)
        fig.tight_layout(rect=(0,.10,1,.87));fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
        plt.close(fig)
    destination=output/'goal.gif'
    frames[0].save(destination,save_all=True,append_images=frames[1:],duration=750,loop=0)
    with Image.open(destination) as gif:
        if gif.n_frames!=len(records): raise ValueError('decoded goal frames differ from actual clocks')
    if before!={name:sha(output/name) for name in before}: raise ValueError('goal renderer mutated retained arrays or caption metrics')
    guard_callback()
    receipt={'file':'goal.gif','sha256':sha(destination),'bytes':destination.stat().st_size,
             'frames':len(records),'actual_steps':case['media_steps'],'metric_only':False,
             'new_draws':0,'new_forwards':0,'new_scores':0,'new_updates':0,
             'original_gate':verdict,'renderer_sha256':sha(__file__),
             'input_files':before,'input_files_bytes':{name:(output/name).stat().st_size for name in before},
             'raw_arrays_unchanged':True,'caption_metrics_unchanged':True,
             'fixed_comparison_limits':None if limits is None else [v.tolist() for v in limits],
             'verdict_caption':'FINAL full-protocol verdict','per_clock_metrics_from_original_json_only':True}
    write(output/'goal-media-receipt.json',receipt)
    return receipt
