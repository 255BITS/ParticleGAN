"""Observe an unchanged follow-up CLI; stop after N complete outer updates.

No training values or RNG state are changed. Optional --deterministic/--interop
are explicit environment controls, never candidate hyperparameter changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--method', default='holdw15')
parser.add_argument('--updates', type=int, default=3)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--interop', type=int)
a = parser.parse_args()
root = a.repo.resolve()
a.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root))
import torch
if a.deterministic:
    torch.use_deterministic_algorithms(True)
if a.interop:
    torch.set_num_interop_threads(a.interop)
from reports.toy100.alternating_curvature_scratch import BothBoundRecorder
from reports.toy100.pr84_smoothed_candidate import SmoothedBothBoundRecorder

rows = []
arms = []
initial = {}

def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())

class PrefixComplete(Exception):
    pass

original_phases = BothBoundRecorder.phases
original_arm = SmoothedBothBoundRecorder._arm_smoothed_critic

def observed_phases(self, step, opt_d, opt_g, local):
    if not initial:
        tensors = {}
        for role in ('generator', 'critic', 'prior'):
            for name, value in local[role].state_dict().items():
                tensors[f'{role}.{name}'] = value.detach().clone()
        tensors['torch_rng'] = torch.get_rng_state().clone()
        for key, value in local.items():
            if isinstance(value, torch.Generator):
                tensors[f'rng.{key}'] = value.get_state().clone()
        initial.update({name: digest(value) for name, value in tensors.items()})
        torch.save(tensors, a.output / 'initial-state.pt')
    for phase in original_phases(self, step, opt_d, opt_g, local):
        yield phase
    rows.append(dict(self.records[-1]))
    if len(rows) >= a.updates:
        raise PrefixComplete()

def observed_arm(self):
    result = original_arm(self)
    arms.append(dict(outer_step=self.outer_steps+1, phase=self.phase,
                     sharp=self.row.get('critic_sharpness'),
                     base_width=self.row.get('critic_width')))
    return result

BothBoundRecorder.phases = observed_phases
SmoothedBothBoundRecorder._arm_smoothed_critic = observed_arm
runner = root / 'reports/toy100/gan_followup_probe.py'
argv = [str(runner), '--phase', 'stay', '--method', a.method,
        '--output', str(a.output / 'run')]
sys.argv = argv
try:
    runpy.run_path(str(runner), run_name='__main__')
except PrefixComplete:
    pass
finally:
    BothBoundRecorder.phases = original_phases
    SmoothedBothBoundRecorder._arm_smoothed_critic = original_arm

sources = {}
for name, module in sorted(sys.modules.items()):
    path = getattr(module, '__file__', None)
    if path and (name.startswith(('benchmarks.', 'reports.', 'particlegan.')) or name in ('torch','numpy')):
        path = Path(path).resolve()
        sources[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
selected_env = {k: v for k, v in os.environ.items() if k.startswith(
    ('ATEN_', 'MKL_', 'OMP_', 'OPENBLAS_', 'ONEDNN_', 'DNNL_', 'TORCH_', 'PYTHON', 'CUDA_', 'LD_'))}
cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
            if line.startswith('model name')), None)
libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '.so' in line and '/' in line.split()[-1]})
value = dict(repo=str(root), commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
    argv=argv, wrapper_argv=vars(a) | {'repo':str(root), 'output':str(a.output)},
    python=sys.version, executable=sys.executable, torch=torch.__version__, torch_revision=torch.version.git_version,
    cpu=cpu, capability=torch.backends.cpu.get_cpu_capability(), threads=torch.get_num_threads(),
    interop_threads=torch.get_num_interop_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
    deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
    default_dtype=str(torch.get_default_dtype()), matmul_precision=torch.get_float32_matmul_precision(),
    mkldnn_enabled=torch.backends.mkldnn.enabled, torch_config=torch.__config__.show(),
    parallel_info=torch.__config__.parallel_info(), environment=selected_env, sys_path=sys.path,
    initial=initial, records=rows, arms=arms, sources=sources, libraries=libraries)
(a.output/'prefix.json').write_text(json.dumps(value,indent=2,default=str)+'\n')
print(json.dumps(dict(output=str(a.output),records=rows,arms=arms)),flush=True)
