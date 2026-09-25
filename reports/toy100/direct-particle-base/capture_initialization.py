"""Capture missing pinned CPU constructors only; forbid CPU autograd and updates."""
import pathlib,sys,hashlib,json
root=pathlib.Path(__file__).resolve().parent
source=(root/'candidates/direct_particle_response/probe.py').read_text()
needle="        dict(shape=list(v.shape), sha256=digest(v)) for v in values])"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n    if a.init_only and len(initial_values) == 2:\n        raise InitializationCaptured()")
source=source.replace("import mechanism\n", "def forbid_cpu_gradients(*args, **kwargs):\n    raise RuntimeError('Initialization capture reached autograd')\ntorch.Tensor.backward = forbid_cpu_gradients\ntorch.autograd.grad = forbid_cpu_gradients\nimport mechanism\n")
sys.path.insert(0,str(root/'candidates/direct_particle_response'))
exec(compile(source,str(__file__),'exec'))
