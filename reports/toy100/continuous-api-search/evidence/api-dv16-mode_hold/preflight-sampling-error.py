from worker import ROOT,digest
import json,time,torch
from pathlib import Path
from particlegan import GANTrainer,get_recipe
from benchmarks.locked_shared.mlp import SimpleMLPGenerator,SimpleMLPDiscriminator
r=ROOT/'reports/data-drift-api';started=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
recipe=get_recipe(total_steps=None,continuous_policy='dv16',num_particles=12,z_dim=4,batch_size=16,input_noise_std=0.,output_noise_warmup=0.)
torch.manual_seed(0);t=GANTrainer(recipe,SimpleMLPGenerator(4,12,1,2),SimpleMLPDiscriminator(2,12,1,3),serial_backward=True)
assert t.penalty.regularizer.continuous_controller is t.controller
assert t.prior.shape_shear.count_nonzero()==0
real=torch.randn(16,2);state=t.state_dict();t.step(real);a=t.state_dict();t.load_state_dict(state);t.step(real);b=t.state_dict()
# Restoring state also restores global RNG; real tensor is unchanged.
assert digest(a)==digest(b)
assert t.penalty.regularizer.continuous_controller is t.controller
assert t.prior.shape_shear.grad is not None and torch.isfinite(t.prior.shape_shear.grad).all()
assert t.prior.shape_shear.count_nonzero()>0
before=digest(t.state_dict());t.sample(100);t.sample(100,ema=True);assert digest(t.state_dict())==before
row=dict(candidate='regression',gate='dv16_shape_gradient_binding_restore_isolation',status='PASS',seconds=time.monotonic()-started,metrics={'checks':7,'scope':'CPU contracts only'},artifact=str(Path(__file__).resolve()))
with (ROOT.parent/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row),flush=True)
