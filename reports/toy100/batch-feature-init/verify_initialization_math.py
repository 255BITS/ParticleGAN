"""Fixed-tensor algebra checks for the guide; no training or seed experiments."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from particlegan import qr_bz_pq_init as ortho
rng_before=torch.get_rng_state().clone()
checks={}


def max_error(a,b):
    error=float((a.detach()-b.detach()).abs().max())
    assert error<1e-11,error
    return error


matrix_checks=[]
for m,n in [(8,3),(3,8),(5,5),(7,27)]:
    q=ortho.semi_orthogonal(ortho._key('guide_matrix',m,n),m,n)
    rho=1/math.sqrt(3*n)
    gain=rho*math.sqrt(max(m,n))
    w=gain*q
    gram=w.T@w if m>=n else w@w.T
    matrix_checks.append(dict(shape=[m,n],
        gram_error=max_error(gram,gain**2*torch.eye(min(m,n))),
        rms_error=max_error(w.square().mean(),torch.tensor(rho**2))))
checks['linear_and_flattened_kernel']=matrix_checks

bias=ortho.pattern_bias(ortho._key('guide_bias'),(7,),('uniform',-.3,.3))
checks['pattern_bias']=dict(mean_error=max_error(bias.mean(),torch.tensor(0.)),
                          variance_error=max_error(bias.var(unbiased=False),torch.tensor(.3**2/3)))

k=torch.tensor([[1.,1.]])/math.sqrt(2)
operator=torch.tensor([[1.,1.,0.],[0.,1.,1.]])/math.sqrt(2)
checks['convolution_counterexample']=dict(
    kernel_gram_error=max_error(k@k.T,torch.ones(1,1)),
    spatial_gram_error=max_error(operator@operator.T,torch.tensor([[1.,.5],[.5,1.]])),
    spatial_gram_is_not_identity=not torch.allclose(operator@operator.T,torch.eye(2)))
assert checks['convolution_counterexample']['spatial_gram_is_not_identity']

# A neutral pre-normalized residual output has an identity input Jacobian,
# a trainable output map, and initially zero task-loss gradient in its interior.
d=5
x=torch.linspace(-1,1,d).requires_grad_(True)
u=(torch.arange(d*d).reshape(d,d).sin()*.3).requires_grad_(True)
c=torch.zeros(d,d,requires_grad=True)
def residual(v):return v+c@torch.tanh(u@F.layer_norm(v,(d,)))
y=residual(x)
jac=torch.autograd.functional.jacobian(residual,x)
loss=(y-torch.linspace(.2,.6,d)).square().sum()/2
dc,du=torch.autograd.grad(loss,(c,u))
assert float(dc.abs().max())>1e-6
checks['neutral_residual']=dict(output_error=max_error(y,x),
    input_jacobian_error=max_error(jac,torch.eye(d)),
    interior_gradient_error=max_error(du,torch.zeros_like(du)),
    output_map_gradient_max=float(dc.abs().max()))

# The LoRA equalities apply to the defined linear adapter, independent of the
# choice of deterministic nonzero A. The projector formula uses semi-orthogonal A.
di,do,r,n=7,5,2,4
q=ortho.semi_orthogonal(ortho._key('guide_lora',r,di),r,di)
gain=1/math.sqrt(3)
a=(gain*q).requires_grad_(True)
b=torch.zeros(do,r,requires_grad=True)
w0=torch.arange(do*di).reshape(do,di).cos()*.1
inputs=torch.arange(di*n).reshape(di,n).sin()
target=torch.arange(do*n).reshape(do,n).cos()*.2
scale=2.
outputs=(w0+scale*b@a)@inputs
loss=(outputs-target).square().sum()/2
db,da=torch.autograd.grad(loss,(b,a))
g=(outputs.detach()-target)@inputs.T
eta=.03
next_b=b.detach()-eta*db
next_a=a.detach()-eta*da
projector=q.T@q
checks['lora']=dict(
    no_op_error=max_error(outputs,w0@inputs),
    b_gradient_error=max_error(db,scale*g@a.detach().T),
    a_gradient_error=max_error(da,torch.zeros_like(da)),
    b_gradient_max=float(db.abs().max()),
    projector_error=max_error(projector@projector,projector),
    first_sgd_update_error=max_error(scale*next_b@next_a,-eta*scale**2*gain**2*g@projector))
assert checks['lora']['b_gradient_max']>1e-6

zero_a=torch.zeros_like(a,requires_grad=True)
zero_b=torch.zeros_like(b,requires_grad=True)
zero_outputs=(w0+scale*zero_b@zero_a)@inputs
zero_loss=(zero_outputs-target).square().sum()/2
zdb,zda=torch.autograd.grad(zero_loss,(zero_b,zero_a))
checks['both_lora_factors_zero']=dict(a_gradient_max=max_error(zda,torch.zeros_like(zda)),
                                     b_gradient_max=max_error(zdb,torch.zeros_like(zdb)))
assert torch.equal(rng_before,torch.get_rng_state())
checks['torch_rng_unchanged']=True
checks['scope']='Algebra checks only. No transformer/LoRA training, no added benchmark passes.'
path=Path(__file__).resolve().parent/'initialization-math-checks.json'
path.write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
