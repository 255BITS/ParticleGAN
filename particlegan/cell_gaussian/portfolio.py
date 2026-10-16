"""Fixed portfolio mechanisms; no evaluator inputs or external randomness."""
import torch
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def minimum_switch(cost, slots, previous):
    expanded=cost[:,slots].double()
    span=expanded.max()-expanded.min()
    scaled=(expanded-expanded.min())/span if float(span)>0 else torch.zeros_like(expanded)
    changed=previous[:,None]!=slots[None,:]
    _,columns=linear_sum_assignment((scaled+(len(cost)+1)*changed).cpu().numpy())
    assignment=slots[torch.as_tensor(columns,device=slots.device)]
    required=int((torch.bincount(previous,minlength=cost.shape[1])-torch.bincount(slots,minlength=cost.shape[1])).abs().sum())//2
    actual=int((assignment!=previous).sum())
    if actual!=required:raise ValueError('Minimum-switch solver missed exact lower bound')
    return assignment,dict(required_assignment_switches=required,actual_minimum_switches=actual)

def project_vectors(original):
    result=[];conflicts=0
    for i,gradient in enumerate(original):
        gradient=gradient.clone()
        for j,other in enumerate(original):
            if i==j:continue
            dot=torch.dot(gradient,other);norm=torch.dot(other,other)
            if float(dot)<0 and float(norm)>0:
                gradient=gradient-dot/norm*other;conflicts+=1
        result.append(gradient)
    return torch.stack(result).sum(0),conflicts

def projected_backward(noncell,terms,optimizer):
    params=[p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    def flatten(loss):
        values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,values)])
    ordinary=flatten(noncell);gradients=[flatten(term) for term in terms]
    combined,conflicts=project_vectors(gradients);combined=combined+ordinary
    offset=0
    for p in params:
        p.grad=combined[offset:offset+p.numel()].view_as(p).detach().clone();offset+=p.numel()
    return conflicts

def extragradient_step(trainer,real,generator_real,collect_stats):
    # Cache any caller-owned data callback once. Every internal evaluation sees
    # the same real tensors and starts from the same optimizer/history/streams.
    batch=generator_real() if callable(generator_real) else generator_real
    original=trainer.state_dict()
    trainer._step(real,generator_real=batch,collect_stats=collect_stats)
    predicted=trainer.state_dict()['models']
    trainer.load_state_dict(original)
    for name,model in predicted.items():getattr(trainer,name).load_state_dict(model)
    result=trainer._step(real,generator_real=batch,collect_stats=collect_stats)
    def correction(before,predictor,after):
        if isinstance(after,torch.Tensor):
            return before+(after-predictor) if after.is_floating_point() else after
        if isinstance(after,dict):return {k:correction(before[k],predictor[k],v) for k,v in after.items()}
        if isinstance(after,(list,tuple)):return type(after)(correction(a,b,c) for a,b,c in zip(before,predictor,after))
        return after
    with torch.no_grad():
        for name,before in original['models'].items():
            module=getattr(trainer,name)
            module.load_state_dict(correction(before,predicted[name],module.state_dict()))
    trainer.cell_objective.last['public_map_evaluations_per_step']=2
    result['public_map_evaluations']=2
    return result
