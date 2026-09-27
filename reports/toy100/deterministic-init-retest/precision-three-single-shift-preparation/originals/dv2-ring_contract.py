"""Frozen public ring construction. Imported code performs no construction."""
import hashlib

def construct(torch,package,host,recipe,device):
    assert str(torch.get_default_device())=='cpu'
    generator=host.SimpleMLPGenerator(recipe.z_dim,96,3,2).to(device)
    critic=host.SimpleMLPDiscriminator(2,96,3,3).to(device)
    return package.GANTrainer(recipe,generator,critic,seed=0,
        optimizer_options={'foreach':False,'fused':False})

def material(torch,trainer):
    def visit(value):
        if isinstance(value,torch.Tensor):
            t=value.detach().cpu().contiguous()
            return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if isinstance(value,dict):return {str(k):visit(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [visit(v) for v in value]
        if value is None or isinstance(value,(str,int,float,bool)):return value
        raise TypeError(type(value))
    state=trainer.state_dict()
    for key in ('streams','cpu_rng','cuda_rng','device'):state.pop(key,None)
    modules={}
    for role in ('G','D','prior','ema_G','ema_D','ema_prior'):
        m=getattr(trainer,role)
        modules[role]=dict(parameters=dict(m.named_parameters()),buffers=dict(m.named_buffers()))
    return visit(dict(state=state,all_parameters_and_buffers=modules))
