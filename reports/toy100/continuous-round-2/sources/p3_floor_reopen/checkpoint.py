"""Save own-state CUDA objects and stable response-history keys; no training calls."""
import torch,copy,hashlib
optimizers=[]
frames={}
def remember():
    import inspect
    frame=inspect.currentframe().f_back
    while frame:
        filename=frame.f_code.co_filename
        if '/benchmarks/' in filename or '/particlegan/' in filename or filename.startswith('<'):
            frames[(filename,frame.f_code.co_name)]=frame
        frame=frame.f_back
def save(path,response):
    modules={};streams={};trainers={};scalars={}
    for (filename,name),frame in frames.items():
        prefix=filename+':'+name+':'
        for key,value in frame.f_locals.items():
            label=prefix+key
            if isinstance(value,torch.nn.Module):modules[label]=copy.deepcopy(value.state_dict())
            elif isinstance(value,torch.Generator):streams[label]=value.get_state()
            elif value.__class__.__name__=='GANTrainer' and hasattr(value,'state_dict'):trainers[label]=copy.deepcopy(value.state_dict())
            elif type(value) in (int,float,str,bool):scalars[label]=value
    history=[]
    for (identity,group),value in response.previous.items():
        index=next(i for i,o in enumerate(optimizers) if id(o)==identity)
        history.append(dict(optimizer=index,group=group,value=value.clone()))
    payload=dict(modules=modules,trainers=trainers,streams=streams,scalars=scalars,
        optimizers=[copy.deepcopy(o.state_dict()) for o in optimizers],
        parameters=[[p.detach().clone() for g in o.param_groups for p in g['params']] for o in optimizers],
        response_history=history,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(payload,path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),modules=len(modules),trainers=len(trainers),streams=len(streams),optimizers=len(optimizers),response_history=len(history),response_devices=sorted({str(h['value'].device) for h in history}))
