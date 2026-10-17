"""Read-only tensor hash shared by the focused saved production contract."""
import hashlib
import torch


def tensor_state_hash(obj):
    h=hashlib.sha256()
    def walk(value):
        if isinstance(value,torch.Tensor):
            h.update(str((value.dtype,value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):
                h.update(str(key).encode());walk(value[key])
        elif isinstance(value,(list,tuple)):
            for part in value:walk(part)
        else:h.update(repr(value).encode())
    walk(obj);return h.hexdigest()
