"""UCD score selection and discriminator-only class supervision."""
import torch
from torch import nn
from torch.nn import functional as F


def ucd_labels(labels, timestep=None, *, num_classes, target="class", num_steps=None, validate_args=True):
    """Map class labels to class-only or joint time/class output indices.

    Disable ``validate_args`` only for caller-validated label/time ranges to
    avoid bounds-check synchronization on CUDA. Shape/dtype checks remain.
    """
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("labels must be LongTensor of shape [batch]")
    if validate_args and bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("class label is out of range")
    if target == "class":
        return labels
    if target != "time_class" or timestep is None:
        raise ValueError("time_class UCD requires timesteps")
    if timestep.dtype != torch.long or timestep.shape != labels.shape or timestep.device != labels.device:
        raise ValueError("timesteps must match label shape and device and use torch.long")
    if validate_args:
        if bool((timestep < 1).any()) or (num_steps is not None and bool((timestep > num_steps).any())):
            raise ValueError("timestep is out of range")
    return (timestep - 1) * num_classes + labels


def ucd_scores(logits, labels, timestep=None, *, num_classes, target="class",
               num_steps=None, validate_args=True):
    """Select class scores from existing logits without wrapping the network.

    This is the same selection used by :class:`UCD`. It preserves gradients
    and the caller's model/checkpoint layout. Joint heads require ``num_steps``.
    """
    if target == "time_class" and (type(num_steps) is not int or num_steps < 1):
        raise ValueError("time_class requires num_steps")
    targets = ucd_labels(labels, timestep, num_classes=num_classes, target=target,
                         num_steps=num_steps, validate_args=validate_args)
    heads = num_classes * (num_steps if target == "time_class" else 1)
    if logits.shape != (len(labels), heads):
        raise ValueError(f"UCD network must return [batch, {heads}] logits")
    if logits.device != labels.device:
        raise ValueError("class labels and logits must share a device")
    return logits.gather(1, targets[:, None]).squeeze(1)


def ucd_loss(real_logits, fake_logits, targets, weight=0.02):
    """CE(real) + CE(fake), applied only in the discriminator update."""
    import math
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    return weight * (F.cross_entropy(real_logits, targets) + F.cross_entropy(fake_logits, targets))


class UCD(nn.Module):
    """Wrap a class-logit network without injecting class labels into it.

    The network accepts ``x`` for a GAN, ``x, xt=..., t=...`` for class-only
    DDGAN, or ``x, xt=...`` for joint time/class DDGAN. Returns (score, logits).
    Set ``validate_args=False`` for caller-validated label/time ranges to skip
    tensor bounds checks and their CUDA host synchronization.
    """
    def __init__(self, network, num_classes, *, target="class", num_steps=None, validate_args=True):
        super().__init__()
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError("num_classes must be positive")
        if target not in ("class", "time_class"):
            raise ValueError("target must be class or time_class")
        if target == "time_class" and (type(num_steps) is not int or num_steps < 1):
            raise ValueError("time_class requires num_steps")
        self.network, self.num_classes = network, num_classes
        self.target, self.num_steps = target, num_steps
        self.validate_args = bool(validate_args)

    def ucd_labels(self, c, t=None):
        return ucd_labels(c, t, num_classes=self.num_classes, target=self.target,
                          num_steps=self.num_steps, validate_args=self.validate_args)

    def forward(self, x, c, xt=None, t=None):
        if c is None or c.dtype != torch.long or c.shape != (len(x),):
            raise ValueError("UCD requires LongTensor class labels of shape [batch]")
        if c.device != x.device:
            raise ValueError("class labels and inputs must share a device")
        kwargs = {} if xt is None else {"xt": xt}
        if t is not None and self.target == "class":
            kwargs["t"] = t
        logits = self.network(x, **kwargs)
        scores = ucd_scores(logits, c, t, num_classes=self.num_classes,
                            target=self.target, num_steps=self.num_steps,
                            validate_args=self.validate_args)
        return scores, logits
