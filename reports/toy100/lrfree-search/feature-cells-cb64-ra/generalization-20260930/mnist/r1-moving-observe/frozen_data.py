import math
import torch

def angle(step):
    # target rotation (radians) in force for the update that completes `step` (step 0 = before training)
    return 0. if args.rotate_every <= 0 else math.radians(args.rotate_deg) * ((max(step, 1) - 1) // args.rotate_every)

def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(((c, -s), (s, c)), device=device)

def real_batch(step):
    x = problems.sample_real(args.task, batch, device=device, generator=stream)
    return x if args.rotate_every <= 0 else x @ rotation(angle(step)).T

