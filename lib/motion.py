"""HumanAct12 prefix completion: subject splits, original-coordinate motion metrics.

Arrays are [frames, 24 joints, xyz]. One translation anchors the last observed
root; later root movement is preserved. Scalar normalization uses training
subjects only. No skeleton projection or reconstruction loss is applied.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

CHAINS = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15],
          [9, 13, 16, 18, 20, 22], [9, 14, 17, 19, 21, 23]]
EDGES = [(a, b) for chain in CHAINS for a, b in zip(chain, chain[1:])]
ACTIONS = ['warm up', 'walk', 'run', 'jump', 'drink', 'lift dumbbell',
           'sit', 'eat', 'steering wheel', 'phone', 'boxing', 'throw']
NAME = re.compile(r'P(\d+)G(\d+)R(\d+)F(\d+)T(\d+)A(\d{2})(\d{2})\.npy')


def identify(name):
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError(f'Unrecognized source name: {name}')
    p, g, r, first, last, action, sub = map(int, match.groups())
    return dict(subject=p, recording=f'P{p:02}G{g:02}R{r:02}',
                first=first, last=last, action=action-1, subaction=sub)


def normalize_window(window, past, scale):
    """Only observed values set the translation; future cannot affect context."""
    return (window - window[past-1, 0]) / scale


class MotionData:
    def __init__(self, cfg, device='cpu'):
        self.cfg, self.device = cfg, device
        self.past, self.future = cfg['past_length'], cfg['length']
        self.window = self.past + self.future
        sets = {s: set(cfg[s+'_subjects']) for s in ('train', 'validation', 'test')}
        if any(sets[a] & sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
            raise ValueError('Subjects must be disjoint before windowing')
        files = sorted(Path(cfg['data_dir']).glob('*.npy'))
        if len(files) != cfg['expected_clips']:
            raise ValueError(f'Expected {cfg["expected_clips"]} files, found {len(files)}. Run prepare_motion.py first.')
        arrays, rows, excluded = [], [], []
        square_sum, coord_count = 0., 0
        hashes = set()
        for path in files:
            row = {**identify(path.name), 'name': path.name,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            split = next((s for s, ids in sets.items() if row['subject'] in ids), None)
            if split is None:
                raise ValueError(f'Unassigned subject: {path.name}')
            x = np.load(path, allow_pickle=False).astype(np.float32)
            if x.ndim != 3 or x.shape[1:] != (24, 3) or not np.isfinite(x).all():
                raise ValueError(f'Invalid motion: {path}')
            digest = hashlib.sha256(x.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f'Duplicate motion array: {path}')
            hashes.add(digest)
            row.update(split=split, frames=len(x))
            if len(x) < self.window:
                excluded.append(row)
                continue
            if split == 'train':
                relative_pose = x - x[:, :1]
                square_sum += float(np.square(relative_pose, dtype=np.float64).sum())
                coord_count += relative_pose.size
            rows.append(row)
            arrays.append(x)
        self.scale = float(np.sqrt(square_sum / coord_count))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid training scale')
        self.rows, self.arrays = rows, arrays
        offsets = np.cumsum([0] + [len(x) for x in arrays[:-1]])
        self.frames = torch.tensor(np.concatenate(arrays), device=device)
        self.offsets = torch.tensor(offsets, device=device)
        self.lengths = torch.tensor([len(x) for x in arrays], device=device)
        self.labels = torch.tensor([r['action'] for r in rows], device=device)
        self.train_groups = [torch.tensor([i for i, r in enumerate(rows)
                             if r['split'] == 'train' and r['action'] == c], device=device) for c in range(12)]
        if any(len(g) == 0 for g in self.train_groups):
            raise ValueError('Every action must have training clips')
        self.group_sizes = torch.tensor([len(g) for g in self.train_groups], device=device)
        self.group_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), self.group_sizes.cumsum(0)[:-1]])
        self.group_indices = torch.cat(self.train_groups)
        self.manifest = dict(source='https://github.com/EricGuo5513/action-to-motion',
                             split_protocol='custom subject holdout, NOT an official benchmark split',
                             normalization='one translation at last observed root / train-only root-relative pose RMS',
                             scale=self.scale, coordinate_units='source units; reported metrics use normalized units',
                             past=self.past, future=self.future, rows=rows, excluded_short=excluded,
                             counts={s: {ACTIONS[c]: sum(r['split']==s and r['action']==c for r in rows)
                                         for c in range(12)} for s in sets})
        self.manifest['source_fingerprint'] = hashlib.sha256(json.dumps(sorted(
            (r['name'], r['sha256']) for r in rows + excluded)).encode()).hexdigest()
        self.manifest['fingerprint'] = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()

    def window_batch(self, ids, starts):
        positions = self.offsets[ids, None] + starts[:, None] + torch.arange(self.window, device=self.device)
        x = self.frames[positions]
        x = (x - x[:, self.past-1:self.past, :1]) / self.scale
        past = x[:, :self.past].reshape(len(x), self.past, 72).transpose(1, 2).contiguous()
        future = x[:, self.past:].reshape(len(x), self.future, 72).transpose(1, 2).contiguous()
        return self.labels[ids], past.flatten(1), future

    def batch(self, n, rng):
        c = torch.randint(12, (n,), device=self.device, generator=rng)
        k = (torch.rand(n, device=self.device, generator=rng) * self.group_sizes[c]).long()
        ids = self.group_indices[self.group_offsets[c]+k]
        starts = (torch.rand(n, device=self.device, generator=rng) * (self.lengths[ids]-self.window+1)).long()
        return self.window_batch(ids, starts)

    def evaluation(self, split, per_class):
        ids = []
        for c in range(12):
            choices = [i for i, r in enumerate(self.rows) if r['split']==split and r['action']==c]
            if not choices:
                raise ValueError(f'No evaluation clips: {split} {ACTIONS[c]}')
            ids.extend(choices[i] for i in np.linspace(0, len(choices)-1, min(per_class, len(choices)), dtype=int))
        index = torch.tensor(ids, device=self.device)
        starts = (self.lengths[index]-self.window)//2
        c, context, real = self.window_batch(index, starts)
        return c, context, real, [dict(name=self.rows[i]['name'], start=int(s)) for i, s in zip(ids, starts.cpu())]


def joints(x):
    """[..., channels, frames] -> [..., frames, joints, xyz]."""
    return x.transpose(-1, -2).reshape(*x.shape[:-2], x.shape[-1], 24, 3)


@torch.no_grad()
def motion_metrics(samples, real, context, past_length, labels):
    """samples [contexts,K,72,T]; single recorded future per context.

    Spatial metrics are normalized source coordinates, dynamics per frame.
    Class SW1 is a pooled marginal diagnostic, not conditional mode coverage.
    """
    from lib.toy_metrics import sliced_w1
    x, r = joints(samples), joints(real)
    past = joints(context.reshape(len(context), 72, past_length))
    n, k = x.shape[:2]
    errors = (x-r[:, None]).norm(dim=-1).mean(-1)
    full = torch.cat([past[:, None].expand(-1, k, -1, -1, -1), x], 2)
    velocity = full[:, :, 1:] - full[:, :, :-1]
    speed = velocity[:, :, past_length-1:].norm(dim=-1)
    accel = (velocity[:, :, 1:] - velocity[:, :, :-1])[:, :, past_length-2:].norm(dim=-1)
    a, b = zip(*EDGES)
    bone = (x[..., a, :] - x[..., b, :]).norm(dim=-1)
    observed_bone = (past[:, -1, a, :] - past[:, -1, b, :]).norm(dim=-1)
    rel_bone = (bone / observed_bone[:, None, None].clamp_min(1e-4) - 1).abs()
    diversity = torch.stack([(x[:, i]-x[:, j]).norm(dim=-1).mean((1, 2))
                             for i in range(k) for j in range(i)]) if k > 1 else x.new_zeros(1, n)
    rows = []
    for c in labels.unique().tolist():
        mask = labels == c
        # Exactly one generated draw per prefix for equal-size marginal SW1.
        sw = sliced_w1(samples[mask, 0].flatten(1), real[mask].flatten(1), 128, seed=31415)
        rows.append(dict(action=ACTIONS[c], contexts=int(mask.sum()), sw1=sw,
                         ade=float(errors[mask].mean()), diversity=float(diversity[:, mask].mean())))
    boundary = (x[:, :, 0] - past[:, None, -1]).norm(dim=-1).mean()
    boundary_accel = (x[:, :, 0] - 2*past[:, None, -1] + past[:, None, -2]).norm(dim=-1).mean()
    return dict(contexts=n, samples_per_context=k, single_ade=float(errors[:, 0].mean()),
                mean_ade=float(errors.mean()), best_of_k_ade=float(errors.mean(-1).min(1).values.mean()),
                single_fde=float(errors[:, 0, -1].mean()),
                best_of_k_fde=float(errors[:, :, -1].min(1).values.mean()),
                diversity=float(diversity.mean()), bone_relative_error=float(rel_bone.mean()),
                bone_bad_fraction=float((rel_bone > .2).float().mean()),
                speed=float(speed.mean()), acceleration=float(accel.mean()),
                boundary_displacement=float(boundary), boundary_acceleration=float(boundary_accel),
                class_sw1=float(np.mean([r['sw1'] for r in rows])), actions=rows)


def baselines(real, context, past_length):
    past = context.reshape(len(context), 72, past_length)
    held = past[:, :, -1:].expand_as(real)
    time = torch.arange(1, real.shape[-1]+1, device=real.device)
    cv = held + (past[:, :, -1:]-past[:, :, -2:-1]) * time
    return {'held_pose': held, 'constant_velocity': cv, 'recorded': real}
