"""Where a 1e-7 twin splits on the hid_q ring, and the one-step Jacobian.

Runs the same probe path as ``benchmarks.toy100.det_init_screen --gate ring
--init hid_q``. The hook records parameter groups after every update and, on
the baseline, the top eigenvalues of a random-subspace Jacobian of one joint
D/G/prior step. Twin runs add 1e-7 to one coordinate and nothing else.

    python -u reports/toy100/diag-sensitivity/amplify.py --tag base --jac
    python -u reports/toy100/diag-sensitivity/amplify.py --tag g --perturb g
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p/probe.py"
CONFIG = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p/config.json"
JAC_STEPS = (0, 4, 9, 19, 49, 99, 399, 799)
JAC_EPS = 1e-4
N_DIR = {"D": 8, "G": 8, "prior": 4}
PERTURB = 1e-7

_INSTALLED = False
_ORIG_SAMPLE = None
_ORIG_CHECKPOINT = None


def _clone(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return value


def _frame(name):
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code.co_name == name:
            return frame
        frame = frame.f_back
    return None


def _flat(tensors):
    import torch
    parts = [tensor.detach().reshape(-1).float().cpu() for tensor in tensors]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _params(module):
    return [parameter for parameter in module.parameters()]


class Recorder:
    def __init__(self, out: Path, perturb: str, jac: bool, power: bool = False):
        self.out = out
        self.perturb = perturb
        self.jac = jac
        self.power = power
        self.depth = 0
        self.calls = 0
        self.perturbed = False
        self.checked = False
        self.expect = None
        self.check_err = None
        self.rows = []
        self.jac_rows = []
        self.store = {}
        self.out.mkdir(parents=True, exist_ok=True)

    def perturb_once(self, frame):
        if self.perturbed or self.perturb in ("", "none"):
            self.perturbed = True
            return
        local = frame.f_locals
        if self.perturb == "g":
            next(local["generator"].parameters()).data.view(-1)[0] += PERTURB
        elif self.perturb == "d":
            next(local["critic"].parameters()).data.view(-1)[0] += PERTURB
        elif self.perturb == "prior":
            local["prior"].z.data.view(-1)[0] += PERTURB
        elif self.perturb != "batch":
            raise ValueError(self.perturb)
        self.perturbed = True

    def groups(self, frame):
        import torch
        local = frame.f_locals
        prior_ids = {id(parameter) for parameter in local["prior"].parameters()}
        groups = {
            "D": _params(local["critic"]),
            "G": [parameter for parameter in local["generator"].parameters() if id(parameter) not in prior_ids],
            "prior": _params(local["prior"]),
            "ema_G": list(local["ema_g"]),
            "ema_prior": [local["ema_z"]],
        }
        anchor = []
        for optimizer in (local["opt_d"], local["opt_g"]):
            for state in optimizer.state.values():
                prev = state.get("anchor_prev")
                if torch.is_tensor(prev):
                    anchor.append(prev)
        groups["latent_anchor"] = anchor
        ema = None
        try:
            import mechanism
            ema = mechanism._state.get("ema")
        except Exception:
            ema = None
        groups["ema_critic"] = list(ema) if ema else []
        return groups

    def remember(self, step, frame):
        groups = self.groups(frame)
        row = {"step": int(step)}
        for name, tensors in groups.items():
            flat = _flat(tensors)
            row[name] = float(flat.square().mean().sqrt()) if flat.numel() else 0.0
            row[name + "_n"] = int(flat.numel())
            bucket = self.store.setdefault(name, [])
            bucket.append(flat.numpy())
        self.rows.append(row)
        if step == 0 or step % 100 == 0 or step == 1200:
            print(json.dumps({"event": "state", "tag": self.out.name, "step": step,
                              **{key: row[key] for key in ("D", "G", "prior", "ema_G", "ema_critic")}}),
                  flush=True)

    def dump(self):
        import numpy as np
        arrays = {}
        ragged = []
        for name, values in self.store.items():
            if not values:
                continue
            if len({value.shape for value in values}) != 1:
                ragged.append(name)
                continue
            arrays[name] = np.stack(values)
        self.ragged = ragged
        np.savez_compressed(self.out / "traj.npz", **arrays)
        (self.out / "norms.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        payload = {"perturb": self.perturb, "check_abs": self.check_err, "jacobian": self.jac_rows,
                   "ragged": ragged, "steps": len(self.rows)}
        (self.out / "diag.json").write_text(json.dumps(payload) + "\n")
        print(json.dumps({"event": "diag_saved", "path": str(self.out), "steps": len(self.rows),
                          "check_abs": self.check_err, "jac_rows": len(self.jac_rows)}), flush=True)


_REC: Recorder | None = None


def _snap(frame):
    import torch
    local = frame.f_locals
    noise = local["noise_policy"]
    return {
        "groups": {name: [tensor.detach().clone() for tensor in tensors]
                   for name, tensors in _REC.groups(frame).items() if name in ("D", "G", "prior", "ema_G", "ema_prior")},
        "opt_d": _clone(local["opt_d"].state_dict()),
        "opt_g": _clone(local["opt_g"].state_dict()),
        "rng": torch.get_rng_state().clone(),
        "stream": local["stream"].get_state().clone(),
        "input": noise.input_stream.get_state().clone() if noise is not None else None,
        "output": noise.output_stream.get_state().clone() if noise is not None and noise.output_stream is not None else None,
    }


def _restore(frame, snap):
    import torch
    local = frame.f_locals
    live = _REC.groups(frame)
    with torch.no_grad():
        for name, saved in snap["groups"].items():
            for tensor, value in zip(live[name], saved):
                tensor.copy_(value)
        local["opt_d"].load_state_dict(_clone(snap["opt_d"]))
        local["opt_g"].load_state_dict(_clone(snap["opt_g"]))
    torch.set_rng_state(snap["rng"].clone())
    local["stream"].set_state(snap["stream"].clone())
    noise = local["noise_policy"]
    if snap["input"] is not None:
        noise.input_stream.set_state(snap["input"].clone())
    if snap["output"] is not None:
        noise.output_stream.set_state(snap["output"].clone())


def _one_step(frame, step):
    """One mode_hold update. ``set_step`` has already run; this consumes the same RNG."""
    import torch
    from contextlib import nullcontext
    from benchmarks.locked_shared.mode_hold import SIGMA
    local = frame.f_locals
    generator, critic, prior = local["generator"], local["critic"], local["prior"]
    opt_g, opt_d = local["opt_g"], local["opt_d"]
    gan, regularizer = local["gan"], local["regularizer"]
    stream, means, recipe = local["stream"], local["means"], local["recipe"]
    noise_policy, batch = local["noise_policy"], local["batch"]
    ema_g, ema_z, vicreg = local["ema_g"], local["ema_z"], local["vicreg"]
    real = _ORIG_SAMPLE(means, batch, SIGMA, stream)
    latent, _ = prior.sample(batch, generator=stream)
    context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
    with context:
        fake = generator(latent).detach()
    d_loss = gan.d_loss(critic(real), critic(fake))
    d_loss = d_loss + regularizer(critic, real, fake, step=step + 1)
    opt_d.zero_grad()
    d_loss.backward()
    from benchmarks.locked_shared import mode_hold
    mode_hold.schedule_optimizer(opt_d, step)
    opt_d.step()
    latent, _ = prior.sample(batch, generator=stream)
    fake = generator(latent)
    if gan.mode in ("rp", "ra"):
        real_g = _ORIG_SAMPLE(means, batch, SIGMA, stream)
        g_loss = gan.g_loss(critic(fake), critic(real_g))
    else:
        g_loss = gan.g_loss(critic(fake))
    g_loss = g_loss + recipe.particle_l2 * prior.z.pow(2).mean()
    g_loss = g_loss + vicreg(prior.z)
    opt_g.zero_grad()
    g_loss.backward()
    mode_hold.schedule_optimizer(opt_g, step)
    opt_g.step()
    with torch.no_grad():
        for ema, param in zip(ema_g, generator.parameters()):
            ema.mul_(recipe.ema).add_(param, alpha=1.0 - recipe.ema)
        ema_z.mul_(recipe.ema).add_(prior.z, alpha=1.0 - recipe.ema)


def _vector(frame):
    groups = _REC.groups(frame)
    return _flat(groups["D"] + groups["G"] + groups["prior"])


def _dirs(frame):
    import torch
    groups = _REC.groups(frame)
    blocks = []
    cursor = 0
    spans = {}
    pieces = []
    for name in ("D", "G", "prior"):
        flat = _flat(groups[name])
        spans[name] = (cursor, cursor + flat.numel())
        pieces.append(flat)
        cursor += flat.numel()
    full = torch.cat(pieces)
    generator = torch.Generator().manual_seed(12345)
    directions = []
    labels = []
    for name, count in N_DIR.items():
        start, end = spans[name]
        width = end - start
        draw = torch.randn(width, count, generator=generator)
        q, _ = torch.linalg.qr(draw)
        for column in range(count):
            vec = torch.zeros_like(full)
            vec[start:end] = q[:, column]
            directions.append(vec)
            labels.append(name)
    return full, directions, labels, spans


def _apply(frame, base, delta):
    import torch
    groups = _REC.groups(frame)
    cursor = 0
    with torch.no_grad():
        for name in ("D", "G", "prior"):
            for tensor in groups[name]:
                count = tensor.numel()
                chunk = base[cursor:cursor + count]
                if delta is not None:
                    chunk = chunk + delta[cursor:cursor + count]
                tensor.copy_(chunk.view_as(tensor))
                cursor += count


def _jvp(frame, step, base_vec, direction, image, eps):
    snap = _snap(frame)
    _REC.depth += 1
    try:
        _apply(frame, base_vec, eps * direction)
        _one_step(frame, step)
        out = (_vector(frame) - image) / eps
    finally:
        _REC.depth -= 1
        _restore(frame, snap)
    return out


def _shares(vector, spans):
    shares = {}
    for name, (start, end) in spans.items():
        part = vector[start:end]
        shares[name] = float(part.dot(part).sqrt())
    return shares


def _power(frame, step):
    """Top singular value of the one-step map, joint and within each block."""
    import torch
    snap = _snap(frame)
    _REC.depth += 1
    try:
        _one_step(frame, step)
        image = _vector(frame).clone()
    finally:
        _REC.depth -= 1
        _restore(frame, snap)
    base, _, _, spans = _dirs(frame)
    # _dirs draws a fixed subspace; rebuild one random direction per block from it.
    generator = torch.Generator().manual_seed(7 + int(step))
    rows = []
    for eps in (1e-4, 1e-5):
        for mask in ("joint", "D", "G", "prior"):
            vec = torch.randn(base.numel(), generator=generator)
            if mask != "joint":
                start, end = spans[mask]
                keep = vec[start:end].clone()
                vec.zero_()
                vec[start:end] = keep
            vec = vec / vec.norm().clamp_min(1e-30)
            sigma = None
            out_shares = None
            for _ in range(6):
                image_vec = _jvp(frame, step, base, vec, image, eps)
                sigma = float(image_vec.norm())
                if sigma < 1e-30:
                    break
                nxt = image_vec / sigma
                if mask != "joint":
                    start, end = spans[mask]
                    nxt = torch.zeros_like(nxt)
                    block = image_vec[start:end]
                    nxt[start:end] = block / block.norm().clamp_min(1e-30)
                vec = nxt
                out_shares = _shares(image_vec, spans)
            rows.append({"step": int(step + 1), "eps": eps, "mask": mask,
                         "singular": sigma, "out_l2": out_shares})
            print(json.dumps({"event": "power", "step": int(step + 1), "eps": eps,
                              "mask": mask, "singular": sigma, "out_l2": out_shares}), flush=True)
    _REC.jac_rows.extend(rows)


def _jacobian(frame, step):
    import torch
    snap = _snap(frame)
    _REC.depth += 1
    try:
        _one_step(frame, step)
        image = _vector(frame).clone()
        if step == 0 and _REC.expect is None:
            _REC.expect = image.clone()
        _restore(frame, snap)
        base, directions, labels, spans = _dirs(frame)
        columns = []
        for direction in directions:
            _apply(frame, base, JAC_EPS * direction)
            # Adam moments stay at the checkpoint; only parameters move.
            _one_step(frame, step)
            columns.append((_vector(frame) - image) / JAC_EPS)
            _restore(frame, snap)
    finally:
        _REC.depth -= 1
        _restore(frame, snap)
    import numpy as np
    basis = torch.stack(directions)
    # Project the image of each direction back onto the subspace.
    matrix = torch.stack([basis @ column for column in columns]).numpy()
    eig = np.linalg.eigvals(matrix)
    blocks = {}
    for name in ("D", "G", "prior"):
        idx = [i for i, label in enumerate(labels) if label == name]
        blocks[name] = np.linalg.eigvals(matrix[np.ix_(idx, idx)])
    cross = matrix.copy()
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if left == right:
                cross[i, j] = 0.0
    diagonal = matrix - cross
    def pack(values):
        order = sorted(values, key=lambda value: -abs(value))
        return [{"re": float(value.real), "im": float(value.imag), "abs": float(abs(value))}
                for value in order[:4]]
    row = {
        "step": int(step + 1),
        "eps": JAC_EPS,
        "spectral_radius": float(max(abs(value) for value in eig)),
        "block_radius": {name: float(max(abs(value) for value in values)) for name, values in blocks.items()},
        "cross_radius": float(max(abs(value) for value in np.linalg.eigvals(cross))),
        "diag_radius": float(max(abs(value) for value in np.linalg.eigvals(diagonal))),
        "top": pack(eig),
        "complex_fraction": float(np.mean(np.abs(eig.imag) > 1e-6 * np.maximum(1.0, np.abs(eig.real)))),
        "spans": {name: [int(start), int(end)] for name, (start, end) in spans.items()},
    }
    _REC.jac_rows.append(row)
    print(json.dumps({"event": "jacobian", "step": row["step"], "radius": row["spectral_radius"],
                      "blocks": row["block_radius"], "cross": row["cross_radius"],
                      "diag": row["diag_radius"], "top": row["top"][:2]}), flush=True)


def _sample_ring(means, n, sigma, generator):
    frame = _frame("train_mode_hold")
    start = frame is not None and _REC is not None and _REC.depth == 0 and _REC.calls % 2 == 0
    if start:
        step = _REC.calls // 2
        if _REC.calls == 0:
            _REC.perturb_once(frame)
            if _REC.perturb == "batch":
                # Applied to the draw below, not to parameters.
                pass
            _REC.remember(0, frame)
        if _REC.jac and step in JAC_STEPS:
            _jacobian(frame, step)
        if _REC.power and step in (2, 3, 4):
            _power(frame, step)
    _REC.calls += 1 if _REC is not None and _REC.depth == 0 else 0
    real = _ORIG_SAMPLE(means, n, sigma, generator)
    if (start and _REC.perturb == "batch" and _REC.calls == 1):
        real = real.clone()
        real.view(-1)[0] += PERTURB
    return real


def _checkpoint(step, measure):
    frame = _frame("train_mode_hold")
    if frame is not None and _REC is not None and _REC.depth == 0:
        if _REC.expect is not None and _REC.check_err is None:
            err = float((_vector(frame) - _REC.expect).abs().max())
            _REC.check_err = err
            _REC.expect = None
            print(json.dumps({"event": "replay_check", "max_abs": err}), flush=True)
        _REC.remember(step, frame)
    return _ORIG_CHECKPOINT(step, measure)


def install_hooks(out: Path, perturb: str, jac: bool, power: bool = False) -> None:
    global _INSTALLED, _ORIG_SAMPLE, _ORIG_CHECKPOINT, _REC
    if _INSTALLED:
        return
    import torch
    from benchmarks.locked_shared import mode_hold
    torch.set_num_threads(1)
    _REC = Recorder(out, perturb, jac, power)
    _ORIG_SAMPLE = mode_hold.sample_ring
    _ORIG_CHECKPOINT = mode_hold.checkpoint
    mode_hold.sample_ring = _sample_ring
    mode_hold.checkpoint = _checkpoint
    import atexit
    atexit.register(_REC.dump)
    _INSTALLED = True


def _shift_seed(offset: int) -> None:
    import torch
    if not offset:
        return

    def shift(seed):
        return (int(seed) + offset) % (2 ** 63)

    def manual_seed(seed):
        return torch.default_generator.manual_seed(shift(seed))

    torch.manual_seed = torch.random.manual_seed = manual_seed
    base = torch.Generator

    class Generator(base):
        def manual_seed(self, seed):
            return super().manual_seed(shift(seed))

    torch.Generator = Generator


def run_child(offset: int, perturb: str, jac: bool, out: Path, power: bool = False) -> None:
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    sys.path.insert(0, str(PROBE.parent))
    sys.path.insert(0, str(ROOT))
    _shift_seed(offset)
    install_hooks(out, perturb, jac, power)
    sys.argv = [str(PROBE), "--repo", str(ROOT), "--config", str(CONFIG), "--backend", "cpu",
                "--output", str(out / "probe"), "--task", "mode_hold", "--init", "hid_q"]
    try:
        runpy.run_path(str(PROBE), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code not in (0, None):
            raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--perturb", default="none", choices=("none", "g", "d", "prior", "batch"))
    parser.add_argument("--jac", action="store_true")
    parser.add_argument("--power", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    run_child(args.offset, args.perturb, args.jac, args.out, args.power)


if __name__ == "__main__":
    main()
