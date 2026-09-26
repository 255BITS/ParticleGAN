"""Sample-splice hook for the K3P ring probe.

Installed only when ``K3P_SPLICE_HOOK`` names this module. The probe's
recipe, coefficients, learning rates and clips stay as they are. This
module records or replays the randomness that still depends on the seed
under ``--init hid_q``:

* data mode index and jitter (the two ``sample_ring`` draws per update)
* particle-prior indices (critic draw, then generator draw)
* critic input noise and generator output noise
* evaluation prior indices and evaluation output noise

Replay substitutes those tensors and leaves every other update unchanged.
``K3P_SPLICE_SWITCH`` is the first 0-based update that reads tape B;
updates ``[0, switch)`` read tape A. ``K3P_SPLICE_WINDOW=a:b`` instead
reads tape B only on updates ``[a, b)``. ``K3P_SPLICE_STREAMS`` limits
which of those streams move; the rest stay on tape A.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

GROUPS = {
    "data": {"data_d_idx", "data_d_eps", "data_g_idx", "data_g_eps"},
    "prior": {"prior_d", "prior_g"},
    "output": {"output_train"},
    "input": {"input_train"},
    "noise": {"output_train", "input_train"},
    "eval": {"prior_eval", "output_eval", "input_eval"},
    "train": {
        "data_d_idx", "data_d_eps", "data_g_idx", "data_g_eps",
        "prior_d", "prior_g", "output_train", "input_train",
    },
}
STREAM_NAMES = sorted(set().union(*GROUPS.values()))
_INSTALLED = False
_CTRL = None


def resolve_streams(spec: str) -> set[str] | None:
    """``all`` switches every stream. Otherwise a comma list of groups or names."""
    text = spec.strip()
    if text == "all":
        return None
    chosen = set()
    for part in text.split(","):
        name = part.strip()
        if not name:
            continue
        if name in GROUPS:
            chosen |= GROUPS[name]
        elif name in STREAM_NAMES:
            chosen.add(name)
        else:
            known = ", ".join(["all", *GROUPS, *STREAM_NAMES])
            raise ValueError(f"unknown splice stream {name!r}; expected one of {known}")
    if not chosen:
        raise ValueError("splice streams are empty")
    return chosen


def parse_window(text: str | None) -> tuple[int, int] | None:
    if text is None or text == "":
        return None
    left, right = text.split(":", 1)
    start, stop = int(left), int(right)
    if start < 0 or stop < start:
        raise ValueError("window must satisfy 0 <= start <= stop")
    return start, stop


class Controller:
    """Per-process record/replay cursor. One training run, then ``finish``."""

    def __init__(self):
        self.mode = os.environ.get("K3P_SPLICE_MODE", "record")
        if self.mode not in ("record", "replay"):
            raise ValueError("K3P_SPLICE_MODE must be record or replay")
        self.directory = Path(os.environ["K3P_SPLICE_DIR"])
        self.switch = int(os.environ.get("K3P_SPLICE_SWITCH", "1000000000"))
        self.window = parse_window(os.environ.get("K3P_SPLICE_WINDOW"))
        self.selected = resolve_streams(os.environ.get("K3P_SPLICE_STREAMS", "all"))
        self.dense = int(os.environ.get("K3P_SPLICE_DENSE", "0"))
        self.suspend = False
        self.step = 0
        self.data_calls = 0
        self.prior_calls = 0
        self.train_batch = None
        self.tape = {}
        self.tapes = {}
        self.cursors = {}
        self.curve_path = self.directory / "splice-curve.jsonl"
        if self.mode == "replay":
            self.tapes["a"] = _load_tape(os.environ["K3P_SPLICE_TAPE_A"])
            self.tapes["b"] = _load_tape(os.environ["K3P_SPLICE_TAPE_B"])
        print(json.dumps({
            "event": "splice_install", "mode": self.mode, "switch": self.switch,
            "window": self.window, "streams": "all" if self.selected is None else sorted(self.selected),
            "dense": self.dense,
        }), flush=True)

    def source(self, step: int, stream: str) -> str:
        if self.selected is not None and stream not in self.selected:
            return "a"
        if self.window is not None:
            start, stop = self.window
            return "b" if start <= step < stop else "a"
        return "b" if step >= self.switch else "a"

    def _bucket(self, step: int) -> dict:
        return self.tape.setdefault(step, {})

    def save(self, stream: str, value: torch.Tensor) -> None:
        bucket = self._bucket(self.step).setdefault(stream, [])
        bucket.append(value.detach().cpu().clone())

    def pop(self, stream: str, like: torch.Tensor) -> torch.Tensor:
        src = self.source(self.step, stream)
        key = (src, self.step, stream)
        seq = self.tapes[src]["map"].get(self.step, {}).get(stream, [])
        cursor = self.cursors.get(key, 0)
        if cursor >= len(seq):
            raise RuntimeError(
                f"splice underflow step={self.step} stream={stream} source={src} "
                f"used={cursor} have={len(seq)}")
        self.cursors[key] = cursor + 1
        return seq[cursor].to(device=like.device, dtype=like.dtype)

    def log(self, row: dict) -> None:
        with self.curve_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def finish(self) -> None:
        if self.mode == "record":
            payload = {
                "version": 1,
                "steps": self.data_calls // 2,
                "data_calls": self.data_calls,
                "prior_calls": self.prior_calls,
                "tape": {str(step): streams for step, streams in sorted(self.tape.items())},
            }
            path = self.directory / "splice-tape.pt"
            torch.save(payload, path)
            counts = {}
            for streams in self.tape.values():
                for name, values in streams.items():
                    counts[name] = counts.get(name, 0) + len(values)
            print(json.dumps({"event": "splice_recorded", "steps": payload["steps"],
                              "counts": counts, "tape": str(path)}), flush=True)
            return
        missing = []
        steps = max(self.tapes["a"]["steps"], self.tapes["b"]["steps"])
        names = STREAM_NAMES
        for step in range(steps):
            for stream in names:
                src = self.source(step, stream)
                have = len(self.tapes[src]["map"].get(step, {}).get(stream, []))
                used = self.cursors.get((src, step, stream), 0)
                if used != have:
                    missing.append({"step": step, "stream": stream, "source": src,
                                    "used": used, "have": have})
        if missing:
            preview = missing[:8]
            raise RuntimeError(f"splice cursor mismatch {len(missing)} streams, first {preview}")
        print(json.dumps({"event": "splice_replay_ok", "steps": steps}), flush=True)


def _load_tape(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != 1:
        raise ValueError(f"unsupported splice tape {path}")
    return {
        "steps": int(payload["steps"]),
        "map": {int(step): streams for step, streams in payload["tape"].items()},
    }


def _note_data(ctrl: Controller) -> str:
    which = "d" if ctrl.data_calls % 2 == 0 else "g"
    if which == "d":
        ctrl.step = ctrl.data_calls // 2
    ctrl.data_calls += 1
    return which


def _patch_sample_ring(ctrl: Controller) -> None:
    import benchmarks.locked_shared.mode_hold as mode_hold

    original = mode_hold.sample_ring

    def sample_ring(means, n, sigma, generator):
        which = _note_data(ctrl)
        if ctrl.train_batch is None:
            ctrl.train_batch = int(n)
        idx_name = f"data_{which}_idx"
        eps_name = f"data_{which}_eps"
        if ctrl.mode == "replay" and not ctrl.suspend:
            idx = ctrl.pop(idx_name, means.new_empty(0, dtype=torch.long))
            eps = ctrl.pop(eps_name, means)
            return means[idx.long()] + float(sigma) * eps
        captured = {}
        real_randint = torch.randint
        real_randn = torch.randn

        def randint(*args, **kwargs):
            value = real_randint(*args, **kwargs)
            captured["idx"] = value
            return value

        def randn(*args, **kwargs):
            value = real_randn(*args, **kwargs)
            captured["eps"] = value
            return value

        torch.randint = randint
        torch.randn = randn
        try:
            points = original(means, n, sigma, generator)
        finally:
            torch.randint = real_randint
            torch.randn = real_randn
        if not ctrl.suspend and ctrl.mode == "record":
            if "idx" not in captured or "eps" not in captured:
                raise RuntimeError("sample_ring did not draw an index and a jitter")
            ctrl.save(idx_name, captured["idx"])
            ctrl.save(eps_name, captured["eps"])
        return points

    mode_hold.sample_ring = sample_ring


def _patch_prior(ctrl: Controller) -> None:
    from particlegan.particle_prior import ParticlePrior

    original = ParticlePrior.sample

    def sample(self, batch_size, generator=None, **kwargs):
        train = (ctrl.train_batch is not None and batch_size == ctrl.train_batch
                 and not kwargs.get("fixed_first_n"))
        if train:
            stream = "prior_d" if ctrl.prior_calls % 2 == 0 else "prior_g"
            ctrl.prior_calls += 1
        else:
            stream = "prior_eval"
        if ctrl.mode == "replay" and not ctrl.suspend:
            idx = ctrl.pop(stream, self.z.new_empty(0, dtype=torch.long)).long()
            return self.z[idx], idx
        latent, idx = original(self, batch_size, generator=generator, **kwargs)
        if not ctrl.suspend and ctrl.mode == "record":
            ctrl.save(stream, idx)
        return latent, idx

    ParticlePrior.sample = sample


def _patch_noise(ctrl: Controller) -> None:
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    original_output = NoisePolicy.output
    original_input = NoisePolicy.input

    def _intercept(draw, stream):
        def wrapped(*args, **kwargs):
            value = draw(*args, **kwargs)
            if ctrl.suspend or ctrl.mode == "record":
                if not ctrl.suspend and ctrl.mode == "record":
                    ctrl.save(stream, value)
                return value
            return ctrl.pop(stream, value)
        return wrapped

    def output(self, generated, *args, **kwargs):
        stream = "output_eval" if self._evaluating else "output_train"
        real_like = torch.randn_like
        real_randn = torch.randn
        torch.randn_like = _intercept(real_like, stream)
        torch.randn = _intercept(real_randn, stream)
        try:
            return original_output(self, generated, *args, **kwargs)
        finally:
            torch.randn_like = real_like
            torch.randn = real_randn

    def input(self, data, *args, **kwargs):
        stream = "input_eval" if self._evaluating else "input_train"
        real_randn = torch.randn
        torch.randn = _intercept(real_randn, stream)
        try:
            return original_input(self, data, *args, **kwargs)
        finally:
            torch.randn = real_randn

    NoisePolicy.output = output
    NoisePolicy.input = input


def _patch_checkpoint(ctrl: Controller) -> None:
    import benchmarks.locked_shared.mode_hold as mode_hold
    from benchmarks.toy100.device import rng_fork_devices

    original = mode_hold.checkpoint

    def log_row(step, row, kind):
        item = {"event": "obs", "kind": kind, "step": int(step),
                "modes": int(row["modes"]), "hq": float(row["hq"])}
        if "missing_modes" in row:
            item["missing_modes"] = list(row["missing_modes"])
        if "hq_counts" in row:
            item["hq_counts"] = list(row["hq_counts"])
        ctrl.log(item)

    def checkpoint(step, measure):
        def wrapped():
            row = measure()
            if not ctrl.suspend:
                log_row(step, row, "gate")
            return row

        original(step, wrapped)
        if ctrl.dense and step % ctrl.dense == 0:
            ctrl.suspend = True
            try:
                with torch.random.fork_rng(devices=rng_fork_devices()):
                    row = measure()
                log_row(step, row, "dense")
            finally:
                ctrl.suspend = False

    mode_hold.checkpoint = checkpoint


def _patch_train(ctrl: Controller) -> None:
    import benchmarks.locked_shared.mode_hold as mode_hold

    original = mode_hold.train_mode_hold

    def train_mode_hold(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            ctrl.finish()

    mode_hold.train_mode_hold = train_mode_hold


def install() -> None:
    """Patch the ring host for this process. Default training never calls this."""
    global _INSTALLED, _CTRL
    if _INSTALLED:
        return
    _CTRL = Controller()
    _patch_sample_ring(_CTRL)
    _patch_prior(_CTRL)
    _patch_noise(_CTRL)
    _patch_checkpoint(_CTRL)
    _patch_train(_CTRL)
    _INSTALLED = True
