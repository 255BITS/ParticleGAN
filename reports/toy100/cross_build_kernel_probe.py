"""Attribute one cross-build divergence to an ATen op.

Same constant-rate mode-hold factory as the cold ring and the stay gate.
``--step 0`` records initialization ops until the first ``set_step(0)``.
``--step K`` records ops only during update K, then stops. The dispatch
mode is not entered on any other update, so earlier updates stay on the
gate path.

Compare ``ops.jsonl`` across builds. The first row whose op or output hash
differs is the kernel. This does not change the candidate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from reports.toy100.gan_followup_probe import ROOT, build_fingerprint, emit, factory


class StopAtStep(Exception):
    pass


def _tensor_row(value):
    import torch
    if isinstance(value, torch.Tensor):
        if not (value.is_floating_point or value.is_complex):
            return None
        array = value.detach().contiguous().cpu().numpy()
        reduced = float(array.astype("float64", copy=False).sum()) if array.size else 0.0
        return dict(hash=hashlib.sha256(array.tobytes()).hexdigest(),
                    shape=list(array.shape), dtype=str(array.dtype), numel=int(array.size),
                    sum=reduced)
    if isinstance(value, (list, tuple)) and value:
        parts = [part for part in (_tensor_row(item) for item in value) if part is not None]
        return parts or None
    return None


def _particle_hash():
    frame = sys._getframe(1)
    while frame is not None:
        local = frame.f_locals
        prior = local.get("prior") if isinstance(local, dict) else None
        if prior is not None and hasattr(prior, "z"):
            return frame.f_code.co_name, _tensor_row(prior.z)
        frame = frame.f_back
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step < 0:
        raise SystemExit("--step must be >= 0")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from benchmarks.locked_shared import mode_hold
    from benchmarks.toy100.continuous_probe import FROZEN_STEPS, run_probe
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)
    fingerprint = build_fingerprint()
    (args.output / "build.json").write_text(json.dumps(fingerprint, indent=2) + "\n")
    emit(event="KERNEL_PROBE", method=args.method, step=args.step,
         torch=fingerprint["torch"], cpu=fingerprint["cpu_capability"])

    class OpTrace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.recording = False

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **({} if kwargs is None else kwargs))
            if not self.recording:
                return out
            self.recording = False
            try:
                summary = _tensor_row(out)
                if summary is not None:
                    qual = getattr(func, "name", None)
                    if callable(qual):
                        try:
                            qual = qual()
                        except Exception:
                            qual = str(func)
                    self.rows.append(dict(i=len(self.rows), op=str(qual or func), out=summary))
            finally:
                self.recording = True
            return out

    mode = OpTrace()
    real_checkpoint = mode_hold.checkpoint
    real_set_step = NoisePolicy.set_step
    state = {"active": False, "init_host": None, "init_z": None, "stop_host": None, "stop_z": None}

    def arm():
        if not state["active"]:
            mode.__enter__()
            state["active"] = True
        mode.recording = True

    def disarm():
        mode.recording = False
        if state["active"]:
            mode.__exit__(None, None, None)
            state["active"] = False

    def set_step(self, completed_steps):
        if completed_steps == 0:
            mode.recording = False
            state["init_host"], state["init_z"] = _particle_hash()
            if args.step == 0:
                disarm()
                raise StopAtStep()
        if args.step >= 1 and completed_steps == args.step - 1:
            arm()
        return real_set_step(self, completed_steps)

    def checkpoint(step, measure):
        if args.step >= 1 and int(step) == args.step:
            mode.recording = False
            state["stop_host"], state["stop_z"] = _particle_hash()
            disarm()
            raise StopAtStep()
        return real_checkpoint(step, measure)

    mode_hold.checkpoint = checkpoint
    NoisePolicy.set_step = set_step
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    try:
        if args.step == 0:
            arm()
        with factory(args.method)(task="mode_hold"):
            run_probe(config, mode="constant", steps=FROZEN_STEPS, diagnostic_every=50)
    except StopAtStep:
        pass
    finally:
        disarm()
    rows = mode.rows
    (args.output / "ops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = dict(event="KERNEL_DONE", method=args.method, step=args.step, ops=len(rows),
                   init_host=state["init_host"], init_z=state["init_z"],
                   stop_host=state["stop_host"], stop_z=state["stop_z"],
                   torch=fingerprint["torch"], cpu=fingerprint["cpu_capability"],
                   first_ops=[row["op"] for row in rows[:8]])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(**{k: v for k, v in summary.items() if k not in ("init_z", "stop_z")})


if __name__ == "__main__":
    main()
