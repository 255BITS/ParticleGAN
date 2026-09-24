"""One mode-hold update under the active MKL preload. Records ATen outputs.

Same factory and seed hook as dispatch_seed_run. Stops at the end of update
``--step`` (1-based). Does not change the candidate.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

REPO = Path(os.environ["AUDIT_REPO"]).resolve()
sys.path.insert(0, str(REPO))

import importlib.util
_hook = Path(__file__).with_name("dispatch_seed_run.py")
_spec = importlib.util.spec_from_file_location("audit_seed_hook", _hook)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class StopAtStep(Exception):
    pass


def _summarize(value):
    import torch
    if isinstance(value, torch.Tensor):
        if not (value.is_floating_point or value.is_complex):
            return None
        array = value.detach().float().cpu().contiguous().numpy()
        flat = array.reshape(-1)
        exact = array.astype("float64") if array.size else array
        quant = (exact * 1e6).round().astype("int64") if array.size else array
        return dict(
            hash=hashlib.sha256(array.tobytes()).hexdigest(),
            qhash=hashlib.sha256(quant.tobytes()).hexdigest() if array.size else None,
            shape=list(array.shape), numel=int(array.size),
            sum=float(exact.reshape(-1).sum()) if array.size else 0.0,
            max=float(flat.max()) if array.size else 0.0,
            min=float(flat.min()) if array.size else 0.0,
            head=[float(x) for x in flat[:4]],
        )
    if isinstance(value, (list, tuple)) and value:
        parts = [part for part in (_summarize(item) for item in value) if part is not None]
        return parts or None
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-op", type=int, default=-1)
    parser.add_argument("--tensor-dir", type=Path, default=None,
                        help="Write each floating output as i.npy (or i_k.npy).")
    parser.add_argument("--compare-dir", type=Path, default=None,
                        help="Max-abs against a previous --tensor-dir. Writes diffs.jsonl.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from benchmarks.locked_shared import mode_hold
    from benchmarks.toy100.continuous_probe import FROZEN_STEPS, run_probe
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.gan_followup_probe import ROOT, factory

    torch.set_num_threads(1)

    class OpTrace(TorchDispatchMode):
        def __init__(self, save_op):
            super().__init__()
            self.rows = []
            self.recording = False
            self.saved = None
            self.stack = None
            self.save_op = save_op
            self.tensor_dir = None
            self.compare_dir = None
            self.diffs = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **({} if kwargs is None else kwargs))
            if not self.recording:
                return out
            index = len(self.rows)
            self.recording = False
            try:
                if index == self.save_op:
                    self.saved = out.detach().cpu().clone() if isinstance(out, torch.Tensor) else None
                    self.stack = "".join(traceback.format_stack(limit=30))
                self._tensors(index, str(getattr(func, "name", None)() if callable(getattr(func, "name", None)) else func), out)
                summary = _summarize(out)
                if summary is not None:
                    qual = getattr(func, "name", None)
                    if callable(qual):
                        try:
                            qual = qual()
                        except Exception:
                            qual = str(func)
                    self.rows.append(dict(i=index, op=str(qual or func), out=summary))
                else:
                    self.rows.append(dict(i=index, op=str(func), out=None))
            finally:
                self.recording = True
            return out

        def _tensors(self, index, op, out):
            if self.tensor_dir is None and self.compare_dir is None:
                return
            import numpy as np
            import torch
            items = out if isinstance(out, (list, tuple)) else (out,)
            for k, item in enumerate(items):
                if not isinstance(item, torch.Tensor) or not item.is_floating_point():
                    continue
                array = item.detach().float().cpu().contiguous().numpy()
                name = f"{index}.npy" if len(items) == 1 else f"{index}_{k}.npy"
                if self.tensor_dir is not None:
                    np.save(self.tensor_dir / name, array)
                if self.compare_dir is not None:
                    ref = self.compare_dir / name
                    if not ref.exists():
                        self.diffs.append(dict(i=index, op=op, missing=name))
                        continue
                    delta = float(np.max(np.abs(array.astype("float64") - np.load(ref).astype("float64"))))
                    self.diffs.append(dict(i=index, op=op, max_abs=delta, file=name))

    mode = OpTrace(args.save_op)
    if args.tensor_dir is not None:
        args.tensor_dir.mkdir(parents=True, exist_ok=False)
        mode.tensor_dir = args.tensor_dir
    if args.compare_dir is not None:
        mode.compare_dir = args.compare_dir
    real_checkpoint = mode_hold.checkpoint
    real_set_step = NoisePolicy.set_step
    state = {"active": False, "z": None}

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

    def particle():
        frame = sys._getframe(1)
        while frame is not None:
            prior = frame.f_locals.get("prior") if isinstance(frame.f_locals, dict) else None
            if prior is not None and hasattr(prior, "z"):
                return prior.z.detach().cpu().clone()
            frame = frame.f_back
        return None

    def set_step(self, completed_steps):
        if completed_steps == args.step - 1:
            arm()
        return real_set_step(self, completed_steps)

    def checkpoint(step, measure):
        if int(step) == args.step:
            mode.recording = False
            state["z"] = particle()
            disarm()
            raise StopAtStep()
        return real_checkpoint(step, measure)

    mode_hold.checkpoint = checkpoint
    NoisePolicy.set_step = set_step
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    try:
        with factory(args.method)(task="mode_hold"):
            run_probe(config, mode="constant", steps=FROZEN_STEPS, diagnostic_every=50)
    except StopAtStep:
        pass
    finally:
        disarm()
    if state["z"] is not None:
        torch.save(state["z"], args.output / "z.pt")
    if mode.saved is not None:
        torch.save(mode.saved, args.output / "op.pt")
    if mode.stack:
        (args.output / "stack.txt").write_text(mode.stack)
    (args.output / "ops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in mode.rows))
    if mode.diffs:
        (args.output / "diffs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in mode.diffs))
        crossed = next((row for row in mode.diffs if row.get("max_abs", 0) > 1e-6), None)
        print(json.dumps(dict(event="OP_DIFF", first_gt_1e_6=crossed, n=len(mode.diffs))), flush=True)
    print(json.dumps(dict(event="OP_DONE", ops=len(mode.rows), step=args.step)), flush=True)


if __name__ == "__main__":
    main()
