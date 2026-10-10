"""Bounded CPU-wheel diagnostic: compare clean/noisy scoring at native sizes.

Forty updates, the unchanged 7k schedule, seed and BCap host. This measures
training/RNG equivalence only; it is not a shorter scientific quality gate.
Full checkpoints and logs remain in the requested local artifact directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.toy100 import train as runner


def differences(left, right, path=""):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            return [dict(path=path, dtype=str(left.dtype), shape=list(left.shape),
                max_abs=float((left.double()-right.double()).abs().max()),
                exact_equal=False)]
        return []
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        return [row for key in left for row in differences(left[key], right[key], path + "/" + str(key))]
    if isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        return [row for i, (a,b) in enumerate(zip(left,right)) for row in differences(a,b,path + "/" + str(i))]
    return [] if left == right else [dict(path=path, left=left, right=right)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = runner.make_trainer
    trainers = []
    class PrefixComplete(Exception):
        pass
    def capture(*params, **options):
        trainer = original(*params, **options)
        step = trainer.step
        def bounded_step(*args, **kwargs):
            if trainer.completed_steps == 40:
                raise PrefixComplete("controlled software-probe stop at 40/7000 updates")
            return step(*args, **kwargs)
        trainer.step = bounded_step
        trainers.append(trainer)
        return trainer
    runner.make_trainer = capture
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(problem="grid100", device="cpu")
    states = []
    for enabled in (False, True):
        try:
            runner.train({**config, "eval_output_noise": enabled}, args.output / str(enabled))
        except PrefixComplete:
            pass
        assert trainers[-1].completed_steps == 40
        state = trainers[-1].state_dict()
        torch.save(state, args.output / f"state-{enabled}.pt")
        states.append(state)
    changed = differences(*states)
    receipt = dict(diagnostic_only=True, scientific_qualification=False,
        new_training_updates=80, updates_per_case=40, schedule_horizon=7000,
        sampling_laws=["clean", "training_noise"], unchanged_seed=1234,
        torch=str(torch.__version__), python=sys.version, cpu_capability=torch.backends.cpu.get_cpu_capability(),
        torch_build=torch.__config__.show(),
        environment={name:os.environ.get(name) for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY",
            "MKL_ENABLE_INSTRUCTIONS", "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        checkpoint_exact_equal=not changed, differences=changed,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "comparison.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(checkpoint_exact_equal=not changed, differences=len(changed), new_training_updates=80)))


if __name__ == "__main__":
    main()
