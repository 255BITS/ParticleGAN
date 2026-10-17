"""Record the existing evaluator's actual observations without changing training.

Clouds and full curves stay in a local artifact directory. Compact summaries
and rendered media can be committed. This is not a replacement training loop.
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.transfer_suite import image_tasks, vector_tasks
from benchmarks.transfer_suite.protocol import test_verdict


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Capture:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.current = None
        self._vector_run = vector_tasks.run_episode
        self._image_run = image_tasks.run_episode
        self._vector_score = vector_tasks.score_samples
        self._image_measure = image_tasks.measure

    def vector_score(self, fake, spec, step):
        metrics = self._vector_score(fake, spec, step)
        if self.current is not None:
            self.current["frames"].append(fake.detach().cpu().numpy().copy())
            self.current["frame_steps"].append(step)
        return metrics

    def image_measure(self, generator, prior, centers, thresholds):
        metrics = self._image_measure(generator, prior, centers, thresholds)
        if self.current is not None:
            # Enumerating the finite table is deterministic. Preserve global
            # RNG and module modes even if a future model gains stochasticity.
            modes = [(m, m.training) for m in generator.modules()]
            try:
                devices=[] if centers.device.type=="cpu" else [centers.device.index]
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    images = generator(prior.z).detach().cpu().numpy().copy()
            finally:
                for module, mode in modes:
                    module.training = mode
            self.current["frames"].append(images)
            self.current["templates"] = centers.detach().cpu().numpy().copy()
        return metrics

    def episode(self, kind, spec, *args, **kwargs):
        if self.current is not None:
            raise RuntimeError("nested audit capture")
        started = time.perf_counter()
        record = dict(kind=kind, spec=deepcopy(spec), frames=[], frame_steps=[])
        self.current = record
        run = self._image_run if kind == "image" else self._vector_run
        try:
            result = run(spec, *args, **kwargs)
            record["result"] = result
        finally:
            self.current = None
        index = len(self.records)
        case = self.output / f"episode-{index:02d}"
        case.mkdir(exist_ok=True)
        frames = record.pop("frames")
        frame_steps = record.pop("frame_steps")
        templates = record.pop("templates", None)
        # Existing hosts measure live followed by EMA at every checkpoint.
        # Bind that ordering to the recorded metrics; refuse incomplete pairs.
        observations = result.get("observations", [])
        if len(frames) != 2 * len(observations):
            raise ValueError(f"capture/evaluator cadence differs: {len(frames)} frames, {len(observations)} observations")
        arrays = dict(live=np.asarray(frames[::2], dtype=np.float32),
                      ema=np.asarray(frames[1::2], dtype=np.float32),
                      steps=np.asarray([x["step"] for x in observations]))
        if templates is not None:
            arrays["templates"] = templates
        if kind == "vector" and frame_steps != [s for x in observations for s in (x["step"], x["step"])]:
            raise ValueError("vector sample and metric steps differ")
        np.savez_compressed(case / "observations.npz", **arrays)
        write(case / "result.json", result)
        spec = result.get("spec", spec)
        verdict = test_verdict(spec, result)
        summary = dict(name=spec["name"], kind=kind, spec=spec, verdict=verdict,
                       live=result.get("live", {}), ema=result.get("ema", {}),
                       error=result.get("error"), seconds=time.perf_counter()-started,
                       sampling="exact clean live particle enumeration" if kind == "image" else "original clean live evaluator draw",
                       frames=len(observations), capture_sha256=hashlib.sha256((case / "observations.npz").read_bytes()).hexdigest(),
                       result_sha256=hashlib.sha256((case / "result.json").read_bytes()).hexdigest(),
                       artifact=str(case))
        write(case / "summary.json", summary)
        self.records.append(summary)
        write(self.output / "capture-index.json", self.records)
        print(json.dumps(dict(event="AUDIT_EPISODE", name=spec["name"], status=verdict["status"],
                              frames=len(observations), artifact=str(case))), flush=True)
        return result

    @contextmanager
    def installed(self):
        @wraps(self._vector_run)
        def vector_run(*args, **kwargs):
            return self.episode("vector", *args, **kwargs)
        @wraps(self._image_run)
        def image_run(*args, **kwargs):
            return self.episode("image", *args, **kwargs)
        with patch.object(vector_tasks, "score_samples", self.vector_score), \
             patch.object(image_tasks, "measure", self.image_measure), \
             patch.object(vector_tasks, "run_episode", vector_run), \
             patch.object(image_tasks, "run_episode", image_run):
            yield self
