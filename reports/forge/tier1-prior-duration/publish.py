"""Publish saved continuation metrics and actual training GIFs; no model draws."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from benchmarks.toy_audit import tier1_prior_duration as study
from benchmarks.toy_audit.api_run import render_gif
from experiments.forge.contracts import atomic_json, file_hash


def publish(raw, destination):
    protocol = study.declaration()
    results = []
    sources = set()
    for row in protocol["runs"]:
        name = row["arm"] + "-" + row["task"]
        directory = raw / name
        receipt = json.loads((directory / "receipt.json").read_text())
        for filename, digest in receipt["artifacts"].items():
            if file_hash(directory / filename) != digest:
                raise ValueError("saved artifact changed: " + filename)
        observations = torch.load(directory / "observations.pt", weights_only=True)
        curve = json.loads((directory / "curve.json").read_text())
        original = study.parent.declaration()
        task = json.loads((study.ROOT / original["tasks"][row["task"]]["path"]).read_text())
        target, _ = study.parent.scorer(row["task"])
        reference = target(task["execution"]["host_definition"], 4096,
                           torch.Generator(device="cpu").manual_seed(78013), 0)
        frames = []
        for index in np.linspace(0, len(observations)-1, 9).round().astype(int):
            observed = observations[index]
            samples = observed["samples"]
            if samples.shape[1] == 1:
                edges = np.linspace(-2., 5., 57)
                def density(values):
                    return np.histogram(values[:, 0].numpy(), edges)[0] / len(values) / np.diff(edges)
                outside = int(((samples[:, 0] < -2) | (samples[:, 0] > 5)).sum())
                view = dict(kind="bar", title="Target and actual Gaussian histograms",
                            target=density(reference), samples=density(samples),
                            bin_centers=(edges[:-1]+edges[1:])/2, bin_width=float(edges[1]-edges[0]),
                            xlim=[-2.,5.], ylim=[0.,8.5], xlabel="x", ylabel="density",
                            caption=f"Exact checkpoint continuation; full-law CDF gate includes {outside}/4096 samples outside the fixed histogram range.")
            else:
                view = dict(kind="scatter", title="Target and actual sixteen-mode ring",
                            target=reference, samples=samples, xlim=[-4.,4.], ylim=[-4.,4.], xlabel="x", ylabel="y",
                            caption="Exact checkpoint continuation; clean live prior/G draws. Full covariance/precision gates remain unchanged.")
            metrics = observed["metrics"]
            frames.append(dict(step=observed["step"], views=[view],
                               passed=not study.parent._bounds(metrics, task["evaluation"]["thresholds"]),
                               metrics={k:float(v) for k,v in metrics.items() if isinstance(v,(float,int))}))
        gif = name + ".gif"
        render_gif(dict(id=name, goal="Does four times the original budget achieve sustained full-quality acquisition?", default_steps=row["max_total_updates"]),
                   frames, destination / gif, full_budget=True, requested_steps=row["max_total_updates"], final_verdict=receipt["full_verdict"])
        cuts = []
        for multiple in range(1,5):
            prefix = [point for point in curve if point["step"] <= multiple * row["parent_updates"]]
            cuts.append(dict(updates=multiple * row["parent_updates"], metrics=prefix[-1]["metrics"],
                             full_terminal_suffix=study.parent.suffix(prefix,"full_pass"),
                             smoke_terminal_suffix=study.parent.suffix(prefix,"smoke_pass")))
        first = None
        longest = count = 0
        for point in curve:
            count = count+1 if point["full_pass"] else 0
            longest = max(longest,count)
            if count == 5 and first is None:
                first = point["step"]
        source = json.loads((directory / "source.json").read_text())
        sources.add(source["digest"])
        results.append({**receipt, "raw_receipt_sha256":file_hash(directory / "receipt.json"),
                        "source_digest":source["digest"], "source_commit":source["origin_commit"],
                        "resume_proof":json.loads((directory / "resume-proof.json").read_text()),
                        "budget_cuts":cuts, "first_five_passing_checks_update":first,
                        "longest_passing_streak":longest, "gif":gif,"gif_sha256":file_hash(destination / gif)})
    assert len(sources) == 1
    rings = [r for r in results if r["task"] == "ring16_acquisition"]
    assert len({r["new_data_sequence_sha256"] for r in rings}) == 1
    atomic_json(destination / "results.json",dict(schema_version=1,scope=protocol["scope"],qualification_input=False,
                continuations=3,additional_updates=5400,resume_and_batch_checks_passed=True,
                continuation_loop_seconds=sum(r["continuation_loop_seconds"] for r in results),runs=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    publish(args.raw,args.output)
