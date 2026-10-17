"""Audit saved train/evaluation streams and the unchanged CPU target law."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.rng import NamedStreams
from benchmarks.toy_audit.gaussian1d_quality import sample_target


def verify(raw, baseline, output):
    result = json.loads((raw / "results.json").read_text())
    request = json.loads((raw / "request.json").read_text())
    spec = deepcopy(next(task for task in request["tasks"].values()
                         if task["evaluation"]["kind"] == "gaussian_smoke")["execution"]["host_definition"])
    data = NamedStreams(0).generator("data", component="target", purpose="training", device="cpu")
    segment = {name: hashlib.sha256() for name in ("smoke", "hold", "shift", "stationary_all")}
    for step in range(1, 6001):
        if step == 4001:
            spec["means"] = [[3.]]
        payload = sample_target(spec, 128, data, step - 1).numpy().tobytes()
        segment["smoke" if step <= 1000 else "hold" if step <= 4000 else "shift"].update(payload)
        if step <= 4000:
            segment["stationary_all"].update(payload)
    digests = {name: value.hexdigest() for name, value in segment.items()}
    assert result["smoke"]["evidence"]["data_sha256"]["stationary"] == digests["smoke"]
    assert result["stability"]["evidence"]["data_sha256"] == dict(stationary=digests["hold"], shift=digests["shift"])
    assert digests["stationary_all"] == "a3b2f48e279bc16fe035cc754f57bc1085465e3a75b34c03c092d8ef9198fb8d"
    assert digests["shift"] == "dd6e1f6297c495d55222ca2d114743b3da84e99efe83d48df12d7228017e39ba"
    saved_root = Path(result["stability"]["evidence"]["artifact_root"])
    streams = []
    for step, saved_name, phase in ((4000, "pre-shift-state.pt", "stationary"), (6000, "state.pt", "shift")):
        path = saved_root / saved_name
        reference = baseline / f"alternating-gaussian1d_acquisition-{phase}" / "state.pt"
        state, original = (torch.load(item, weights_only=True, map_location="cpu") for item in (path, reference))
        assert set(state["trainer"]["streams"]) == set(original["trainer"]["streams"])
        assert all(torch.equal(value, original["trainer"]["streams"][name])
                   for name, value in state["trainer"]["streams"].items())
        def selected(value):
            return {(binding["family"], binding["component"], binding["purpose"]): value["streams"]["states"][key]
                    for key, binding in value["streams"]["manifest"]["bindings"].items()
                    if binding["family"] == "data" or
                    (binding["family"], binding["component"], binding["purpose"]) == ("eval", "live", "samples")}
        current, old = selected(state), selected(original)
        assert current.keys() == old.keys()
        assert all(torch.equal(value, old[key]) for key, value in current.items())
        streams.append(dict(step=step, matched_train_data_and_primary_eval_streams=True,
                            reference_checkpoint_sha256=file_hash(reference), current_checkpoint_sha256=file_hash(path),
                            reference_source_sha256=file_hash(reference.parent / "source.json")))
    proof = dict(passed=True, training_updates=0, model_sampling_draws=0,
                 scope="saved_stream_and_cpu_target_sequence_audit", target_digests=digests, stream_checks=streams,
                 confirmation_stream="Separate new smoke confirmation stream; primary eval/training streams remain matched.")
    atomic_json(output, proof)
    print(json.dumps(proof))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.raw, args.baseline, args.output)
