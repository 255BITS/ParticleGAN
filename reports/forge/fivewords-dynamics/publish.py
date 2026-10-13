"""Reduce retained endpoint outputs and archive exact inputs; no model updates.

The optional correspondence readout is a retrospective transform of saved
latent coordinates, evaluated on CUDA. Original diagnostic protocol unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import subprocess
import tarfile
import io

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summarize(observation):
    import torch
    encoded = torch.tensor(observation["encoded_latents"], device="cuda:0", dtype=torch.float64)
    prior = torch.tensor(observation["prior_latents"], device="cuda:0", dtype=torch.float64)
    distance = torch.cdist(encoded, prior)
    choices = torch.tensor(list(permutations(range(5))), device="cuda:0")
    cost = distance[torch.arange(5, device="cuda:0")[None], choices].square().mean(1)
    chosen = int(cost.argmin())
    prior_separation = torch.pdist(prior).min()
    rms = float(cost[chosen].sqrt())
    metrics = observation["metrics"]
    return dict(passed=observation["passed"], failed_bounds=observation["failed_bounds"],
        quality_fraction=metrics["quality_fraction"], modes=metrics["modes"], mass_tv=metrics["mass_tv"],
        reconstruction_exact=metrics["reconstruction_exact"],
        minimum_reconstruction_token_probability=metrics["minimum_reconstruction_token_probability"],
        generated_row_word_assignment=observation["prior_row_word_assignment"],
        reconstructed_word_assignment=observation["reconstruction_word_assignment"],
        code_matching=dict(optimal_prior_rows=choices[chosen].tolist(), rms_distance=rms,
            minimum_prior_pair_distance=float(prior_separation),
            rms_over_minimum_prior_separation=rms/float(prior_separation) if bool(prior_separation > 0) else None,
            nearest_rows=observation["encoded_to_prior_nearest_rows"],
            unique_nearest_rows=len(set(observation["encoded_to_prior_nearest_rows"]))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--unsmoothed-root", type=Path, required=True)
    parser.add_argument("--smoothed-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    args = parser.parse_args()
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Saved latent reduction requires reserved GPU1; no CPU fallback")
    raw = json.loads((args.raw/"readout.json").read_text())
    if sha(HERE/"protocol.json") != raw["execution"]["protocol_sha256"] or sha(HERE/"diagnose.py") != raw["execution"]["script_sha256"]:
        raise RuntimeError("Executed diagnostic source changed")
    supplement_path=args.raw.parent/"latent-line-readout.json"
    supplement=json.loads(supplement_path.read_text())
    if supplement["protocol_sha256"]!=sha(HERE/"supplement-protocol.json") or supplement["source_sha256"]!=sha(HERE/"supplement.py"):
        raise RuntimeError("Executed supplement source changed")
    write(HERE/"supplement-readout.json",supplement)
    endpoints = []
    members = {}
    for endpoint in raw["endpoints"]:
        row = deepcopy(endpoint)
        row["exact_uniform_endpoint"] = summarize(row["exact_uniform_endpoint"])
        row["first_full_step"] = summarize(row["first_full_step"])
        for probe in row["role_probes"].values():
            probe["first"] = summarize(probe["first"])
            probe["final"] = summarize(probe["final"])
        for values in row["expected_role_step"].values():
            values.pop("layers", None)
        recipe = row.pop("original_recipe")
        row["recipe_sha256"] = sha_bytes(json.dumps(recipe, sort_keys=True).encode())
        row["smoothing_lambda"] = recipe.get("optimizer_smoothing", 0.)
        source = (args.unsmoothed_root/row["arm"]["id"] if row["cohort"] == "unsmoothed" else
                  args.smoothed_root/("five_word_joint_acquisition--"+row["arm"]["id"]))
        source_manifest=json.loads((source/"source-manifest.json").read_text())
        row["original_training_source"]={key:source_manifest[key] for key in ("origin_commit","digest")}
        request=json.loads((source/"request.json").read_text())["request"]
        task=request["tasks"]["five_word_joint_acquisition"]
        row["original_task_sha256"]=sha_bytes(json.dumps(task,sort_keys=True).encode())
        for name, receipt in row["input_files"].items():
            path = source/name
            if sha(path) != receipt["sha256"]:
                raise RuntimeError(f"Changed original input: {path}")
            members[f"inputs/{row['cohort']}/{row['arm']['id']}/{name}"] = path.read_bytes()
        endpoints.append(row)
    compact = {key:value for key,value in raw.items() if key != "endpoints"}
    compact["endpoints"] = endpoints
    compact["zero_update_supplement"] = dict(readout_path="supplement-readout.json",readout_sha256=sha(HERE/"supplement-readout.json"),model_updates=0,sampling_draws=0,forward_points=supplement["forward_points"],wall_seconds=supplement["wall_seconds"])
    compact["reduction"] = dict(source_sha256=sha(args.raw/"readout.json"),
        exact_uniform_law="Five finite rows, deterministic equal-mass weighting;1025 repeated representations are not independent samples or ordinary gate credit",
        correspondence_method="CUDAfloat64 exact120-permutation minimum mean-square encoder/prior matching; RMS divided by minimum prior-row separation; retrospective saved-output reduction, no updates",
        projected_jacobian_scope="RAW loss-minimization field, four-dimensional local directional projection, not normalized-update Jacobian or full spectrum",
        probe_scope="T(E[gradient]), not E[T(stochastic gradient)]; Cartesian batch25 versus historical sampled batch256")
    write(HERE/"readout.json",compact)
    (HERE/"media").mkdir(exist_ok=True)
    gif=HERE/"media/archived-smoothed-truncated-serial.gif"
    gif.write_bytes(args.gif.read_bytes())
    write(HERE/"media/receipt.json",dict(sha256=sha(gif),bytes=gif.stat().st_size,
        source_commit="6d3d5b21144d85061fce6349b706e9c2800b0f96",
        source_path="reports/forge/smooth-polar-factorial/media/five_word_joint_acquisition--truncated-serial.gif",
        executed_training_commit=raw["execution"]["runtime_commit"],
        scope="Historical actually-scored training outputs; no new model execution or GIF draws"))
    for path in sorted(args.raw.parent.rglob("*")):
        if path.is_file() and path.name != "publish.log":
            members["diagnostic/"+path.relative_to(args.raw.parent).as_posix()]=path.read_bytes()
    for name in ("README.md","diagnose.py","protocol.json","publish.py","readout.json","supplement.py","supplement-protocol.json","supplement-readout.json","media/receipt.json","media/archived-smoothed-truncated-serial.gif"):
        members["report/"+name]=(HERE/name).read_bytes()
    for name, receipt in raw["execution"]["source_files"].items():
        value=(args.runtime_root/name).read_bytes()
        if sha_bytes(value)!=receipt:
            raise RuntimeError(f"Changed frozen numerical source: {name}")
        members["frozen-runtime/"+name]=value
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    with args.archive.open("wb") as file, gzip.GzipFile(filename="",mode="wb",fileobj=file,mtime=0) as compressed, tarfile.open(fileobj=compressed,mode="w") as archive:
        for name,value in sorted(members.items()):
            info=tarfile.TarInfo(name)
            info.size=len(value);info.mode=0o644;info.mtime=0
            archive.addfile(info,io.BytesIO(value))
    with tarfile.open(args.archive,"r:gz") as archive:
        for entry in archive.getmembers():
            if archive.extractfile(entry).read()!=members[entry.name]:
                raise RuntimeError(f"Archive verification failed: {entry.name}")
    receipt=dict(sha256=sha(args.archive),bytes=args.archive.stat().st_size,file_count=len(members),
        archive_name=args.archive.name,bulk_artifacts_committed=False,
        contents="Complete eight original model/optimizer/recipe/RNG endpoints and scored tensors, original source manifests/requests, diagnostic outputs/logs including preflight rejection, exact executed diagnostic and imported numerical source subset, compact readout and historical GIF",
        original_full_source="Original full1192-file source packets remain under published PR343/344 archives and their immutable git commits; copied imported numerical subset is not claimed as standalone full-package source",
        members=[dict(path=name,bytes=len(value),sha256=sha_bytes(value)) for name,value in sorted(members.items())])
    write(HERE/"archive.json",receipt)
    print(json.dumps(dict(archive_sha256=receipt["sha256"],archive_bytes=receipt["bytes"],members=len(members),readout_sha256=sha(HERE/"readout.json"))))


if __name__ == "__main__":
    main()
