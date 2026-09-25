#!/usr/bin/env python
"""Posthoc GAN sample distributions on held-out expert terrain contexts, CPU only."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.evaluate_gym_gan_control import verify_protocol, verify_checkpoint, verify_training_protocol
from lib.gym_control import build_expert_records
from lib.gym_data import sha256
from lib.gym_evaluation import distribution_metrics
from lib.gym_gan_control import load_gan_control_checkpoint
from lib.gym_transition import contact_record
from lib.toy_metrics import sliced_w1


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@torch.no_grad()
def sample_records(bundle, records):
    """Identical context order, batch boundaries, and RNG streams for each model."""
    rngs = {name: torch.Generator(device="cpu").manual_seed(seed)
            for name,seed in (("latent", 92173), ("prior_contacts", 92174), ("encoded_contacts", 92175))}
    result = {"prior": [], "encoded": []}
    sampled_components = []
    for start in range(0, len(records["states"]), 256):
        states = torch.as_tensor(records["states"][start:start+256])
        context = torch.as_tensor(records["terrain"][start:start+256])
        z, indices = bundle["prior"].sample(len(states), rngs["latent"])
        sampled_components.append(indices.cpu().numpy())
        encoded = bundle["E"](bundle["scaler"].state(states), context, bundle["prior"])
        for path,latent,contacts in (("prior", z, "prior_contacts"),
                                     ("encoded", encoded.codes[:,0], "encoded_contacts")):
            output = contact_record(bundle["G"](latent, context), rng=rngs[contacts], mode="sample")
            array = output.cpu().numpy()
            if not np.isfinite(array).all() or not np.isin(array[:, [6,7,16,17]], [0,1]).all():
                raise ValueError("Expected finite records with sampled binary contacts")
            result[path].append(array)
    arrays = {path: np.concatenate(values) for path,values in result.items()}
    evidence = dict(final_rng_state_sha256={name: array_hash(rng.get_state().numpy()) for name,rng in rngs.items()},
        prior_component_indices_sha256=array_hash(np.concatenate(sampled_components)),
        generated_array_sha256={name: array_hash(value) for name,value in arrays.items()})
    return arrays, evidence


def diagnose(report_dir):
    torch.set_num_threads(1)
    protocol = verify_protocol(report_dir / "protocol.json")
    records = build_expert_records(protocol["source_episodes"], "test")
    if len(records["states"]) != 2444:
        raise ValueError("Expected the frozen 2,444 held-out expert behavior transitions")
    scaler_values = protocol["supervision"]["normalization_statistics"]
    output = dict(kind="Posthoc descriptive sample quality; never used for checkpoint selection",
        protocol_sha256=sha256(report_dir / "protocol.json"), source_sha256=sha256(__file__),
        metrics_source_sha256=sha256(inspect.getfile(distribution_metrics)),
        projection_source_sha256=sha256(inspect.getfile(sliced_w1)),
        source_episodes=dict(path=protocol["source_episodes"], sha256=protocol["source_episodes_sha256"]),
        reference=dict(split="test", records=len(records["states"]),
            episode_ids=np.unique(records["episode_ids"]).tolist(),
            array_hashes={key: array_hash(records[key]) for key in ("states", "actions", "next_states", "terrain", "episode_ids", "steps")}),
        normalization=scaler_values, simulator_steps=0, simulator_resets=0, training_updates=0, device="cpu",
        sampling=dict(batch_size=256, latent_seed=92173, prior_contact_seed=92174, encoded_contact_seed=92175,
            contexts="Exact held-out expert record terrain order, one fake per real record for each path",
            common_randomness="Same component draws and Gaussian-noise RNG stream across arms; trained centers differ. Independent common contact-uniform streams for prior and encoded paths.",
            contacts="Sampled binary contacts, matching discriminator training representation"),
        metrics=dict(joint_sw1="128 fixed directions, inherited projection seed31415; lower is better",
            reference_radius="95th percentile of each real record's nearest other real record distance; shared scalar radius",
            precision="Fraction of generated records within reference radius of any real record",
            coverage="Fraction of real records within reference radius of any generated record",
            contact_pattern_tv="Total variation between distributions of four binary contact bits; lower is better"),
        limitations=["Pooled normalized joint records exclude terrain from metric distances; matching context frequencies does not prove conditional fidelity or physical consistency.",
                     "Reference records are temporally correlated expert behavior observations, not independent samples or an unrestricted counterfactual dynamics benchmark.",
                     "Encoded path receives actual held-out current states; it is not an unconditional prior sample and does not receive held-out actions or successors.",
                     "All explicit action labels in the reference are held-out posthoc scoring targets; no new labels enter training or checkpoint selection.",
                     "Nearest-neighbor precision/coverage depend on reference density and the inherited radius convention; these are descriptive metrics without confidence intervals."], controllers={})
    common_sampling = None
    common_real = None
    for arm in ("joint", "marginals"):
        selection_path = report_dir / "selections" / f"{arm}.json"
        selection = json.loads(selection_path.read_text())
        checkpoint = Path(selection["selected_checkpoint"])
        if sha256(checkpoint) != selection["validation"]["checkpoint_sha256"]:
            raise RuntimeError("Selected checkpoint differs from validation selection")
        bundle = load_gan_control_checkpoint(checkpoint, "cpu")
        verify_checkpoint(bundle, checkpoint)
        verify_training_protocol(bundle, protocol)
        physical = np.concatenate([records[key] for key in ("states", "actions", "next_states")], 1)
        real = bundle["scaler"](torch.as_tensor(physical)).numpy()
        if common_real is not None and not np.array_equal(real, common_real):
            raise RuntimeError("Models do not share the frozen normalized reference")
        common_real = real
        samples, evidence = sample_records(bundle, records)
        sampling_identity = {key:evidence[key] for key in ("final_rng_state_sha256", "prior_component_indices_sha256")}
        if common_sampling is not None and sampling_identity != common_sampling:
            raise RuntimeError("Models did not consume identical sampling streams")
        common_sampling = sampling_identity
        output["controllers"][arm] = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
            selection_sha256=sha256(selection_path), step=bundle["step"], sampling=evidence,
            paths={name: dict(records=len(values), **distribution_metrics(values, real, device="cpu")) for name,values in samples.items()})
    output["reference"]["normalized_joint_array_sha256"] = array_hash(common_real)
    return output


def markdown(value):
    lines = ["# GAN sample quality on held-out expert contexts", "",
        "Both paths generate one complete record per each of the same 2,444 held-out expert terrain contexts. Prior noise/component draws and contact-sampling streams are matched across models. Contacts are binary samples. Checkpoints were already selected by control validation.", "",
        "| GAN | Path | SW1 ↓ | Precision ↑ | Coverage ↑ | Contact TV ↓ | Reference radius |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm,row in value["controllers"].items():
        for path,g in row["paths"].items():
            lines.append(f"| {arm} | {path} | {g['joint_sw1']:.5f} | {g['precision']:.1%} | {g['coverage']:.1%} | {g['contact_pattern_tv']:.5f} | {g['reference_radius']:.5f} |")
    lines += ["", "Precision and coverage use the 95th percentile of real nearest-other-real distances as their common radius. SW1 uses 128 fixed projections. All distances use the common sparse-training scaler.", "",
        "These are pooled joint-record metrics: terrain is excluded from metric distances. They do not prove conditional physics or successful control. The encoded path is conditioned on actual current states; the prior path is not.", "", *value["limitations"], "",
        "Zero simulator steps/resets and zero training updates. Hashes and exact definitions are in `prior_sample_diagnostics.json`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports/gym/lunar_lander_gan_control")
    args = parser.parse_args()
    target = args.reports / "prior_sample_diagnostics.json"
    if target.exists() or target.with_suffix(".md").exists():
        raise FileExistsError("Use fresh diagnostics output")
    value = diagnose(args.reports)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    target.with_suffix(".md").write_text(markdown(value))
    print(markdown(value), flush=True)


if __name__ == "__main__":
    main()
