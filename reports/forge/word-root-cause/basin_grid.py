"""Fixed deterministic latent-box query of saved word generators; no training.

This grid probes word basins and does not estimate the public prior law.
Absence on the finite grid does not establish global absence of a word.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.toy_audit.api_images import WordGenerator, WordEncoder, word_bank, WORDS


def analyze(directory):
    path = directory / "state.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)["fixture"]["api_state"]
    with torch.random.fork_rng(devices=[]):
        G, E = WordGenerator(), WordEncoder()
    G.load_state_dict(state["models"]["generator"])
    E.load_state_dict(state["models"]["encoder"])
    G.eval().requires_grad_(False)
    E.eval().requires_grad_(False)
    words = word_bank()
    points = torch.cat((state["models"]["prior"]["z"], E(words).detach()))
    low, high = points.amin(0) - .25, points.amax(0) + .25
    axes = [torch.linspace(float(low[j]), float(high[j]), 81) for j in range(2)]
    x, y = torch.meshgrid(*axes, indexing="ij")
    latents = torch.stack((x.flatten(), y.flatten()), 1)
    probabilities = G(latents)
    decoded = probabilities.argmax(1)
    tokens = words.argmax(1)
    matches = (decoded[:, None] == tokens[None]).all(2)
    correct = torch.einsum("ncl,wcl->nwl", probabilities, words).amin(2)
    maximum, best = correct.amax(0), correct.argmax(0)
    return {"id": directory.name, "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "grid": {"points": len(latents), "side": 81, "lower": low.tolist(), "upper": high.tolist(),
                 "law": "Uniform rectangular grid containing saved prior/encoder locations with .25 padding; diagnostic only"},
        "words": {word: {"correct_argmax_grid_points": int(matches[:, j].sum()),
            "confident_grid_points": int((correct[:, j] >= .9).sum()),
            "maximum_minimum_correct_token_probability": float(maximum[j]),
            "best_grid_location": latents[best[j]].tolist()}
            for j, word in enumerate(WORDS)},
        "invalid_argmax_grid_points": int((~matches.any(1)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps({"schema_version": 1,
        "scope": "No training, no random draws, no prior-law or global-solvability claim",
        "states": [analyze(p) for p in args.directories]}, indent=2))


if __name__ == "__main__":
    main()
