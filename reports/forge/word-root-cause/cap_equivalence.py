"""Compare the two completed coefficient170 arms; no updates or sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def identity(path):
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    args = parser.parse_args()
    ids = ("k3p-coeff170-cap1", "k3p-coeff170-cap0p1")
    directories = [args.runs / name for name in ids]
    states = [torch.load(path / "state.pt", map_location="cpu", weights_only=True)
              ["fixture"]["api_state"]["models"] for path in directories]
    tensors = [(role, name) for role, values in states[0].items() for name in values]
    unequal = [f"{role}.{name}" for role, name in tensors
               if not torch.equal(states[0][role][name], states[1][role][name])]
    arrays = [np.load(path / "observations.npz") for path in directories]
    observations_equal = (arrays[0].files == arrays[1].files
                          and all(np.array_equal(arrays[0][key], arrays[1][key])
                                  for key in arrays[0].files))
    result = {
        "scope": "Exact comparison of final model tensors and recorded observations; no training or new sampling.",
        "arms": [{"id": name, "raw_artifacts": {file: identity(path / file)
                  for file in ("state.pt", "observations.npz", "observed-records.pt")}}
                 for name, path in zip(ids, directories)],
        "model_tensor_count": len(tensors), "observation_array_count": len(arrays[0].files),
        "observations_bit_identical": observations_equal,
        "all_model_tensors_bit_identical": not unequal, "unequal_model_tensors": unequal,
        "interpretation": "No observed incremental benefit of lowering the cap under coefficient170; select cap1 as the coefficient-only intervention. Recorded observations and final model tensors do not establish equality of the unobserved parameter trajectory or global cap inactivity.",
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
