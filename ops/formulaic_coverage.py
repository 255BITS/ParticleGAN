"""Retired alternate coverage executor; retained for explicit compatibility errors.

The old runner changed candidate recipes, added host training algorithms, and
harvested results by family/task without exact scientific bindings. New work
must use frozen Forge studies and the public adapters. Historical measurements
remain under their original source and contracts; this module never reruns,
rewrites, or promotes them to repair a publication.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RETIREMENT_REASON = (
    "The alternate formulaic coverage executor is retired: it cannot supply "
    "formula-faithful evidence. Use python -m experiments.forge plan/run with "
    "an explicit frozen study, protocol and budget. Unsupported capabilities "
    "remain BLOCKED; preserved historical diagnostic receipts are not new measurements."
)


class RetiredCoverageExecutionError(RuntimeError):
    """A caller attempted to bypass ordinary frozen experiment admission."""


DDGAN_TASKS = frozenset(
    {
        "model_ddgan_gaussian_chain",
        "model_ae_ddgan_gaussian_chain",
        "model_ddgan_bimodal_chain",
        "model_ae_ddgan_bimodal_chain",
    }
)
STRETCH_FIXTURE_CASE_IDS = {
    "misgan_block_recovery": "api-misgan-block-recovery",
    "misgan_block_neural_imputer": "api-misgan-block-neural-imputer",
    "misgan_mcar_p50_recovery": "api-misgan-mcar-p50-recovery",
    "misgan_mcar_p80_anchored": "api-misgan-mcar-p80-anchored",
    "pacgan_unequal_mass": "api-pacgan-unequal-mass",
    "pacgan8_grid100": "api-pacgan8-grid100",
    "ring16_bigan_continuous": "api-ring16-bigan-continuous",
    "alice_spiral_bigan": "api-alice-spiral-bigan",
    "infogan_anisotropic_modes": "api-infogan-anisotropic-modes",
    "cgan_overlapping_classes": "api-cgan-overlapping-classes",
    "cramer_energy_overlap": "api-cramer-energy-overlap",
    "cyclegan_unpaired_constellation": "api-cyclegan-unpaired-constellation",
}
STRETCH_FIXTURE_TASKS = frozenset(STRETCH_FIXTURE_CASE_IDS)
STRETCH_TASKS = frozenset(
    {
        "misgan_block_recovery",
        "misgan_block_neural_imputer",
        "misgan_mcar_p50_recovery",
        "misgan_mcar_p80_anchored",
        "ring16_bigan_continuous",
        "pacgan_unequal_mass",
        "pacgan8_grid100",
        "grid100_mlp16",
        "grid100_res16_tanh",
        "grid100_res16_scaled_tanh",
    }
)
HELD_TIER34_TASKS = frozenset(
    {
        "zero_tanh_rail_escape",
        "grid100_mlp16",
        "grid100_res16_tanh",
        "grid100_res16_scaled_tanh",
        *STRETCH_FIXTURE_TASKS,
    }
)
CONTINUATION_PARENTS = {
    "grid100_14k": "grid100",
    "rotated100_14k": "rotated100",
    "staggered100_14k": "staggered100",
    "ring_extension": "ring_hold",
    "five_word_joint_hold": "five_word_joint_smoke",
}



def load_candidate_card(root: Path, candidate_id: str) -> dict[str, Any]:
    from experiments.forge.contracts import read_json
    for subdir in ("configs/forge/ideas", "configs/forge/configurations"):
        path = Path(root) / subdir / f"{candidate_id}.json"
        if path.is_file():
            return read_json(path)
    raise FileNotFoundError(f"candidate declaration not found: {candidate_id}")


def adapt_candidate_for_task(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Legacy accessor: return an independent, scientifically unchanged card.

    Resource and architecture binding belongs to Forge's declared task resolver;
    this function must never alter priors, policies, schedules, guards or noise.
    """
    return deepcopy(candidate)


def ensure_formulaic_attempt_coverage(root: Path, attempt_state_root: Path,
                                     gpus: list[str], *, json_mode: bool = False,
                                     tasks=None, force: bool = False) -> dict[str, Any]:
    """Refuse the unregistered executor before creating directories or launching work."""
    raise RetiredCoverageExecutionError(RETIREMENT_REASON)


def execute_single_task(*args, **kwargs):
    """Old direct callers must also pass through declared Forge admission."""
    raise RetiredCoverageExecutionError(RETIREMENT_REASON)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--attempt-state-root", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--tasks", default="stretch")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-batch", type=Path)
    parser.parse_args(argv)
    print(json.dumps({"status": "BLOCKED", "execution_authorized": False,
                      "qualification_input": False, "reason": RETIREMENT_REASON}), flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
