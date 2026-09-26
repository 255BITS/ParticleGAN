"""Lookahead-minmax (Chavdarova et al., ICLR 2021) at the published defaults.

The inner optimizer is the host Adam, at the learning rate already on the
group. Fast weights move for ``k`` paired updates. The slow weights of every
player then move ``alpha`` of the way to the fast weights, and the fast
weights are reset to them (Algorithm 1). ``k=5`` and ``alpha=0.5``.

The slow step is joint, not one Lookahead wrapped on each optimizer. It runs
only after every Adam player seen so far — the critic, and the generator
optimizer that holds G and the particles — has taken ``k`` fast steps. Adam's
moments stay with the fast weights; they are not reset and not rescaled.
"""
from __future__ import annotations

import atexit
import json

import torch

K = 5
ALPHA = 0.5

receipt = {"mechanism": "lookahead_minmax", "fast_steps": 0, "syncs": 0, "k": K, "alpha": ALPHA, "players": 0}
_ORIGINAL = None
_SLOW: dict[int, torch.Tensor] = {}
_PARAMS: dict[int, torch.nn.Parameter] = {}
_COUNTS: dict[int, int] = {}
_PLAYERS: list[int] = []
_LOGGED = False
_ATEXIT = False


def uninstall() -> None:
    """Restore ``Adam.step``. Used by tests; training processes leave it installed."""
    global _ORIGINAL, _LOGGED
    if _ORIGINAL is None:
        return
    torch.optim.Adam.step = _ORIGINAL
    _ORIGINAL = None
    _SLOW.clear()
    _PARAMS.clear()
    _COUNTS.clear()
    _PLAYERS.clear()
    _LOGGED = False
    receipt["fast_steps"] = 0
    receipt["syncs"] = 0
    receipt["players"] = 0


def install() -> None:
    """Replace ``Adam.step`` until process exit. Call before anything captures it."""
    global _ORIGINAL, _ATEXIT
    if _ORIGINAL is not None:
        return
    _ORIGINAL = torch.optim.Adam.step
    torch.optim.Adam.step = _step
    _step._lookahead_minmax = True
    if not _ATEXIT:
        _ATEXIT = True
        atexit.register(_emit)
    print(json.dumps({
        "event": "dynamics",
        "name": "lookahead_minmax",
        "setting": "joint Lookahead-minmax k=5 alpha=0.5 on D, G, and particles; fast Adam at the group LR",
    }), flush=True)


def _emit() -> None:
    receipt["players"] = len(_PLAYERS)
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def _remember(optimizer) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            key = id(parameter)
            if key not in _SLOW:
                _SLOW[key] = parameter.detach().clone()
                _PARAMS[key] = parameter


def _synchronize() -> None:
    global _LOGGED
    with torch.no_grad():
        for key, slow in _SLOW.items():
            fast = _PARAMS[key]
            slow.add_(fast.data - slow, alpha=ALPHA)
            fast.data.copy_(slow)
    receipt["syncs"] += 1
    if not _LOGGED:
        _LOGGED = True
        print(json.dumps({
            "event": "dynamics_step",
            "name": "lookahead_minmax",
            "syncs": receipt["syncs"],
            "fast_steps": receipt["fast_steps"],
            "players": len(_PLAYERS),
        }), flush=True)


def _step(optimizer, closure=None, *args, **kwargs):
    _remember(optimizer)
    key = id(optimizer)
    if key not in _COUNTS:
        _COUNTS[key] = 0
        _PLAYERS.append(key)
    loss = _ORIGINAL(optimizer, closure, *args, **kwargs)
    receipt["fast_steps"] += 1
    _COUNTS[key] += 1
    # Joint: both players (and any later one) must have finished k fast steps.
    # A lone first optimizer must not backtrack before its opponent exists.
    if len(_PLAYERS) >= 2 and all(_COUNTS[player] >= K for player in _PLAYERS):
        _synchronize()
        for player in _PLAYERS:
            _COUNTS[player] = 0
    return loss
