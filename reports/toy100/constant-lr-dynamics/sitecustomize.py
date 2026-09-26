"""Install one constant-LR dynamic before the probe captures Adam.step.

The screen puts this directory on PYTHONPATH. With ``K3P_DYNAMICS`` unset no
dynamic is installed, and the probe is the constant-LR K3P baseline.

The unequal-mass penalty fix is installed either way. It only changes the
frozen ``scaled_penalty`` call that passes ``ema_critic``; ring, hold, and
stay do not make that call. That fix and ``lookahead_minmax`` are loaded by
file path so this process does not import ``particlegan`` before the probe
sets its threads and deterministic flags.
"""
import importlib.util
import os
import sys
from pathlib import Path

_DYNAMICS = Path(__file__).resolve().parents[3] / "particlegan" / "dynamics"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_k3p_dyn_{name}", _DYNAMICS / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_load("unequal_penalty").install()

name = os.environ.get("K3P_DYNAMICS")
if name == "unit_rms":
    from particlegan.dynamics.unit_rms import install
    install()
elif name == "pair_chord":
    from particlegan.dynamics.pair_chord import install
    install()
elif name == "shared_batch":
    from particlegan.dynamics.shared_batch import install
    install()
elif name == "lookahead_minmax":
    _load("lookahead_minmax").install()
elif name:
    sys.stderr.write(f"unknown K3P_DYNAMICS={name}\n")
    raise SystemExit(2)
