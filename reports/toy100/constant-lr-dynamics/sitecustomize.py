"""Install one constant-LR dynamic before the probe captures Adam.step.

The screen puts this directory on PYTHONPATH. With ``K3P_DYNAMICS`` unset this
file does nothing, and the probe is the constant-LR K3P baseline.
"""
import os
import sys

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
elif name:
    sys.stderr.write(f"unknown K3P_DYNAMICS={name}\n")
    raise SystemExit(2)
