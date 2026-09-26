"""Load a K3P relief before the probe captures ``Adam.step``.

Activated only when ``K3P_RELIEF`` is set. The screen launcher puts this
directory on ``PYTHONPATH`` so Python imports it at startup.
"""
import os

name = os.environ.get("K3P_RELIEF")
if name:
    import relief
    relief.install(name)
