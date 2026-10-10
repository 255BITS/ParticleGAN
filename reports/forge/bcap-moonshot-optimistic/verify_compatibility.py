"""Run original assertion-preserving comparator with checked inactive defaults."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "bcap-three-phase/verify_compatibility.py"
spec = importlib.util.spec_from_file_location("baseline_compatibility", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.INACTIVE_DEFAULTS = {**module.INACTIVE_DEFAULTS, "optimizer_optimism": "none"}
if __name__ == "__main__":
    module.main()
