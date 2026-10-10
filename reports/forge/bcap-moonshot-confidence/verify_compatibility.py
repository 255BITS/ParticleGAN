"""Original actual-develop comparator with one extra checked inactive default.

No historical comparator assertion or receipt is changed. The adapted test
exists only in the outside-Git software archive. Active confidence arithmetic
and resume are checked separately by test_bcap_confidence_mobility.py.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('confidence_compatibility',
    ROOT / 'reports/forge/bcap-three-phase/verify_compatibility.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.INACTIVE_DEFAULTS = {**module.INACTIVE_DEFAULTS, 'transport_mobility_mode': 'none'}

if __name__ == '__main__':
    module.main()
