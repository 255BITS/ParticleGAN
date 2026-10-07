"""Reuse the frozen saved-sample publisher under this diagnostic's declaration."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from benchmarks.toy_audit import gaussian_combined_magnitude as study
from experiments.forge.contracts import atomic_json,file_hash

path=ROOT/'reports/forge/bcap-past-extrapolation/publish.py'
spec=importlib.util.spec_from_file_location('combined_saved_publisher',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.DEST=Path(__file__).resolve().parent
module.PROTOCOL=study.PROTOCOL
module.declaration=study.declaration

if __name__=='__main__':
    module.main()
    verification=module.DEST/'verification.json'
    import json
    result=json.loads(verification.read_text())
    result['publication_adapter_sha256']=file_hash(Path(__file__))
    result['publication_source_sha256']=file_hash(path)
    atomic_json(verification,result)
