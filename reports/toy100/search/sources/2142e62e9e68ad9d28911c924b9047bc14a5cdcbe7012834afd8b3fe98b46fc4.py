"""Declared one-variable staggered repairs around the fixed noise mechanism."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
source_path = ROOT/'artifacts/toy100/instance_noise_probe.py'
spec = importlib.util.spec_from_file_location('staggered_base_probe', source_path)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
candidate = json.loads((ROOT/'configs/toy100/search_staggered.json').read_text())[int(sys.argv[1])]
probe.PEAK_SIGMA['sigma05'] = candidate['input_noise_std']
probe.DECAY_END_FRACTION = candidate['input_noise_anneal_end']
original_train = probe.runner.train
def train(config, directory):
    config.update(candidate['overrides'], name=candidate['name'])
    return original_train(config, directory)
probe.runner.train = train
original_provenance = probe.runner._source_provenance
def provenance():
    result = original_provenance()
    result['source_sha256']['artifacts/toy100/staggered_probe.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['repair_candidate'] = candidate
    return result
probe.runner._source_provenance = provenance
output = ROOT/'artifacts/toy100/staggered-repairs-cpu'/candidate['name']
output.mkdir(parents=True, exist_ok=True)
(output/'wrapper_probe_source.py').write_bytes(Path(__file__).read_bytes())
raise SystemExit(probe.main(['--variant','sigma05','--problem','staggered100','--device','cpu','--output',str(output)]))
