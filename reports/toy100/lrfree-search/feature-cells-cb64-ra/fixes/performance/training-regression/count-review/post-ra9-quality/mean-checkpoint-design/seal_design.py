"""Seal a source-only contract. No Torch, checkpoint read, test or model run."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-transport'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = HERE / 'FROZEN.json'
    assert not target.exists() and not (HERE / 'receipt.json').exists()
    sources = [ROOT / f'pkg-CB64-RA9/particlegan/{name}.py' for name in
               ('feature_cells', 'training', 'continuous', 'row_evidence', 'birth_phase', 'anchor_birth')]
    sources += [ROOT / 'configs/overrides-CB64-RA9.json', ROOT / 'quality/ra9/READY.json',
                ROOT / 'quality/ra9/COMPOSITION.json', OWNER / 'witness.py', OWNER / 'transport.py',
                OWNER / 'run_prototype.py', OWNER / 'DESIGN.md', OWNER / 'PROTOCOL.json',
                OWNER / 'SOURCE-FROZEN.json',
                HERE.parent / 'mean-witness-review/FROZEN.json']
    assert all(path.suffix != '.pt' for path in sources)
    inputs = {str(path): sha(path) for path in sources}
    assert inputs[str(ROOT / 'quality/ra9/READY.json')] == 'ba558fff064f5e5613656a1c1550431091efc9989f4dd34f1d85a58d17d55990'
    assert inputs[str(OWNER / 'SOURCE-FROZEN.json')] == '9818d0439d36b6801ee9c5c21554c5e9b504bf1f1494b1443ebcc89d1ee520ab'
    value = dict(status='SOURCE_ONLY_DESIGN', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 source_sha256=inputs, design_sha256=sha(HERE / 'DESIGN.md'),
                 production_code=False, law_changes=False, tests=False, Torch_import=False,
                 checkpoint_reads=False, model_forwards=False, RNG_draws=0, CUDA_context=False,
                 quality_acceptance=False, backend_schema_if_integrated=9, trainer_schema=5)
    (HERE / 'receipt.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    frozen = dict(status=value['status'], post_exit_design_only=True, input_sha256=inputs,
                  receipt_sha256=sha(HERE / 'receipt.json'),
                  local_sha256={str(path): sha(path) for path in sorted(HERE.iterdir()) if path.is_file()})
    target.write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=value['status'], receipt_sha256=frozen['receipt_sha256'],
                         FROZEN_sha256=sha(target), source_files=len(inputs)), sort_keys=True))


if __name__ == '__main__':
    main()
