"""Pin an unexecuted API blueprint and its frozen source utilities."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
API = ROOT / 'quality/ra8/integration-contract/check_api.py'
NATIVE = Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/toy_models.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = HERE / 'SKELETON-FROZEN.json'
    assert not target.exists()
    paths = [API, NATIVE,
             ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra8-quality/resolution-api/check_api.py',
             ROOT / 'configs/overrides-CB64-RA9.json', ROOT / 'quality/ra9/READY.json',
             HERE.parent / 'mean-checkpoint-design/FROZEN.json']
    assert all(path.suffix != '.pt' for path in paths)
    source_sha256 = {str(path): sha(path) for path in paths}
    assert source_sha256[str(API)] == '36ab993ffcd9408ae84f9c35a1d35c41cb6d40176e42f54121878d416cddac61'
    assert source_sha256[str(NATIVE)] == '70f32797882de0dc88085afcf5e05821b65eb5f248c9690818a2494225d782a3'
    value = dict(status='SOURCE_ONLY_UNEXECUTED_SKELETON', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source_sha256,
        local_sha256={str(path): sha(path) for path in sorted(HERE.iterdir()) if path.is_file()},
        checkpoint_reads=False, Torch_import=False, numerical_tests=False, production_edits=False,
        optimizer_updates=0, sample_calls=0, RNG_draws=0, quality_samples=0, CUDA_context=False,
        execution_blocked=True, final_source_fields_and_fixture_binding_pending=True,
        final_receipt_target=str(ROOT / 'quality/ra10/integration-contract/receipt.json'))
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=value['status'], skeleton_sha256=sha(HERE / 'check_api_skeleton.py'),
                         seal_sha256=sha(target)), sort_keys=True))


if __name__ == '__main__':
    main()
