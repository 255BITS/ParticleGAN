"""Write a bounded outside-Git adapter for the archived develop comparison.

Only checked inactive recipe metadata and nested source-provenance ancestry are
adapted. Every actual model, optimizer, stream, observation and loss stays strict.
"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.output.resolve().is_relative_to(ROOT):
        raise ValueError('Adapted software probe belongs outside Git')
    source = (ROOT / 'tests/test_bcap_integration_compatibility.py').read_text()
    old = '    "kinetic_transport_projections": 32,\n'
    new = old + ('    "critic_step_mode": "none",\n'
                 '    "optimizer_svd_backend": "native",\n'
                 '    "kinetic_transport_local_geometry": "isotropic",\n')
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '''            if revision != old_revision:
                assert revision["original_task_commit"] == DEVELOP
                assert revision["original_task_path"] == f"configs/forge/tasks/{name}.json"
                _assert_equal(old_revision, revision["previous_revision"], name + ".original-revision")'''
    new = '''            if revision != old_revision:
                ancestry = revision
                while ancestry and ancestry.get("original_task_commit") != DEVELOP:
                    ancestry = ancestry.get("previous_revision")
                assert ancestry and ancestry["original_task_commit"] == DEVELOP
                assert ancestry["original_task_path"] == f"configs/forge/tasks/{name}.json"
                _assert_equal(old_revision, ancestry["previous_revision"], name + ".original-revision")'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '                    assert revision["previous_source_sha256"][path] == old_hash'
    new = '''                    ancestry = revision
                    while ancestry and ancestry.get("previous_source_sha256", {}).get(path) != old_hash:
                        ancestry = ancestry.get("previous_revision")
                    assert ancestry and ancestry["previous_source_sha256"][path] == old_hash'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(source)


if __name__ == '__main__':
    main()
