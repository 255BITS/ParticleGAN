"""ROOT-paid CPU software runner; definitions are inert on import.

It runs exactly the authored15 controls and returns their measured outcomes.
No scientific admission or numerical learner is launched.
"""
import json
from pathlib import Path
import time
import unittest


def run_controls():
    import torch
    from .atlas844_contract import SOFTWARE_CHECKS, EXTRA_OPERATIONS
    root=Path(__file__).resolve().parents[2]
    import importlib.util
    path=root/'tests/test_forge_atlas844_radius_observer.py'
    spec=importlib.util.spec_from_file_location('atlas844_private_software_controls',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names=['test_'+name for name in SOFTWARE_CHECKS]
    if set(names)!={name for name in vars(module.Atlas844SoftwareControls) if name.startswith('test_')}:
        raise ValueError('software control roster differs')
    original_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    checks={}
    started=time.monotonic()
    try:
        for name in names:
            result=unittest.TestResult()
            module.Atlas844SoftwareControls(name).run(result)
            checks[name[5:]]='PASS' if result.wasSuccessful() and result.testsRun==1 else 'FAIL'
            if not result.wasSuccessful():
                for _,error in result.errors+result.failures:
                    print(error,flush=True)
    finally:
        torch.set_num_threads(original_threads)
    operations=dict(module.ADDED_OPERATIONS)
    status='PASS' if all(value=='PASS' for value in checks.values()) and operations==dict.fromkeys(EXTRA_OPERATIONS,0) else 'FAIL'
    return dict(schema='forge_atlas844_software_control_result_v1',status=status,
        checks=checks,added_operations=operations,software_elapsed_seconds=time.monotonic()-started,
        evidence_scope='CPU_SOFTWARE_FIXTURES_ONLY',scientific_attempts=0,
        actual_cuda_success='UNKNOWN',full_arbitrary_callable_leaf_scope='UNKNOWN')


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run_controls()
    path=Path(args.output)
    if path.exists() or path.is_symlink():raise ValueError('new owned software report only')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(result,handle,sort_keys=True,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(result,sort_keys=True),flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    main()
