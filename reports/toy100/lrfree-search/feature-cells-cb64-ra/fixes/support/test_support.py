"""Private support variance patch: fitted-state isolation and fixed-fixture gates."""
import argparse
import os
import sys
sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
parser.add_argument('--small', action='store_true')
parser.add_argument('--output', required=True)
args = parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES='0' if args.device == 'cuda' else '',
                  CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  PYTHONDONTWRITEBYTECODE='1')

import hashlib
import importlib
import json
import time
import copy
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent
OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929')
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929')
DEVICE = torch.device('cuda:0' if args.device=='cuda' else 'cpu')
sys.path.insert(0,str(OLD/'geometry'))
import run_validation as common
sys.path.insert(0,str(PREV/'scaling_a'))
import shared_toy as cost
sys.path.insert(0,str(PREV/'geometry_a'))
import toy_family as geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def oracle_support(snapshot, features, calibration):
    # Run the unchanged archived scoring and hypothesis-test source with just
    # the analytically specified predictive scale. This also verifies that the
    # patch retained equality handling, conformal floor, BH, and both guards.
    oracle = copy.copy(snapshot)
    oracle.cell_scale = snapshot.cell_scale*(1.+snapshot.reference_counts.clamp_min(1).double().reciprocal()).sqrt()
    oracle.null_scores = oracle._scores_metric(oracle.transform(calibration)).sort().values
    return oracle.support(features)


def invariants(reference, fix, real, query, seed, cells=64, rank=8):
    ga,gb = generator(seed),generator(seed)
    old = reference.FeatureCellSnapshot.fit(real,generator=ga,cells=cells,rank=rank)
    new = fix.FeatureCellSnapshot.fit(real,generator=gb,cells=cells,rank=rank)
    fields = ('mean','scale','basis','centers','reference_counts','cell_scale',
              'real_representatives','real_representative_rows','real_calibration_counts')
    for field in fields:
        torch.testing.assert_close(getattr(old,field),getattr(new,field),rtol=5e-13,atol=5e-14,msg=field)
    assert torch.equal(ga.get_state(),gb.get_state()),'support patch consumed RNG'
    assert torch.equal(old.assign(query)[0],new.assign(query)[0]),'support patch changed cells'
    left,right = old.cell_comparison(query),new.cell_comparison(query)
    for field in left:
        if left[field].is_floating_point():
            torch.testing.assert_close(left[field],right[field],rtol=5e-13,atol=5e-14,msg=field)
        else:
            assert torch.equal(left[field],right[field]),f'categorical evidence changed: {field}'
    expected_scale = new.cell_scale*(1.+new.reference_counts.clamp_min(1).double().reciprocal()).sqrt()
    assert torch.equal(expected_scale,new.support_cell_scale)
    actual = new.support(query)
    expected = oracle_support(old,query,real[1::2])
    torch.testing.assert_close(actual[2],expected[2],rtol=5e-13,atol=5e-14)
    assert torch.equal(actual[0],expected[0]) and torch.equal(actual[1],expected[1])
    return old,new,dict(fit_geometry_unchanged=True,categorical_evidence_unchanged=True,
                       rng_unchanged=True,scalar_formula_and_empirical_ranks_match=True,
                       formula_max_abs_error=float((actual[2]-expected[2]).abs().max()))


@torch.no_grad()
def main():
    if args.device=='cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.2,0)
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        assert uuid.removeprefix('GPU-').lower()=='72c1b506-891d-b8bc-b353-e020585e1c47'
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    else:
        assert not torch.cuda.is_initialized()
        uuid = None
    common.namespace('support_test_reference',OLD/'pkg-CB64-RA'/'particlegan')
    common.namespace('support_test_fix',ROOT/'pkg'/'particlegan')
    common.namespace('support_test_e22',common.REFERENCE)
    reference = importlib.import_module('support_test_reference.feature_cells')
    fix = importlib.import_module('support_test_fix.feature_cells')
    shared = SimpleNamespace(cb=reference,ref=importlib.import_module('support_test_e22.birth_death'),
        cb_recipe=importlib.import_module('support_test_reference.recipes').Recipe,
        ref_recipe=importlib.import_module('support_test_e22.recipes').Recipe,
        geometry=geometry,cost=cost,bundle=geometry.load_bundle(PREV/'geometry_a'/'bundle.pt'),
        config=json.loads(common.CONFIG.read_text()),ref_config=json.loads(common.REF_CONFIG.read_text()))
    files = [Path(__file__),ROOT/'pkg/particlegan/feature_cells.py',ROOT/'SUPPORT-SCORE.diff',
             ROOT/'PROTOCOL.md',common.CONFIG,common.REF_CONFIG,
             PREV/'geometry_a'/'bundle.pt',PREV/'geometry_a'/'toy_family.py',
             PREV/'scaling_a'/'shared_toy.py',OLD/'geometry'/'run_validation.py',
             OLD/'pkg-CB64-RA/particlegan/feature_cells.py']
    hashes = {str(p):sha(p) for p in files}
    result = dict(status='PASS',scope='support-only diagnosis; quality gates reported separately',
                  device=str(DEVICE),physical_gpu0_uuid=uuid,source_sha256=hashes,
                  seeds=dict(geometry=geometry.SEED,cost=cost.SEED),unit_cases=[],cases=[])
    units = [('tiny-odd',torch.arange(14,dtype=torch.float64).reshape(7,2)),
             ('constant',torch.ones(16,3,dtype=torch.float64)),
             ('duplicate-guard',torch.cat((torch.zeros(96,3),torch.eye(3),torch.ones(1,3))).double()),
             ('rare-counts',torch.cat((torch.arange(240,dtype=torch.float64).reshape(120,2)/1000,
                                       torch.tensor([[4.,1.],[4.1,1.1],[4.2,1.2],[4.3,1.3]]))))]
    for name,real in units:
        real = real.to(DEVICE)
        query = real.flip(0)+.01
        old,new,checks = invariants(reference,fix,real,query,geometry.SEED)
        if name in ('constant','duplicate-guard'):
            assert not bool(new.support(query)[0].any())
        result['unit_cases'].append(dict(name=name,rows=len(real),counts=new.reference_counts.tolist(),**checks))
    invalid = [torch.zeros(5,2),torch.zeros(6,0),torch.ones(6,2,dtype=torch.long),
               torch.full((6,2),float('nan')),torch.full((6,2),float('inf'))]
    for real in invalid:
        try:
            fix.FeatureCellSnapshot.fit(real.to(DEVICE),generator=generator(geometry.SEED))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid features accepted')
    # Changing held-out rows cannot change the fitted score's geometry/variance.
    real = torch.arange(48,dtype=torch.float64).reshape(16,3).to(DEVICE)
    changed = real.clone();changed[1::2]+=100.
    a = fix.FeatureCellSnapshot.fit(real,generator=generator(geometry.SEED),cells=4,rank=2)
    b = fix.FeatureCellSnapshot.fit(changed,generator=generator(geometry.SEED),cells=4,rank=2)
    for field in ('mean','scale','basis','centers','reference_counts','support_cell_scale'):
        assert torch.equal(getattr(a,field),getattr(b,field)),f'held-out data fitted metric: {field}'
    assert not torch.equal(a.null_scores,b.null_scores)
    result['malformed_inputs_rejected'] = len(invalid)
    result['heldout_isolation'] = True
    cases = [('geometry',2048,'fold',2,'trained600'),('geometry',2048,'fold',128,'trained600'),
             ('cost',2048,'highdim',128,'frozen_toy_head')]
    if not args.small:
        cases = [('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
                 for n in (1024,2048,4096)]
        cases += [('geometry',2048,'fold',2,'trained600')]
        cases += [('cost',n,m,128 if m=='highdim' else 8,'frozen_toy_head')
                  for m in cost.SCENARIOS for n in cost.SIZES]
    for case in cases:
        ev = common.Evaluator(*case,shared)
        trainer,bd = common.make_trainer(ev,'cb64_ra',shared)
        R = bd._features(trainer,ev.real_raw).to(DEVICE)
        q = bd._features(trainer,ev.G(ev.z)).to(DEVICE)
        old,new,checks = invariants(reference,fix,R,q,ev.seed)
        oldflags,newflags = old.support(q)[0].cpu(),new.support(q)[0].cpu()
        row = dict(fixture=case,baseline=ev.detector(oldflags),proposed=ev.detector(newflags),
                   removed_flags=(oldflags&~newflags).nonzero().flatten().tolist(),
                   added_flags=(newflags&~oldflags).nonzero().flatten().tolist(),**checks)
        result['cases'].append(row)
        assert ev.state_hash()==ev.initial_state_hash
        print(json.dumps(dict(event='case',fixture=case,
            metrics={key:{k:v for k,v in row[key].items() if k!='flag_ids'} for key in ('baseline','proposed')})),flush=True)
    assert hashes=={str(p):sha(p) for p in files},'source changed during test'
    result['cuda_initialized']=torch.cuda.is_initialized()
    assert result['cuda_initialized']==(args.device=='cuda')
    result['quality_cases_passed']=sum(row['proposed']['passes'] for row in result['cases'])
    result['quality_cases_total']=len(result['cases'])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='complete',status='PASS',quality_cases_passed=result['quality_cases_passed'],
                         quality_cases_total=len(cases),cuda_initialized=result['cuda_initialized'])),flush=True)


if __name__=='__main__':
    main()
