"""Source-bound dormant-path parity and fixed-target remembered-rest receipt."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import forward_kl_gh9_stress as frozen
from reports.toy100 import forward_kl_gh9_remembered as wrapper
from reports.toy100.forward_kl_free_filter import cross_entropy
from reports.toy100.sample_anchor_free1200 import initial_support, load_states


ROOT=Path(__file__).resolve().parents[2]
SOURCES=(
    'reports/toy100/forward_kl_gh9_remembered_filter.py',
    'reports/toy100/forward_kl_gh9_remembered.py',
    'reports/toy100/forward_kl_remembered_donor_rescue.py',
    'reports/toy100/forward_kl_gh9_stress.py',
    'reports/toy100/forward_kl_chunked.py',
    'reports/toy100/forward_kl_free_filter.py',
    'reports/toy100/sample_anchor_free1200.py',
    'reports/toy100/sample_anchor_mmd_filter.py',
    'reports/toy100/sample_anchor_local_mmd_filter.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_early_geometry.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canon(value):return json.dumps(value,sort_keys=True,separators=(',',':'),
                                   allow_nan=False).encode()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    paths={n:(ROOT/n if (ROOT/n).exists() else args.code_root/n) for n in SOURCES}
    hashes={n:sha(p.read_bytes()) for n,p in paths.items()}
    if (hashes['reports/toy100/forward_kl_gh9_remembered.py']!=
            '86639dfefe0c11f82eccbd8dfe106e5797f36e75cf09a906de1768b02c28e1df'):
        raise RuntimeError('wrapper source differs from frozen source')
    args.output.mkdir(parents=True)
    for n,p in paths.items():
        target=args.output/'source'/n;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    declaration=dict(scope='one-bank cold/warm dormant-path parity plus synthetic rest/no-signal',
        source=hashes,method=wrapper.METHOD,no_training=True,no_seed_sweep=True)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    cold,warm,input_hashes=load_states()
    rng=torch.random.get_rng_state().clone()
    means=mode_hold.ring_means()
    histories={};cases={}
    for name,state,sigma in (('cold1',cold,0.),('warm1324',warm,.029)):
        stream=torch.Generator().set_state(state['rng']['data'])
        bank=mode_hold.sample_ring(means,128,mode_hold.SIGMA,stream)
        points=initial_support(state).double()
        pristine=points.clone();bank_pristine=bank.clone()
        old,oldrow=frozen.optimize(bank,bank,points,.031286240422040236,sigma,
                                   means=means)
        new,newrow=wrapper.optimize(bank,bank,points,.031286240422040236,sigma,
                                    means=means)
        neutral,neutralrow,rule=wrapper.propose_target(bank,bank,points,
                                                      .031286240422040236,sigma)
        variance=.031286240422040236**2+sigma**2
        normal=(oldrow['selected']!='EXACT_REST' and torch.equal(old,new) and
                oldrow==newrow and torch.equal(old,neutral) and
                all(neutralrow[key]==value for key,value in oldrow.items()
                    if key not in ('initial_quality','final_quality')) and
                'initial_quality' not in neutralrow and
                'final_quality' not in neutralrow and
                torch.equal(points,pristine) and torch.equal(bank,bank_pristine) and
                abs(float(cross_entropy(*rule,neutral,variance))-
                    neutralrow['final_audit9'])<1e-12)
        if not normal:raise RuntimeError('normal wrapper/neutral API differs from frozen method')
        histories[name]=bank.tolist()
        cases[name]=dict(source_normal_selected=oldrow['selected'],
            full_frozen_receipt_sha256=sha(canon(oldrow)),
            full_explicit_wrapper_receipt_sha256=sha(canon(newrow)),
            neutral_receipt_sha256=sha(canon(neutralrow)),
            output_sha256=sha(new.contiguous().numpy().tobytes()),
            native_real_sha256=sha(bank.contiguous().numpy().tobytes()),
            ephemeral_GH9_locations_sha256=sha(rule[0].contiguous().numpy().tobytes()),
            ephemeral_GH9_weights_sha256=sha(rule[1].contiguous().numpy().tobytes()),
            exact_target_receipt_parity=True,oracle_free_target_parity=True)
    left=torch.tensor([[-2.,0.]],dtype=torch.float32).repeat(128,1)
    middle=torch.zeros((128,2),dtype=torch.float32)
    right=torch.tensor([[2.,0.]],dtype=torch.float32).repeat(128,1)
    history=torch.cat((left,right,middle),0)
    collapsed=torch.zeros((12,2),dtype=torch.float64)
    old,oldrow=frozen.optimize(history,middle,collapsed,.031286240422040236,.029,
                               means=means)
    selected,row,rule=wrapper.propose_target(history,middle,collapsed,
                                             .031286240422040236,.029)
    if (oldrow['selected']!='EXACT_REST' or not torch.equal(old,collapsed) or
            row['selected']!='REMEMBERED_GH9_DONOR' or
            row['remembered_search']['status']!='STRICT_FINITE_GH9_DONOR' or
            row['final_audit9']>=row['initial_audit9']-row['gh9_strict_tolerance'] or
            not torch.equal(collapsed,torch.zeros_like(collapsed)) or
            abs(float(cross_entropy(*rule,selected,.031286240422040236**2+.029**2))-
                row['final_audit9'])>1e-12):
        raise RuntimeError('synthetic exact-rest rescue failed')
    histories['fixed_target_trap']=history.tolist()
    cases['fixed_target_trap']=dict(frozen_selection=oldrow['selected'],
        rescued_selection=row['selected'],before_gh9=row['initial_audit9'],
        after_gh9=row['final_audit9'],remembered_search=row['remembered_search'],
        remembered_em_steps=len(row['remembered_em']),
        final_points=selected.tolist())
    empty=torch.zeros((128,2),dtype=torch.float32)
    no_signal,no_row=wrapper.optimize(empty,empty,collapsed,.031286240422040236,.029)
    if (no_row['selected']!='EXACT_REST' or
            no_row['remembered_search']['status']!='NO_SINGLE_REMEMBERED_DONOR' or
            not torch.equal(no_signal,collapsed)):
        raise RuntimeError('no-signal exact rest failed')
    histories['no_signal']=empty.tolist()
    cases['no_signal']=dict(selection=no_row['selected'],
        remembered_search=no_row['remembered_search'],exact_output_rest=True)
    if not torch.equal(torch.random.get_rng_state(),rng):
        raise RuntimeError('pure wrapper filter changed global RNG')
    (args.output/'full_histories.json').write_text(json.dumps(histories,
        separators=(',',':'),allow_nan=False)+'\n')
    result=dict(status='COMPLETE',declaration=declaration,input_hashes=input_hashes,
        cases=cases,full_histories_sha256=sha((args.output/'full_histories.json').read_bytes()),
        global_torch_rng_unchanged=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='REMEMBERED_WRAPPER_DONE',
        normal=[cases[k]['exact_target_receipt_parity'] for k in ('cold1','warm1324')],
        rescued=cases['fixed_target_trap']['after_gh9'],
        no_signal=cases['no_signal']['selection'])),flush=True)


if __name__=='__main__':main()
