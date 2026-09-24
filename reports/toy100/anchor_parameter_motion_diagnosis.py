"""One-state pre/native-start target fit comparison; no trainer promotion.

Run exactly one existing update from the completed borrowed hold state, then
fit that identical output target offline from the pre-G parameters. Measure
the native displacement's projection onto the pre-G output Jacobian kernel.
This isolates parameter motion invisible to the linearized clean outputs;
it does not establish a nonlinear minimum-distance theorem or its cause.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch.func import functional_call,jacrev
from benchmarks.locked_shared.mlp import SimpleMLPGenerator
from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import pr84_model_error_recovery as resume
from reports.toy100 import sample_anchor_rest_candidate as candidate
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_critic_refinement_capture import _sha,_clone
from reports.toy100.allocation_continuous_probe import verify_sources
from reports.toy100.coverage_fixed_eval import fixed_draw,score_support


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hold',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    declaration=json.loads((args.hold/'declaration.json').read_text())
    verify_sources(declaration['source'],archive=args.hold/'source')
    evidence=json.loads((args.hold/'forks/candidate.json').read_text())
    receipt=evidence['dynamics_receipt']
    raw=args.hold/receipt['final_state_file']
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==receipt['final_state_file_sha256']
    saved=torch.load(raw,weights_only=True,map_location='cpu')
    assert _sha(saved)==receipt['final_snapshot_sha256']
    assert saved['noise']['step_calls']==2400
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(lr_floor=1.,lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
    recipe,noise,_=declared_recipe(config)
    args.output.mkdir(parents=True,exist_ok=False)
    sources=dict(declaration['source'])
    for name in ['reports/toy100/anchor_parameter_motion_diagnosis.py',
                 'reports/toy100/pr84_model_error_recovery.py',
                 'reports/toy100/pr84_critic_refinement_resume.py']:
        sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in sources:
        p=args.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/name).read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(dict(source=sources,
        scope='one actual update2401 plus offline identical-target fit from pre-G; not cold promotion',
        input_state_sha256=_sha(saved),method=candidate.METHOD),indent=2)+'\n')
    capture={}
    ordinary=candidate.RestOnFailureRecorder.correct
    def observed(recorder,optimizer):
        capture['pre']=[p.clone() for p in recorder.g_base]
        capture['native']=[p.detach().clone() for p in recorder._params(optimizer)]
        ordinary(recorder,optimizer)
        capture['accepted']=[p.detach().clone() for p in recorder._params(optimizer)]
        capture['correction']=_clone(recorder.corrections[-1])
    torch.set_num_threads(1)
    with patch.object(candidate.RestOnFailureRecorder,'correct',observed),patch.object(
        resume,'pr84_critic_refinement_finite',candidate.sample_anchor_rest_candidate):
        result=resume.run_continuation(saved,recipe,noise,completed_steps=2400,target_steps=2401)
    assert result['receipt']['updates']==1
    torch.save(capture,args.output/'captured-parameters.pt')
    generator=SimpleMLPGenerator(mode_hold.Z_DIM,mode_hold.HIDDEN,mode_hold.N_HIDDEN,2)
    prior=torch.nn.Parameter(capture['pre'][-1].clone())
    with torch.no_grad():
        for p,value in zip(generator.parameters(),capture['pre'][:-1]):p.copy_(value)
    names=list(dict(generator.named_parameters()))
    def function(parameters,z):return functional_call(generator,dict(zip(names,parameters)),(z,))
    derivatives=jacrev(function,argnums=(0,1))(tuple(generator.parameters()),prior)
    target=torch.tensor(capture['correction']['mm']['target'],dtype=prior.dtype)
    jac=torch.cat([v.reshape(target.numel(),-1) for v in [*derivatives[0],derivatives[1]]],1).double()
    u,s,vh=torch.linalg.svd(jac,full_matrices=False)
    active=s>1e-6*s.max();basis=vh[active]
    def vector(values):return torch.cat([p.detach().double().flatten() for p in values])
    base=vector(capture['pre'])
    def geometry(values):
        delta=vector(values)-base;visible=basis.T@(basis@delta);null=delta-visible
        return dict(total_norm=float(delta.norm()),linear_output_norm=float((jac@delta).norm()),
            null_norm=float(null.norm()),null_fraction=float(null.norm()/delta.norm()) if delta.norm()>0 else 0.)
    fit=fit_output_targets(generator,prior,target)
    prefit=[p.detach() for p in generator.parameters()]+[prior.detach()]
    support=generator(prior).detach()
    report=dict(scope='local parameter-motion diagnostic, no new controller or long-run claim',
        pre_jacobian=dict(rank=int(active.sum()),singular_min=float(s[-1]),singular_max=float(s[0])),
        native=geometry(capture['native']),post_native_fit=geometry(capture['accepted']),
        pre_start_fit=geometry(prefit),fit=fit,original_correction=capture['correction'],
        one_update_receipt=result['receipt'],pre_start_support=support.tolist(),
        pre_start_grade=score_support(support,*fixed_draw(2401,support),mode_hold.ring_means()),
        accepted_state_sha256=_sha(result['state']))
    (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['pre_jacobian','native','post_native_fit','pre_start_fit']},indent=2))


if __name__=='__main__':main()
