"""Read-only three-bank critic tracking diagnosis at one captured PR84 state.

One fixed original D state, three nonoverlapping 1024-pair banks A/B/C from
cloned host streams: A is an optional previously archived fit, B is one new
same-budget fit, C is evaluation only. All generator partial fields and Adam
proposals begin from the same saved G/prior/optimizer state. No host training,
new seed, target-center fit selection, or optimizer-state mutation occurs.
"""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import pr84_critic_relaxation as fit


def sha(data):
    return hashlib.sha256(data).hexdigest()


def draw_banks(pre,saved):
    """Recreate A/B then continue to C, with no change to live RNG."""
    if (pre['noise']['input_sigma'] != 0 or pre['rng']['output'] is not None
            or pre['noise']['output_sigma'] != saved['noise']['output_sigma']):
        raise ValueError('only the captured fixed global-output-noise host is supported')
    generator,_,prior=fit.modules(saved)
    stream=torch.Generator();stream.set_state(pre['rng']['data'])
    rows=[]
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.set_rng_state(pre['rng']['torch'])
        for _ in range(24):
            real=mode_hold.sample_ring(mode_hold.ring_means(),mode_hold.BATCH,mode_hold.SIGMA,stream)
            latent,indices=prior.sample(mode_hold.BATCH,generator=stream)
            clean=generator(latent)
            noise=torch.randn_like(clean)
            rows.append(dict(real=real,fake=clean+pre['noise']['output_sigma']*noise,
                             indices=indices,noise=noise))
    def joined(subset):
        return {key:torch.cat([row[key] for row in subset]) for key in ('real','fake')}
    return rows[:8],rows[8:16],rows[16:],joined(rows[:8]),joined(rows[8:16]),joined(rows[16:])


def per_batch_field(saved,critic,row,gan,step,sigma):
    """Actual non-saturating Rp G partial gradient, D fixed."""
    generator,_,prior=fit.modules(saved)
    params=list(generator.parameters())+list(prior.parameters())
    fake=generator(prior.z[row['indices']])+sigma*row['noise']
    loss=gan.g_loss(fit.smooth(critic,fake),fit.smooth(critic,row['real']))
    gradient=torch.autograd.grad(loss,params)
    flat=torch.cat([value.detach().double().flatten() for value in gradient])
    batch=dict(real=row['real'],indices=row['indices'],noise=row['noise'],sigma=sigma)
    proposal,_,_=fit.g_proposal(saved,critic,batch,gan,step)
    return flat,dict(loss=float(loss.detach()),rho=proposal['rho'],factor=proposal['factor'],
                     grade_before=proposal['grade_before'],grade_after=proposal['grade_after'],
                     raw_functional=proposal['raw_joint_parameter_descent']['vectors'],
                     accepted_functional=proposal['accepted_joint']['vectors'])


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return float(a@b/denominator) if denominator>0 else None


def finite_partial_check(saved,critic,row,gan,step,sigma,partial_gradient):
    """Central loss secant against an actual accepted G displacement."""
    batch=dict(real=row['real'],indices=row['indices'],noise=row['noise'],sigma=sigma)
    _,accepted,_=fit.g_proposal(saved,critic,batch,gan,step)
    base_g,_,base_z=fit.modules(saved)
    end_saved=deepcopy(saved)
    end_saved['generator']=accepted['generator']
    end_saved['prior']=accepted['prior']
    end_g,_,end_z=fit.modules(end_saved)
    base=list(base_g.parameters())+list(base_z.parameters())
    end=list(end_g.parameters())+list(end_z.parameters())
    direction=[(y-x).detach().clone() for x,y in zip(base,end)]
    flat=torch.cat([x.double().flatten() for x in direction])
    if float(flat.norm())==0:
        raise RuntimeError('actual accepted G displacement is zero')
    analytic=float(partial_gradient@flat)
    def loss_at(scale):
        with torch.no_grad():
            for parameter,old,delta in zip(base,base_values,direction):
                parameter.copy_(old+scale*delta)
            fake=base_g(base_z.z[row['indices']])+sigma*row['noise']
            return float(gan.g_loss(fit.smooth(critic,fake),fit.smooth(critic,row['real'])))
    base_values=[p.detach().clone() for p in base]
    half=.25
    secant=(loss_at(half)-loss_at(-half))/(2.*half)
    error=abs(secant-analytic)
    if error>max(1e-5,.01*abs(analytic)):
        raise RuntimeError(f'G partial gradient/finite secant mismatch: {analytic}, {secant}')
    return dict(actual_accepted_parameter_norm=float(flat.norm()),
                analytic_directional_derivative=analytic,
                central_loss_secant=secant,absolute_error=error,
                half_span_of_accepted_step=half)


def get_archived_a(path,step,source_sha):
    base=Path(path)
    with gzip.open(base/'fit/summary.json.gz','rt') as stream:
        summary=json.load(stream)
    if summary['declaration']['source']['reports/toy100/pr84_critic_relaxation.py']!=source_sha:
        raise RuntimeError('archived A fit source differs from active fit module')
    with gzip.open(base/'fit/fitted-critics-and-banks.pt.gz','rb') as stream:
        tensors=torch.load(stream,weights_only=True)
    row=next(item for item in summary['rows'] if item['step']==step)
    if fit.state_hash(tensors[step]['critic'])!=row['fitted_critic_sha256']:
        raise RuntimeError('archived A critic hash changed')
    return row,tensors[step],dict(summary_sha256=sha(gzip.decompress(
        (base/'fit/summary.json.gz').read_bytes())),
        tensors_sha256=sha(gzip.decompress(
        (base/'fit/fitted-critics-and-banks.pt.gz').read_bytes())))


def _main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--states',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--start-phase',default='post_accepted_d')
    parser.add_argument('--width',type=float,default=.15)
    parser.add_argument('--archived-a',type=Path)
    parser.add_argument('--expect-original-g-parity',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.width<=.15:
        raise ValueError('G stencil width outside original PR84 range')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    source=Path(__file__).read_bytes()
    (args.output/'pr84_critic_bank_swap.py').write_bytes(source)
    source_hashes={name:sha((ROOT/name).read_bytes()) for name in (
        'reports/toy100/pr84_critic_bank_swap.py',
        'reports/toy100/pr84_critic_relaxation.py',
        'reports/toy100/alternating_curvature_scratch.py',
        'particlegan/gan_loss.py','particlegan/grad_regularizers.py',
        'configs/toy100/constraints_simple_regularization.json',
        'benchmarks/locked_shared/mode_hold.py')}
    declaration=dict(scope='read_only_one_saved_state_three_bank_diagnostic',
                     training_candidate=False,seed_changes=False,
                     step=args.step,start_phase=args.start_phase,
                     width=args.width,fit_budget=dict(iterations=fit.MAX_ITER,
                     closures=fit.MAX_CLOSURES),bank_batches=8,bank_batch_size=128,
                     bank_c='evaluation only; common G batches are C D-style draws',
                     states_sha256=sha(args.states.read_bytes()),source=source_hashes,
                     python=platform.python_version(),torch=torch.__version__,
                     cpu_threads=torch.get_num_threads(),
                     cpu_capability=torch.backends.cpu.get_cpu_capability())
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    states=torch.load(args.states,weights_only=True)
    initial_hash=fit.state_hash(states)
    captured=states[args.step]
    pre=captured['pre_step']; saved=captured[args.start_phase]
    if fit.state_hash(dict(generator=pre['generator'],prior=pre['prior'])) != fit.state_hash(
            dict(generator=saved['generator'],prior=saved['prior'])):
        raise RuntimeError('G/prior changed before D fit start point')
    a_rows,b_rows,c_rows,a_bank,b_bank,c_bank=draw_banks(pre,saved)
    generator,d_start,prior=fit.modules(saved)
    a_original,b_original,a_join,b_join,g_batch=fit.banks(pre,saved,generator,prior)
    if any(not torch.equal(row[key],original[key]) for rows,originals in
           ((a_rows,a_original),(b_rows,b_original)) for row,original in zip(rows,originals)
           for key in ('real','fake','indices','noise')):
        raise RuntimeError('continued banks do not recreate audited A/B draws')
    if any(not torch.equal(bank[key],reference[key]) for bank,reference in
           ((a_bank,a_join),(b_bank,b_join)) for key in ('real','fake')):
        raise RuntimeError('joined A/B banks differ from audited replay')
    recipe,_,_=declared_recipe(json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text()))
    gan=recipe.make_loss();regularizer=recipe.make_gradient_penalty()
    d_pre=fit.modules(pre)[1]
    first_d=fit.gradients(fit.d_loss(d_pre,a_rows[0],gan,regularizer,args.step)[0],d_pre)
    expected=[saved['optimizer_d']['state'][index]['exp_avg'] for group in
              saved['optimizer_d']['param_groups'] for index in group['params']]
    if not all(torch.equal(x,y) for x,y in zip(first_d,expected)):
        raise RuntimeError('first D gradient is not bitwise identical to host Adam moment')
    original_g_parity=None
    if args.expect_original_g_parity:
        if 'post_bounded_g' not in captured or 'post_unbounded_g' not in captured:
            raise RuntimeError('requested original G parity without both captured G phases')
        old_width=fit.WIDTH
        with torch.random.fork_rng(devices=[]):
            try:
                fit.WIDTH=args.width
                _,bounded,unbounded=fit.g_proposal(saved,d_start,g_batch,gan,args.step)
            finally:
                fit.WIDTH=old_width
        reference=captured['post_bounded_g']
        ref_bounded=dict(generator=fit.unwrapped(reference['generator']),
                         prior=reference['prior'],optimizer=reference['optimizer_g'])
        reference=captured['post_unbounded_g']
        ref_unbounded=dict(generator=fit.unwrapped(reference['generator']),
                           prior=reference['prior'],optimizer=reference['optimizer_g'])
        original_g_parity=(fit.state_hash(bounded)==fit.state_hash(ref_bounded)
                           and fit.state_hash(unbounded)==fit.state_hash(ref_unbounded))
        if not original_g_parity:
            raise RuntimeError('original unbounded or bounded G proposal parity failed')
    metric=fit.saved_metric(saved['optimizer_d'])
    fit_source_sha=source_hashes['reports/toy100/pr84_critic_relaxation.py']
    if args.archived_a is not None:
        a_row,a_tensors,a_archive=get_archived_a(args.archived_a,args.step,fit_source_sha)
        if (a_row['bank_sha256']!=fit.state_hash(dict(train=a_bank,heldout=b_bank,g_batch=g_batch))
                or any(not torch.equal(a_tensors[key][part],bank[part])
                       for key,bank in (('train',a_bank),('heldout',b_bank))
                       for part in ('real','fake'))):
            raise RuntimeError('archived A fit bank differs from reconstructed A/B')
        d_a=fit.modules(saved)[1];d_a.load_state_dict(a_tensors['critic'])
        fit_a=dict(source='archived',archive=a_archive,
                   closures=a_row['fit']['closure_calls'],
                   train_gradient_linf=a_row['after']['train']['gradient']['raw_linf'],
                   selection=a_row['fit']['selection'])
    else:
        d_a=fit.modules(saved)[1]
        a_fit=fit.relax(d_a,a_bank,gan,regularizer,args.step,metric)
        fit_a=dict(source='new',closures=a_fit['closure_calls'],
                   selection=a_fit['selection'])
    d_b=fit.modules(saved)[1]
    b_fit=fit.relax(d_b,b_bank,gan,regularizer,args.step,metric)
    fit_b=dict(closures=b_fit['closure_calls'],iterations=b_fit['iterations'],
               budget_exhausted=b_fit['closure_budget_exhausted'],
               selection=b_fit['selection'],
               best_training_loss=min(row['total_loss'] for row in b_fit['records']),
               seconds=b_fit['seconds'])
    d_variants=dict(saved=d_start,a=d_a,b=d_b)
    d_receipts={};d_logits={}
    for name,d in d_variants.items():
        evaluation=fit.d_evaluation(d,c_bank,c_rows,gan,regularizer,args.step,metric)
        losses={bank_name:float(fit.d_loss(d,bank,gan,regularizer,args.step)[0].detach())
                for bank_name,bank in (('a',a_bank),('b',b_bank),('c',c_bank))}
        per_batch=[float(fit.d_loss(d,row,gan,regularizer,args.step)[0].detach())
                   for row in c_rows]
        with torch.no_grad():
            logits=torch.cat([d(c_bank['real']).flatten(),d(c_bank['fake']).flatten()]).double()
        d_logits[name]=logits-logits.mean()
        d_receipts[name]=dict(losses=losses,c=evaluation,
                              c_batch_losses=per_batch,
                              c_stationary=evaluation['gradient']['raw_linf']<=1e-7)
    g_receipts={};g_flat={};g_functional={}
    old_width=fit.WIDTH
    outer_rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        try:
            fit.WIDTH=args.width
            for name,d in d_variants.items():
                values=[];grads=[];moves=[]
                for row in c_rows:
                    gradient,proposal=per_batch_field(saved,d,row,gan,args.step,
                                                       saved['noise']['output_sigma'])
                    grads.append(gradient)
                    moves.append(torch.tensor(proposal['accepted_functional'],dtype=torch.float64).flatten())
                    values.append({key:proposal[key] for key in
                                   ('loss','rho','factor','grade_before','grade_after')})
                g_flat[name]=grads;g_functional[name]=moves
                g_receipts[name]=values
            finite_check=finite_partial_check(saved,d_a,c_rows[0],gan,args.step,
                                               saved['noise']['output_sigma'],g_flat['a'][0])
        finally:
            fit.WIDTH=old_width
    if not torch.equal(torch.get_rng_state(),outer_rng):
        raise RuntimeError('G heldout fields changed the global RNG')
    comparisons={}
    for left,right in (('saved','a'),('saved','b'),('a','b')):
        comparisons[f'{left}_vs_{right}']=dict(
            partial_gradient_cosines=[cosine(x,y) for x,y in zip(g_flat[left],g_flat[right])],
            adam_clean_output_cosines=[cosine(x,y) for x,y in zip(
                g_functional[left],g_functional[right])],
            c_paired_d_loss_differences=[x-y for x,y in zip(
                d_receipts[left]['c_batch_losses'],d_receipts[right]['c_batch_losses'])],
            c_centered_critic_logit_rms_difference=float(
                (d_logits[left]-d_logits[right]).square().mean().sqrt()))
    result=dict(declaration=declaration,first_d_gradient_bitwise=True,
                original_g_proposals_exact=original_g_parity,
                a_b_replay_exact=True,fit_a=fit_a,fit_b=fit_b,
                d=d_receipts,g=g_receipts,comparisons=comparisons,
                finite_partial_gradient_check=finite_check,
                saved_states_unchanged=fit.state_hash(states)==initial_hash,
                global_rng_unchanged=torch.equal(torch.get_rng_state(),outer_rng))
    if not result['saved_states_unchanged'] or not result['global_rng_unchanged']:
        raise RuntimeError('diagnostic mutated saved state or RNG')
    (args.output/'summary.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(event='BANK_SWAP_DONE',step=args.step,
                          c_losses={k:v['losses']['c'] for k,v in d_receipts.items()},
                          c_residuals={k:v['c']['gradient']['raw_linf'] for k,v in d_receipts.items()},
                          a_b_partial_cosines=comparisons['a_vs_b']['partial_gradient_cosines'],
                          a_b_output_cosines=comparisons['a_vs_b']['adam_clean_output_cosines'])))


def main():
    # Model constructors draw initial weights before saved weights are loaded.
    # Isolate even those irrelevant draws from a caller's global RNG.
    entry_rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        _main()
    if not torch.equal(torch.get_rng_state(),entry_rng):
        raise RuntimeError('complete diagnostic changed the caller RNG')


if __name__=='__main__':
    main()
