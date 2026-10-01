"""One frozen saved-native mean decomposition; no fit, draws or scoring."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
RUN=ROOT/'validation-cb64-ra9/screens/runs/grid100'
ORIGINAL=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra7-quality/grid-saved-diagnostic/diagnose.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard():
    prep=json.loads((HERE/'PREPARATION-FROZEN.json').read_text())
    for name,h in prep['source_and_input_sha256'].items(): assert sha(name)==h,name
    assert sha(ORIGINAL)=='ab083335ea1df0a2eddd9fe574e657484da8795dfb74dc53377a3584e1ab0a05'
    return prep


def main():
    assert not (HERE/'receipt.json').exists()
    prep=guard()
    spec=importlib.util.spec_from_file_location('frozen_saved_centroid_equations',ORIGINAL)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    np,torch=old.np,old.torch
    initial_torch=torch.get_rng_state().clone()
    initial_numpy=np.random.get_state()
    assert not torch.cuda.is_initialized()
    state=torch.load(RUN/'final-state.pt',map_location='cpu',weights_only=False)['trainer']
    assert state['completed_steps']==7000 and state['recipe']['num_particles']==20000
    assert state['recipe']['z_dim']==2 and state['birth_death']['backend_schema']==8
    sigma=old.SIGMA;centers=old.CENTERS
    panels={}
    for split,name in (('terminal','final_samples.npz'),('holdout','holdout_samples.npz')):
        for kind in ('clean','noisy'):
            with np.load(RUN/f'native-{kind}'/name,allow_pickle=False) as data:
                assert set(data.files)=={'live','ema','target'}
                assert all(data[key].shape==((20000 if split=='terminal' else 100000),2) for key in data.files)
                panels[(split,kind)]={key:data[key].astype(np.float64) for key in data.files}
        assert np.array_equal(panels[(split,'clean')]['target'],panels[(split,'noisy')]['target'])
    models=state['models'];z={};anchor={};maps={}
    for role,g,p in (('FAST','G','prior'),('EMA','ema_G','ema_prior')):
        a=models[g]['weight'].numpy().astype(np.float64);b=models[g]['bias'].numpy().astype(np.float64)
        z[role]=models[p]['z'].numpy().astype(np.float64)
        assert a.shape==(2,2) and b.shape==(2,) and z[role].shape==(20000,2)
        anchor[role]=z[role]@a.T+b
        maps[role]=(a,b)

    def annotated_means(points):
        ids,residual=old.assign(points)
        selected=np.linalg.norm(residual,axis=1)<=3.
        n,mean,_=old.moments(residual,ids)
        nh,mh,_=old.moments(residual[selected],ids[selected])
        assert np.all(n>1) and np.all(nh>1)
        return dict(ids=ids,selected=selected,counts=n,mean=mean,hq_counts=nh,hq_mean=mh)

    references={'oracle_component_centers':dict(mean=np.zeros_like(centers),hq_mean=np.zeros_like(centers))}
    fifo=state['birth_death']['reservoir'].numpy().astype(np.float64)
    references['saved_training_real_FIFO']=annotated_means(fifo)
    for split in ('terminal','holdout'):
        references[f'saved_{split}_real_target']=annotated_means(panels[(split,'clean')]['target'])
    diagnostics={};raw_prior={};anchor_info={}
    for role in ('FAST','EMA'):
        a,b=maps[role];info=annotated_means(anchor[role]);anchor_info[role]=info
        prior=annotated_means(z[role])
        raw_prior[role]=dict(own_prior_annotation_counts=prior['counts'].tolist(),
            prior_conditional_mean_sigma=prior['mean'].tolist(),
            prior_center_rms_sigma=float(np.sqrt(old.mean_square(prior['mean']))),
            prior_versus_actual_output_annotation_agreement=float(np.mean(prior['ids']==info['ids'])))
        role_results={}
        for selection in ('all_rows','anchor_inside_3sigma'):
            keep=np.ones(len(z[role]),dtype=bool) if selection=='all_rows' else info['selected']
            ids=info['ids'][keep]
            counts,zm,_=old.moments(z[role][keep],ids)
            _,xm,_=old.moments(anchor[role][keep],ids)
            assert np.all(counts>1)
            group_results={}
            for reference,ref in references.items():
                target_mean=centers+sigma*ref['mean' if selection=='all_rows' else 'hq_mean']
                transformed_prior=(zm-target_mean)@a.T/sigma
                affine=(target_mean@a.T+b-target_mean)/sigma
                residual=(xm-target_mean)/sigma
                error=float(np.max(np.abs(residual-transformed_prior-affine)))
                assert error<1e-10
                ideal=(target_mean-b)@np.linalg.inv(a).T
                inverse_residual=(zm-ideal)@a.T/sigma
                assert np.max(np.abs(inverse_residual-residual))<1e-10
                table_ms=old.mean_square(transformed_prior);affine_ms=old.mean_square(affine)
                cross=float(2*np.mean(np.sum(transformed_prior*affine,axis=1)))
                total=old.mean_square(residual)
                assert abs(table_ms+affine_ms+cross-total)<1e-9
                group_results[reference]=dict(counts=counts.tolist(),
                    transformed_table_component_sigma=old.norm_stats(transformed_prior),
                    affine_component_sigma=old.norm_stats(affine),
                    total_anchor_minus_reference_centroid_sigma=old.norm_stats(residual),
                    center_mean_square_sigma2=dict(table=table_ms,affine=affine_ms,twice_cross=cross,total=total),
                    table_and_affine_centroid_vectors=old.centroid_comparison(transformed_prior,affine),
                    transformed_table_vectors_sigma=transformed_prior.tolist(),
                    affine_vectors_sigma=affine.tolist(),residual_vectors_sigma=residual.tolist(),
                    inverse_map_latent_mean_interpretation_only=ideal.tolist(),
                    exact_mean_identity_max_residual=error)
            role_results[selection]=group_results
        diagnostics[role]=dict(affine_weight=a.tolist(),affine_bias=b.tolist(),
            affine_matrix_max_abs_difference_from_initialized_identity=float(np.max(np.abs(a-np.eye(2)))),
            affine_condition_number=float(np.linalg.cond(a)),decompositions=role_results)
    cloud_first_moments={}
    for (split,kind),panel in panels.items():
        cloud_first_moments[f'{split}/{kind}']={}
        for role in ('live','ema'):
            info=annotated_means(panel[role])
            cloud_first_moments[f'{split}/{kind}'][role]=dict(counts=info['counts'].tolist(),
                hq_counts=info['hq_counts'].tolist(),conditional_mean_sigma=info['hq_mean'].tolist(),
                unconditional_mean_sigma=info['mean'].tolist(),
                versus_saved_EMA_anchor_unconditional=old.centroid_comparison(info['mean'],anchor_info['EMA']['mean']),
                versus_saved_EMA_anchor_existing_selection=old.centroid_comparison(info['hq_mean'],anchor_info['EMA']['hq_mean']),
                clean_and_noisy_live_equals_saved_ema_array=np.array_equal(panel['live'],panel['ema']))
    af,bf=maps['FAST'];ae,be=maps['EMA']
    table_delta=(z['EMA']-z['FAST'])@af.T
    map_delta=z['EMA']@(ae-af).T+(be-bf)
    total_delta=anchor['EMA']-anchor['FAST']
    assert np.max(np.abs(table_delta+map_delta-total_delta))<1e-10
    assert torch.equal(initial_torch,torch.get_rng_state()) and not torch.cuda.is_initialized()
    current_numpy=np.random.get_state()
    assert initial_numpy[0]==current_numpy[0] and np.array_equal(initial_numpy[1],current_numpy[1]) and initial_numpy[2:]==current_numpy[2:]
    guard()
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),fixed_state_step=7000,
        reused_original_helper_sha256=sha(ORIGINAL),source_and_input_sha256=prep['source_and_input_sha256'],
        sigma=sigma,raw_prior_annotation=raw_prior,affine_and_table_mean_decomposition=diagnostics,
        saved_cloud_first_moment_comparisons=cloud_first_moments,
        paired_FAST_EMA_row_delta=dict(table_sigma=old.norm_stats(table_delta/sigma),
            affine_sigma=old.norm_stats(map_delta/sigma),total_sigma=old.norm_stats(total_delta/sigma),
            max_identity_residual=float(np.max(np.abs(table_delta+map_delta-total_delta))),
            same_row_output_oracle_annotation_fraction=float(np.mean(anchor_info['FAST']['ids']==anchor_info['EMA']['ids']))),
        typed_serving_stamp=dict(state['birth_death']['paired_average']),G_requires_grad=state['requires_grad']['G'],
        original_quality_verdict='FAIL',original_gate_not_recomputed=True,CPU_RNG_exact=True,numpy_RNG_exact=True,
        cpu_only=True,cuda_initialized=False,new_draws=0,new_emissions=0,new_training=0,new_chart_fits=0,
        quality_rescoring_calls=0,production_actions_or_changes=0,
        limits=['Native initialized identity is a diagnostic reference only; final saved FAST/EMA A,b are always used.',
            'Oracle components annotate saved points only; no production targets, learned components or corrections are fitted.',
            'Fixed output-group prior means can compensate affine drift. The algebra is not a causal historical ablation.',
            'All-row and3sigma anchor selections use fixed annotations; they do not change original output scoring.',
            'The inverse latent mean is an interpretation of the affine identity, not an emitted new latent proposal.',
            'Saved FIFO/target-cloud means have finite-sample and training-adaptation uncertainty, without new confidence claims.',
            'Affine outputs use float64 algebra on saved float32 coefficients, not a float32 historical forward replay.',
            'Served live arrays may use EMA while checkpoint state stores raw FAST and EMA models separately.'])
    (HERE/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    compact={role:diagnostics[role]['decompositions']['all_rows']['oracle_component_centers']['center_mean_square_sigma2'] for role in diagnostics}
    print(json.dumps(dict(status='PASS',affine_and_table_mean_square_sigma2=compact,
        FAST_weight=diagnostics['FAST']['affine_weight'],EMA_weight=diagnostics['EMA']['affine_weight'])),flush=True)


if __name__=='__main__': main()
