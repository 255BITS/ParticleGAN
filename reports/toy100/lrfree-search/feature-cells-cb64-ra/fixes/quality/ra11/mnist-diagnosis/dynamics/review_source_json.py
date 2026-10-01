"""Stdlib-only source and closed original JSON dynamics review; no PT reads."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
RUNS = {
    'RA11': ROOT / 'validation-cb64-ra11/learned/training/mnist/CB64-RA11',
    'RA4': ROOT / 'validation-ra4/learned/training/mnist/CB64-RA4',
    'E22': ROOT.parent / 'feature-cells-cuda-retest-20260929/learned/training/mnist/E22',
}
STEPS = [0,100,250,500,750,1000,1250,1500,1750,2000]
EVALUATOR_FIELDS = ('accuracy','active_dimensions','active_mask_sha256','active_mean_sha256',
    'active_std_sha256','raw_reference_sha256','active_reference_sha256','evaluator_model_sha256')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def method(path, owner, name):
    tree = ast.parse(Path(path).read_text())
    body = tree.body if owner is None else next(n.body for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name)
    if isinstance(node,ast.FunctionDef) and node.body and isinstance(node.body[0],ast.Expr) \
            and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
        node.body = node.body[1:]
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def main():
    seal = read(HERE/'SOURCE-FROZEN.json')
    assert seal['status'] == 'FROZEN_STDLIB_SOURCE_AND_CLOSED_JSON_REVIEW'
    for p,h in seal['files'].items(): assert sha(p)==h,p
    configs = {name:read(path/'config.json') for name,path in RUNS.items()}
    packages = {name:Path(value['package']['package_root'])/'particlegan'
                for name,value in configs.items()}
    evaluators = {name:{key:read(path/'evaluator.json').get(key) for key in EVALUATOR_FIELDS}
                  for name,path in RUNS.items()}
    assert evaluators['RA11']==evaluators['RA4']==evaluators['E22']
    initial = {name:{key:value[key] for key in ('initial_generator_sha256','initial_critic_sha256',
        'initial_prior_sha256')} for name,value in configs.items()}
    assert initial['RA11']==initial['RA4']==initial['E22']
    curves={};results={};progress={}
    for name,path in RUNS.items():
        result=read(path/'result.json');assert result['status']=='COMPLETE' and result['steps']==2000
        results[name]=result['final']
        rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines() if line]
        assert [row['step'] for row in rows]==STEPS
        out=[]
        for row in rows:
            d=row['diagnostics'];bd=d['birth_death'];s=d['lr_settle'];c=d['controller'];m=row['metrics']
            out.append(dict(step=row['step'],active_embedding=m['active_embedding'],
                confident_class_coverage=m['confident_class_coverage'],class_mass_tv=m['class_mass_tv'],
                pixel_clipping_fraction=m['pixel_clipping_fraction'],output_sigma=d['output_sigma'],
                actual_lrs=d['lr'],controller={k:c.get(k) for k in ('payoff_error','game_trust','game_ratio',
                    'mobility','last_cosine','alignment','closed')},
                testers={k:dict(s=v['s'],b=v['b'],last_decision=v.get('last',{}).get('decision'),
                    population_active=v.get('population_active'),rebases=v['counts']['rebases']) for k,v in s.items()},
                accepted_moves=bd['counters']['moves'],isolation_moves=bd['counters']['iso_moves'],
                mean_moves=bd['counters'].get('mean_moves'),mean_fires=bd['counters'].get('mean_witness_fires'),
                paired_average=bd.get('paired_average')))
        curves[name]=out
        logpath=Path(seal['logs'][name]);items=[]
        for line in logpath.read_text().splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if item.get('event')=='training_progress':
                items.append({k:item.get(k) for k in ('step','loss_g','loss_d','penalty')})
        progress[name]=items
    assert all(row['accepted_moves']==row['isolation_moves']==row['mean_moves']==row['mean_fires']==0
               for row in curves['RA11'])
    assert all(not row['paired_average']['eligible'] for row in curves['RA11'])
    specs=[('training.py','GANTrainer',name) for name in ('_output_sigma','_average_rate','_serve_apply',
        '_serve_release','_served_parameters','_served_averages','_table_tester','_stray_gate','_settle_observe')]
    specs += [('continuous.py','DataDriftController',name) for name in ('observe_generator','critic_scale',
        'observe_blind','_advance_mobility','observe_game')]
    specs += [('recipes.py','Recipe',name) for name in ('make_optimizers','make_generator_optimizer','make_critic_optimizer')]
    ast_comparisons=[]
    for file,owner,name in specs:
        hashes={variant:method(package/file,owner,name) for variant,package in packages.items()}
        ast_comparisons.append(dict(file=file,owner=owner,name=name,body_ast_sha256=hashes,
            all_equal=len(set(hashes.values()))==1))
    sampling=[]
    for file,owner,name in [('training.py','GANTrainer','_generate'),('training.py','GANTrainer','sample'),
            ('feature_cells.py','FeatureCellBirthDeath','perturb_latent'),
            ('feature_cells.py',None,'BoundedLatentGeometry')]:
        hashes={variant:method(packages[variant]/file,owner,name) for variant in ('RA4','RA11')}
        sampling.append(dict(file=file,owner=owner,name=name,body_ast_sha256=hashes,
            all_equal=len(set(hashes.values()))==1))
    recipe_deltas={}
    for old in ('RA4','E22'):
        a,b=configs[old]['recipe'],configs['RA11']['recipe']
        recipe_deltas[old]={key:dict(old=a.get(key),RA11=b.get(key)) for key in sorted(set(a)|set(b))
                            if a.get(key)!=b.get(key)}
    report=dict(status='PASS',scope='STDLIB_SOURCE_AND_CLOSED_JSON_DYNAMICS_ONLY',utc=datetime.now(timezone.utc).isoformat(),
        source_preseal_sha256=sha(HERE/'SOURCE-FROZEN.json'),source_and_input_sha256=seal['files'],
        initial_model_and_prior_hashes=initial,active_evaluator_frames=evaluators,
        exact_shared_evaluator_frame=True,recipe_deltas=recipe_deltas,curves=curves,progress=progress,
        unchanged_body_ASTs_ignoring_docstrings=ast_comparisons,RA4_sampling_ASTs=sampling,
        optimizer_kernel_bytes={file:{v:sha(p/file) for v,p in packages.items()} for file in ('k3p.py','ka2.py')},
        findings=[
            'No accepted mean, ordinary, isolation or novel population action occurred in RA11 MNIST; there was no population rebase/reset and lineage stayed empty.',
            'Paired EMA eligibility is false at every recorded checkpoint. Active evaluation therefore used FAST, not an averaged serving swap.',
            'Applied sigma grew above the unchanged learnable floor while the G and sigma testers remained scale1; G+sigma base .0010625 is one quarter the previous .00425, prior base .0085 and critic base .00425 are preserved.',
            'The inherited payoff-error controller multiplies critic LR by 1/(1+payoff_error**2). The saved payoff error remains large and D LR remains about2e-5; RA4/E22 recover D LR and sigma falls to .029.',
            'The output moment frame is fitted deterministically without an RNG draw. Its extra EMA observations use the owned-state-neutral context; a nonfiring run returns before preview and paired jitter.',
            'The actual image G and D fixture definitions contain no stochastic eval layers or registered buffers. Their model/gradient/global and dedicated stream observation boundary is preserved by the qualified source.'
        ],
        candidate_direction=dict(kind='ONE_RESERVED_CONFIG_ONLY_PROSPECT',lr=.00425,prior_lr_mult=2.,d_lr_mult=1.,
            scope='restore historical G+sigma base; preserve prior/D base and all algorithm/noise/serving laws',
            evidence='RA4 and corrected E22 original MNIST recover under these bases; actual RA11 G+sigma rates never annealed to compensate the quarter base',
            qualified=False),
        limits=[
            'Zero accepted actions rules out a direct mean-copy effect; observational neutrality is a source/qualified-contract conclusion, not a new live measurement.',
            'The three runs differ in support/controller/serving package laws as well as rates. Stored trajectories establish a failure path, not an isolated causal config experiment.',
            'Clean generator saturation versus late output-noise amplification is not quantified from tensors here. Pixel clipping alone also counts correct near-boundary MNIST background pixels.',
            'A config-only correction has evidence for a prospective test, not a guarantee of retaining toy/grid passes or recovering MNIST.',
            'Registered model state and owned Torch streams are covered by observation guards; arbitrary custom external/Python state is outside that claim.'
        ],Torch_imported=False,PT_objects_loaded=0,model_constructions=0,model_forwards=0,new_draws=0,
        training_updates=0,new_scoring_calls=0,new_seeds=0,CPU_only=True)
    for p,h in seal['files'].items():assert sha(p)==h,p
    with (HERE/'receipt.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',raw_guards=len(seal['files']),receipt_sha256=sha(HERE/'receipt.json'),
        common_ASTs_equal=[r['name'] for r in ast_comparisons if r['all_equal']],
        common_AST_differences=[r['name'] for r in ast_comparisons if not r['all_equal']],
        RA4_sampling_equal=all(r['all_equal'] for r in sampling),PT_loaded=0)))


if __name__=='__main__':main()
