"""Seal the single fixed scratch protocol/helpers/raw inputs before imports or PT."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def main():
    out=HERE/'SOURCE-FROZEN.json';assert not out.exists()
    files={}
    def bind(p,h=None):
        p=Path(p).resolve();actual=sha(p)
        assert h is None or h==actual,p
        assert str(p) not in files or files[str(p)]==actual,p
        files[str(p)]=actual
    ready=ROOT/'quality/ra10/READY.json'
    bind(ready,'9f29f98761b136055ef5bfce3697de0cd45cb8d12c2d65fee01e5f5281edf6fd')
    for p,h in read(ready)['numerical_source_sha256'].items():bind(p,h)
    composition=ROOT/'quality/ra10/COMPOSITION.json'
    bind(composition,'807ed0317ea4f1771c151be5f01c39f80f5f430f4cf4ad9d08aa6910a8656ad2')
    modules=read(composition)['source_sha256']
    assert len(modules)==30 and set(modules)=={p.name for p in (ROOT/'pkg-CB64-RA10/particlegan').glob('*.py')}
    for name,h in modules.items():bind(ROOT/'pkg-CB64-RA10/particlegan'/name,h)
    owner=ROOT/'integration/review/training-regression/post-ra9-quality/mean-category-production'
    source=owner/'SOURCE-FROZEN-v2.json'
    bind(source,'937eba63716e4bb27500b7d4a8fc169d1cf7449d307526244f6d86fc84e96b65')
    for p,h in read(source)['source_and_input_sha256'].items():bind(p,h)
    bind(owner/'run_mechanics_v2.py','6f9fc07a8ceafbc9f60e086de32ea499e131e4e8d2bbc39da92e569c69e538b6')
    # Reuse closed ownership/dirty-eval evidence; no extra control suite.
    controls=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-controls'
    bind(controls/'runtime-attempt1/receipt.json','15c6f8fd89a1f871cbd59bf3e533e136ca47700354f0514530cd906d1160213a')
    bind(controls/'runtime-attempt1/FROZEN.json','7fcab0e0afde30aee845193575416eae8a7cadf9fa169ac93f2e8d357cdbcfba')
    for p,h in read(controls/'runtime-attempt1/FROZEN.json')['files'].items():bind(p,h)
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in ('.py','.md'):
            bind(p)
            if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
    for p in [ROOT/'configs/overrides-CB64-RA9.json',ROOT/'configs/overrides-CB64-RA10.json',
              ROOT/'validation-cb64-ra9/screens/runs/grid100/final-state.pt',
              ROOT/'validation-cb64-ra9/learned/training/toy/CB64-RA9/checkpoint-2000.pt']:
        bind(p)
    assert sha(ROOT/'configs/overrides-CB64-RA10.json')==sha(ROOT/'configs/overrides-CB64-RA9.json')=='b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
    value=dict(status='FROZEN_FIXED_CPU_LINEAR_OUTPUT_MEAN_PROTOTYPE',utc=datetime.now(timezone.utc).isoformat(),
        source_and_input_sha256=files,package_sha256=read(ready)['package_sha256'],config_sha256=read(ready)['config_sha256'],
        external_module_sha256=sha(HERE/'linear_mean.py'),helper_sha256=sha(HERE/'run_prototype.py'),
        moment_policy='even_linear_output_group_clip_unit_residual_EB_3K_plus_3_scratch_v1',
        cases=['fixed_RAW_RA9_grid7000','fixed_RAW_RA9_toy2000'],invocations_allowed=1,per_case_reactions_allowed=1,
        device='cpu',threads=1,fixture_seed=314159,fixture_seed_changed=False,production_files_changed=False,
        PT_interpretations=0,Torch_imported=False,model_forwards=0,new_quality_emissions=0,quality_verdict=None,
        before_numeric_required=['independent_count_math_source_PASS','independent_state_API_source_PASS','root_GO_this_exact_freeze'],
        numerical_command=f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -B {HERE/'run_prototype.py'} --device cpu --output {HERE/'attempt1/result.json'}")
    out.write_text(json.dumps(value,indent=2)+'\n')
    for p,h in files.items():assert sha(p)==h,p
    print(json.dumps(dict(status=value['status'],guards=len(files),source_freeze_sha256=sha(out),module_sha256=value['external_module_sha256'],helper_sha256=value['helper_sha256'])))
if __name__=='__main__':main()
