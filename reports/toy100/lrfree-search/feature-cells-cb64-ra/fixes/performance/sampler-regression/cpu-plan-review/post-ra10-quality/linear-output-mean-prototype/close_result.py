"""Post-exit guard and seal for the one fixed mechanical scratch result."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--closed-log',type=Path,required=True);args=p.parse_args()
    assert args.output.resolve().is_relative_to(HERE)
    source=json.loads((HERE/'SOURCE-FROZEN.json').read_text());result=json.loads(args.output.read_text())
    assert result['status']=='PASS' and result['device']=='cpu' and not result['cuda_initialized']
    assert [r['case'] for r in result['records']]==['grid','toy']
    assert all(r['status']=='PASS' and all(r['checks'].values()) for r in result['records'])
    files=dict(source['source_and_input_sha256'])
    for q in [HERE/'SOURCE-FROZEN.json',HERE/'ROOT-GO.json',args.closed_log]:files[str(q.resolve())]=sha(q)
    approval=json.loads((HERE/'ROOT-GO.json').read_text())
    for q,h in approval['independent_review_sha256'].items():files[q]=h
    for q in args.output.parent.rglob('*'):
        if q.is_file():files[str(q.resolve())]=sha(q)
    for q,h in files.items():assert sha(q)==h,q
    report=args.output.parent/'REPORT.md';assert not report.exists()
    text='# Fixed linear-output mean scratch result\n\nMechanical status PASS, no original quality verdict or production selection.\n\n'
    for r in result['records']:
        w=r['witness'];text+=f"- {r['case']}: witness {w['status']}, lower bound {w['lower_bound']}, moment/chart rank {w['moment_rank']}/{w['chart_rank']}, attempts {w['attempts']}, accepted mean copies {r['mean']}.\n"
    text+='\nDirect raw conditional means and centered covariance before/after are in each raw-moments.json. Clipping, learned groups and nonlinear latent-to-output jitter remain; clean objective does not certify emitted covariance or quality. No altered backend9 checkpoint is resumable.\n'
    report.write_text(text);files[str(report.resolve())]=sha(report)
    receipt=dict(status='PASS',scope='ONE_FIXED_CPU_LINEAR_OUTPUT_MEAN_SCRATCH_MECHANICS',utc=datetime.now(timezone.utc).isoformat(),
        files=files,result_sha256=sha(args.output),source_freeze_sha256=sha(HERE/'SOURCE-FROZEN.json'),
        production_package_unchanged=True,no_resumable_newlaw_checkpoint=True,new_quality_emissions=0,new_scoring_calls=0,
        model_training_updates=0,quality_verdict=None,root_selects_future_law=True)
    out=args.output.parent/'FINAL-FROZEN.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n')
    for q,h in files.items():assert sha(q)==h,q
    print(json.dumps(dict(status='PASS',result_sha256=sha(args.output),final_freeze_sha256=sha(out),guards=len(files))))
if __name__=='__main__':main()
