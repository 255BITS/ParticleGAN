"""Build a compact status report from completed, frozen CUDA evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path

from quality.leaderboard import toy_gate

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'feature-cells-cuda-retest-20260929'
VARIANTS = {'E22':BASE, 'CB64-RA':BASE, 'CB64-RA2':ROOT/'validation',
            'CB64-RA3':ROOT/'validation-ra3', 'CB64-RA4':ROOT/'validation-ra4'}
MONITORS = {'CB64-RA2':ROOT/'integration/review/validation-monitor/summary.json',
            'CB64-RA4':ROOT/'integration/review/ra4-indexed-api-monitor/summary.json'}
for lane in sorted(ROOT.glob('validation-cb64-ra*')):
    if lane.is_dir():
        variant=lane.name.removeprefix('validation-').upper()
        VARIANTS[variant]=lane
        MONITORS[variant]=ROOT/'integration/review'/f'{lane.name}-monitor/summary.json'


def screen_counts(records, field):
    native = ('grid100', 'rotated100', 'staggered100')
    return {group:{status:sum(d.get(field)==status and (d['task'] in native)==is_native
                             for d in records) for status in ('PASS','FAIL','ERROR','PENDING')}
            for group,is_native in (('portability',False),('native',True))}


def main():
    rows = []
    for variant, lane in VARIANTS.items():
        row = dict(variant=variant, lane=str(lane))
        for problem in ('toy', 'mnist'):
            result = lane/'learned/training'/problem/variant/'result.json'
            if not result.exists():
                error=result.with_name('error.json')
                row[problem] = (dict(status='ERROR',error=str(error),
                    error_details=json.loads(error.read_text())) if error.exists()
                    else dict(status='PENDING'))
                continue
            data = json.loads(result.read_text())
            metrics = data['final']['metrics']
            row[problem] = dict(status=data['status'], result=str(result),
                training_seconds=data['training_seconds'], metrics=metrics,
                peak_gpu_allocated_bytes=data['peak_gpu_allocated_bytes'])
            if problem == 'toy':
                row[problem]['quality_gate'] = 'PASS' if toy_gate(metrics) else 'FAIL'
        screens = [json.loads(p.read_text()) for p in sorted((lane/'screens/runs').glob('*/result.json'))]
        row['screens'] = dict(primary=screen_counts(screens,'status'), completed=len(screens))
        monitor = MONITORS.get(variant)
        if monitor and monitor.exists():
            audit=json.loads(monitor.read_text())
            row['screens'].update(accepted=screen_counts(audit['records'],'acceptance_status'),
                audited_completed=audit['completed'], audit_status=audit['status'],
                audit=str(monitor), source_integrity=audit['source_integrity'],
                all_completed_fixtures_valid=all(d['canonical_fixture_validity']=='VALID'
                    for d in audit['records'] if d['acceptance_status']!='PENDING'),
                acceptance_scope=('declared RA4 indexed API; separate one-literal metadata adapter'
                    if variant=='CB64-RA4' else 'frozen declared collector API'))
        elif variant=='CB64-RA':
            baseline_audit=BASE/'audit/CHECKS.json'
            audit=json.loads(baseline_audit.read_text())
            row['screens'].update(accepted=row['screens']['primary'], audited_completed=len(screens),
                audit=str(baseline_audit), audit_status=audit['canonical_gpu_acceptance'],
                all_completed_fixtures_valid=audit['artifact_validity']=='VALID',
                acceptance_scope='original frozen collector; independent baseline artifact audit')
        else:
            row['screens'].update(accepted=screen_counts([],'acceptance_status'), audited_completed=0,
                audit_status='PENDING', acceptance_scope='not run' if variant=='CB64-RA3' else 'reference learned comparison only')
        rows.append(row)
    final_screens=next(row['screens'] for row in rows if row['variant']=='CB64-RA4')
    final_learned_path=ROOT/'integration/review/ra4-artifact-audit-state-review/summary.json'
    final_learned=json.loads(final_learned_path.read_text()) if final_learned_path.exists() else {}
    complete=(final_screens['audited_completed']==16 and
        final_screens.get('all_completed_fixtures_valid') is True and final_learned.get('complete') is True
        and all(record.get('evidence_status')=='VALID' for record in final_learned.get('records',{}).values()))
    leaderboard = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        status='COMPLETE' if complete else 'IN_PROGRESS',
        recommendation='No package is recommended for the toy/grid target until both unchanged gates and required validity/replay checks pass. E22 remains the broad reference; RA4 has the best measured MNIST feature distance.',
        completion_scope='Original RA4 learned/state and 16-screen evidence; later toy/grid candidates have separate quality status.',
        ra4_learned_audit=str(final_learned_path),
        scope='One existing fixed seed per fixture; matched saved inputs and initialization; shared GPU0 timings are descriptive.',
        rows=rows)
    (ROOT/'leaderboard.json').write_text(json.dumps(leaderboard,indent=2)+'\n')
    lines=['# CB64-RA failure diagnostics and corrections','',
        f"Updated {leaderboard['updated_utc']}. Status: {leaderboard['status']}.", '',
        leaderboard['recommendation'], '',
        'See [the current toy/grid leaderboard](quality/REPORT.md) for the required joint target. '
        'A failed toy leaves that candidate\'s grid unrun; runtime errors have no quality verdict.', '',
        '## Matched learned CUDA results','',
        'N=1024, z=128, batch=128, 2000 updates, seed=314159; identical saved data/evaluator and initial G/D/prior tensors.', '',
        '| Variant | Toy precision | Modes /25 | Toy mass TV | MNIST active FD ↓ | MNIST precision | MNIST recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        t,m=row['toy'],row['mnist']
        tm=t.get('metrics');mm=m.get('metrics',{}).get('active_embedding')
        toy=(f"{tm['precision']:.3%} | {tm['coverage']} | {tm['mass_tv']:.6f}"
             if tm else f"{t['status']} | — | —")
        mnist=(f"{mm['embedding_frechet']:.6f} | {mm['embedding_precision']:.3%} | {mm['embedding_recall']:.3%}"
               if mm else f"{m['status']} | — | —")
        lines.append(f"| {row['variant']} | {toy} | {mnist} |")
    lines += ['', 'The frozen toy gate requires precision ≥90%, all 25 modes with at least 1% supported mass each, and mass TV ≤0.10. '
        'MNIST has comparative metrics rather than an absolute promotion gate. '
        'These measurements do not establish broad architectural scalability.', '',
        '## Canonical CUDA screens','',
        '| Variant | Portability passes /13 | Native passes /3 | Completed /16 |',
        '|---|---:|---:|---:|']
    for row in rows[1:]:
        screen=row['screens']
        label='Withheld: indexed API mismatch' if row['variant']=='CB64-RA3' else str(screen['audited_completed'])
        lines.append(f"| {row['variant']} | {screen['accepted']['portability']['PASS']} | {screen['accepted']['native']['PASS']} | {label} |")
    lines += ['', 'All native gates require the original 7000-update budget, 34 observations, '
        'five passing terminal 20k evaluations and an independent 100k holdout. '
        'A passing final cloud alone does not pass the full stability gate.', '',
        'RA4 exposes the declared indexed generation API. The copied strict collector hardcodes the '
        'old candidate\'s `evaluation_generate=plain`, so its original INVALID/ERROR metadata receipts '
        'are preserved. A separate hash-bound checker changes only that expected option to `indexed`; '
        'it retains the original source/data/init/stream/schedule/native evidence and quality checks. '
        'RA4 acceptance is reported explicitly under this declared API correction. '
        'Counts in the table are accepted, audited results; raw primary PASS values alone are insufficient. '
        'The original strict monitor and its ERROR receipts remain alongside the separate indexed-API review.', '',
        '## Measured cost corrections','',
        'Paired CUDA contracts preserve outputs, actions and RNG while reducing synchronization. '
        'The exact count kernel measured 29.669 → 1.088 ms with scalar reads 132 → 1. '
        'Warm indexed sampling at N=20,000 measured 13.78 → 9.39 ms for 2048 queries '
        'and 217.99 → 75.37 ms for 20,000 queries; corresponding axis scalar reads fall to zero. '
        'The final planner reduces warm saved-toy scalar reads 2175 → 106 and nominal-fixture reads 851 → 339. '
        'These are focused measurements on shared GPU0; cold timings and small fixtures vary, '
        'and they do not establish faster end-to-end training or an architectural scaling law.', '',
        '## Reproduced causes and fixes','',
        '- Finite count-test resolution prevents actuation below N=800. The corrected package routes these populations to the established E22 backend.',
        '- With-replacement parents amplified a two-row rare group into 29 rows. Real-reference cell/group targets, unique parents and post-action supported ledgers prevent that reproduced mass inflation.',
        '- A fixed latent noise scale collapsed folded support. A bounded local sampler restores that fixed-input support contract; coordinate candidates alone miss some copied neighbors, addressed by a bounded serialized lineage graph.',
        '- Widespread support flags blocked both isolation and ordinary mass recovery. V4 allows count-certified flagged-only ordinary recovery within its existing 5% budget.',
        '- Nearest-cell counts conceal support holes within cells. The prospective count family combines original cell mass, fixed even-fit inside/outside cell regions and aggregate support counts, with one correction across all hypotheses and one shared action ledger.',
        '- CUDA coordinate tensor indices force repeated device-to-host scalar reads. Python axis metadata removes that overhead without changing sampler values or RNG. Exact planner batching preserves quotas, action order, certificates and the shared budget.',
        '- Canonical indexed generation requires a fifth argument named indices. RA3 exposes rows instead, so its native screens are withheld; the subsequent API fix restores indexed lineage sampling.', '',
        'RA5/RA6 add separately checked live/EMA copy geometry, bounded real-anchor latent births and '
        'population stationarity after replacement. RA5 stopped on diagnostic JSON serialization; RA6 '
        'corrected that error and failed the final toy. RA7 lowers G/sigma base rates and adds exact integer '
        'group-count reduction; its final toy also fails, with adaptive scaling compensating for the lower '
        'base rate. RA8 preserves RA7 training and checks current paired averaged geometry before serving '
        'the average. Its final toy passes and its full grid fails. See the target leaderboard for exact gates.', '',
        '## Validation scope and limitations','',
        'RA2 executes the original 16 canonical screen budgets. RA3 is a learned ablation with focused GPU contracts and exact checkpoint replay; its native suite is not run because the frozen API mismatch was identified first. '
        'RA4 receives the same learned and canonical CUDA tests after source/config/commands are frozen.', '',
        'The standalone high-dimensional support-detector proposals remain unqualified. '
        'Bounded parent selection requires an existing eligible parent in an accessible target region; '
        'it cannot reliably recover an absent mode. Training and served averages can have different mode counts. '
        'Finite fixed-partition count-test checks do not provide cumulative error control for repeatedly adapted learned features/FIFO decisions. '
        'GPU0 is shared with other jobs; timing variation is not a scaling law. '
        'Source integrity, restored optimizer/RNG state and checkpoint replay are audited separately from model quality.', '',
        '## Artifacts','',
        '- [RA2 validation](validation/), [RA3 learned ablation](validation-ra3/) and [RA4 validation](validation-ra4/).',
        '- [RA4 source freeze](integration/iteration-4/READY.json), [declared indexed API checker](performance/sampler-regression/cpu-plan-review/indexed-metadata/), and [separate acceptance audit](integration/review/ra4-indexed-api-monitor/).',
        '- [Lineage tests](geometry/training-regression/LINEAGE-REPORT.md), [training-state diagnosis](performance/training-regression/REPORT.md), and [independent reviews](performance/training-regression/count-review/).',
        '- [Kernel/count profiling](performance/PERFORMANCE.md) and [sampler API/cache proof](performance/sampler-regression/cpu-plan-review/AXIS-ID-REPORT.md).',
        '- [Paired CUDA sampler receipts](integration/axis-gpu/result.json), [exact planner receipts](integration/iteration-4/gpu-plan.json), and [missing-mode diagnosis](integration/review/training-regression/post-ra4-mode-diagnosis/REPORT.md).',
        '- Local experiment root and raw-artifact paths are recorded in SOURCE-ARCHIVE.json when archived. Raw checkpoints, datasets, clouds and large traces are retained locally.', '']
    (ROOT/'FIXES-REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=leaderboard['status'],variants=len(rows),report=str(ROOT/'FIXES-REPORT.md'))))


if __name__ == '__main__':
    main()
