"""Generate a compact report from saved CPU diagnostic evidence only."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
HIGH=ROOT.parent/'geometry'/'support-highdim'
OLD=Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929/geometry/results.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    radial=json.loads((ROOT/'diagnosis-all.json').read_text())
    sparse=json.loads((ROOT/'diagnosis-sparse.json').read_text())
    fisher=json.loads((ROOT/'diagnosis-fisher.json').read_text())
    student=json.loads((ROOT/'diagnosis-student.json').read_text())
    niw=json.loads((ROOT/'diagnosis-niw.json').read_text())
    anchor=json.loads((ROOT/'diagnosis-anchor-posterior.json').read_text())
    margins=json.loads((ROOT/'margins.json').read_text())
    high=json.loads((HIGH/'diagnosis-rank.json').read_text())
    old=json.loads(OLD.read_text())
    rows=[]
    for row in anchor['cases']:
        key=tuple(row['fixture'])
        family,n,mechanism,dim,fmap=key
        def locate(result, kind):
            for case in result['cases']:
                k=tuple(case.get('fixture',(case.get('family'),case.get('n'),case.get('mechanism'),case.get('dim'),case.get('feature_map'))))
                if k==key:
                    return kind(case)
            return None
        original=locate(radial,lambda x:x['scores']['spherical'])
        prediction=locate(sparse,lambda x:x['methods']['radial_predictive_variance']['detector'])
        fisher_metric=locate(fisher,lambda x:x['methods']['bounded_dictionary_fisher8']['detector'])
        if family=='cost' and mechanism=='highdim':
            # The independent highdim receipt uses a size-keyed result.
            item=next(x for x in high['cases'] if tuple(x.get('fixture',()))==key)
            methods=item.get('methods',{})
            if methods:
                fisher_metric=next(v['detector'] for k,v in methods.items() if 'bounded' in k or 'dictionary' in k)
            else:
                fisher_metric=item['detector']
        rows.append(dict(fixture=list(key),original=original,prediction_only=prediction,
            fisher=fisher_metric,student=locate(student,lambda x:x['detector']),
            full_covariance_niw=locate(niw,lambda x:x['detector']),actual_anchor_posterior=row['detector']))
    result=dict(status='UNQUALIFIED_RESEARCH',scope='CPU support diagnostics only',
        gpu_jobs_executed=0,shared_source_changes=0,quality_gates_changed=False,
        seeds=dict(geometry=20260929,cost=90229),rows=rows,
        actual_anchor_posterior=dict(passed=anchor['quality_cases_passed'],total=anchor['quality_cases_total']),
        covariance_identity=dict(max_abs_error=max(r['metadata']['scatter_offset_identity_max_error'] for r in anchor['cases']),
            max_error_to_forward_bound=max(r['metadata']['scatter_offset_identity_max_error_to_bound'] for r in anchor['cases'])),
        reference_e22=[dict(case=r['case'],detector=r['exact']['detector']) for r in old['rows']
            if r['backend']=='e22' and (r['family']=='cost' and r['mechanism']=='highdim'
                or r['family']=='geometry' and r['mechanism']=='fold' and r['dim']==128 and r['n']==2048
                or r['family']=='geometry' and r['mechanism']=='fold_nuisance' and r['n'] in (1024,4096))],
        inputs_sha256={str(p):sha(p) for p in (OLD,HIGH/'fisher_rank.py',HIGH/'metric-contract.json')})
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    def cell(metric):
        return '—' if metric is None else f"{metric['recall']:.3f}/{metric['fp']}/{metric['rare_false_positive']}"
    leaderboard=['# Support diagnostic leaderboard','','Each entry is recall / all false positives / rare false positives.',
        'Every entry uses its original held-out calibration, Q=.05 and unchanged family gate.',
        'None of these support laws qualifies across the required family.','','| Fixture | Original | Prediction only | Fisher | Student | Full covariance NIW | Actual anchor posterior |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        family,n,mechanism,dim,fmap=r['fixture']
        label=f"{family}/{mechanism} N{n} " + ('trained' if fmap=='trained600' else 'frozen' if fmap=='frozen_initialization' else '')
        leaderboard.append('| '+label+' | '+' | '.join(cell(r[k]) for k in
            ('original','prediction_only','fisher','student','full_covariance_niw','actual_anchor_posterior'))+' |')
    leaderboard += ['','Cost gates report rare FP but do not impose the geometry zero-rare-FP condition.',
        'The actual-anchor posterior flags two legitimate rare cost rows at N2048/rare_hole, despite passing its cost detector gate.',
        'Support-only passes do not establish parent, mass, replay or learned-model quality.']
    (ROOT/'LEADERBOARD.md').write_text('\n'.join(leaderboard)+'\n')
    research=HIGH/'fisher_rank.py'
    report=f'''# Support diagnosis

Status: **unqualified research**. No support patch was merged, no support READY
was issued, and this lane initialized no CUDA context. The privately staged
`SUPPORT-SCORE.diff` is the rejected prediction-only experiment, not an
integration instruction. Verified performance artifacts were untouched.

## Established causes

1. **Sparse cell uncertainty.** Original trained folded N2048 flags 85 rows for
   82 unsupported rows; false positives are 131, 1166 and rare1745. Row1745 has
   one even real row and zero odd rows in its nearest cell. The spherical scale
   omits uncertainty in the estimated center. The factor `1+1/n` removes that
   rare flag but adds rare1924 under the frozen critic. It is not a universal fix.
2. **Local shape, not just sparse counts.** The Fisher/Student frozen rare1773 has
   n=13 and nine odd partition rows; it is not a singleton. Full covariance
   uncertainty removes that row, but another rare row3262 fails at frozen N4096
   (n=8, 13 odd partition rows). These shifts reject a sparse-only explanation.
3. **Representation and centroid geometry.** All 128 highdim critic features
   vary on real rows; no real-dead feature is being dropped. Unweighted residual
   magnitude is smaller for anomalies than many real rows. A real-only bounded
   dictionary Fisher8 score gives 46/92/184/369 true positives with zero false
   positives at N1024/2048/4096/8192: the anomaly information was present but
   diluted by nuisance variation. Audit's actual Fisher real anchors also
   recover trained N1024 (41/41, zero FP) where centroids do not. Their raw
   unsupported minimum is 508.55 versus null maximum360.72; centroid minimum
   57.20 is below null maximum82.13. Local-radius normalization loses this gap.

## Finite-reference margins

Geometry head inputs are float32; inherited `_features` converts them to
float64. A rule using `finfo(real_features.dtype)` therefore measures converted
arithmetic precision. The first such attempted source-floor test was a no-op.
The corrected source-head precision test is recorded separately and still fails
the family; its result is not used as a silent precision change.

In the full covariance NIW trained N1024 diagnostic, unsupported scores span
16.0529–18.9526 while the two largest odd-real scores are18.4715 and18.4092.
Only six query rows exceed the largest null score. With M512 and Q=.05, at least
40 rows at empirical floor1/513 are needed for a BH rejection; the best ordered
p/threshold ratio is2.9211. The observed overlap prevents power for this score.
Actual-anchor posterior recovers all41 at unchanged gates, so this is not proof
that the critic or finite real data make detection impossible.

Row1745's semantic position is approximately(3.0307,-.1354), with nearest even
and odd real distances .0377 and .0721 (oracle analysis only, after scoring).
It is a supported rare tail row absent from its fitted cell's odd calibration.
In the centroid NIW diagnostic it has p=.002927 and is kept; in the actual-anchor
posterior it has p=.000976, score17.7195 versus null maximum15.8063, and fails
the unchanged zero-rare-FP gate. Frozen rare1773 is kept by the actual-anchor
posterior with p=.038049. A universal support score remains unresolved.

## One combined posterior qualification

The final diagnostic uses existing actual even-real representatives, bounded
Fisher8 directions, the same four-row pooled covariance prior and degrees n+6.
Even-row anchor scatter includes the representative-to-mean offset. Odd rows
alone calibrate the score; original cell assignments, count tests and gates stay
fixed. It passes **14/18 detector cases**. Rejections are trained N2048 rare1745,
frozen N4096 bulk FPR11/3932=.002798, and zero recall for cost nominal/highdim
N1024. It also flags two supported rare rows in N2048/rare_hole. This is not an
end-to-end result and is not recommended for integration.

## Implementation contracts

Two numerical defects were separated from quality failures. Audit's bounded
Fisher helper previously produced non-finite intermediate eigenvalues for zero
within covariance; its observed-real-scale floor and strict finite contracts
are preserved under `geometry/support-highdim/`. The anchor scatter expansion
initially assumed transformed residuals summed exactly to zero. Adding the
residual-sum cross terms and a scale-conditioned forward-error bound makes the
expanded/direct scatter agree: maximum absolute error{result['covariance_identity']['max_abs_error']:.3g},
maximum error/bound{result['covariance_identity']['max_error_to_forward_bound']:.5f}. Direct residual
scatter drives scores. The old 8e-9 assertion failure is retained in its log;
no quality threshold was altered. Helper SHA256: `{sha(research)}`.

## E22 context and recommendation

The archived E22 detector also has zero recall on all four highdim costs and
trained N1024/fold_nuisance. At trained folded N2048 it recalls68/82=.8293 with
four false positives, including two rare rows; frozen N2048 recalls82/82 with
five FP including one rare. Matching these failures is not qualification.

Proceed with the separately verified performance, sampling, mass and small-N
corrections. Preserve this bounded metric and its contracts as research evidence.
Require one unchanged support law to pass all required trained/frozen/cost gates
before merging it. Do not tune priors, seeds, Q or the zero-rare-FP gate to these
individual rows.

Saved numeric evidence: `results.json`, `LEADERBOARD.md`, `margins.json`,
`diagnosis-student.json`, `diagnosis-niw.json`, and
`diagnosis-anchor-posterior.json`. Other `diagnosis-*.json` files retain rejected
causal decompositions. Every experiment records its source/input hashes; no
GPU or learned-training portability claim is made here.
'''
    (ROOT/'REPORT.md').write_text(report)
    own={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size)
         for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    manifest=dict(status='UNQUALIFIED_RESEARCH',created_utc=datetime.now(timezone.utc).isoformat(),
        own_artifacts=own,external_research={str(p):sha(p) for p in
            (HIGH/'fisher_rank.py',HIGH/'test_metric_contract.py',HIGH/'metric-contract.json',
             HIGH/'diagnosis-local-anchors.json',HIGH/'diagnosis-nearest-local.json',OLD)},
        source_merged=False,gpu_executions=0,proven_performance_changed=False)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(event='complete',status=result['status'],actual_anchor_passed=14,total=18,
                         reports={p:sha(ROOT/p) for p in ('REPORT.md','LEADERBOARD.md','results.json','manifest.json')})))


if __name__=='__main__':
    main()
