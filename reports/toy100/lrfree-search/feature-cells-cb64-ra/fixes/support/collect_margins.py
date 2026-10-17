"""Report existing score margins and captured precision; no new score law."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import json
import math
from pathlib import Path
import torch
from diagnose import ROOT,common,geometry,setup,conformal,sha
from diagnose_niw import niw_score,HIGH


@torch.no_grad()
def main():
    shared=setup()
    paths=(Path(__file__),ROOT/'diagnose.py',ROOT/'diagnose_niw.py',HIGH/'fisher_rank.py',
           common.PREVIOUS/'geometry_a'/'bundle.pt',common.CONFIG)
    hashes={str(p):sha(p) for p in paths}
    result=dict(scope='CPU causal margins of existing NIW diagnostic; no new model',
                source_sha256=hashes,cases=[])
    cases=[('geometry',1024,'fold',128,'trained600'),
           ('geometry',2048,'fold',128,'trained600'),
           ('geometry',2048,'fold',128,'frozen_initialization'),
           ('geometry',4096,'fold',128,'frozen_initialization')]
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        captured=[]
        handle=ev.D.score.register_forward_pre_hook(lambda module,inputs:captured.append(str(inputs[0].dtype)))
        R=bd._features(trainer,ev.real_raw);q=bd._features(trainer,ev.G(ev.z))
        handle.remove()
        source_dtype=next(ev.D.parameters()).dtype
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        score,arrays,metadata=niw_score(snap,R[0::2],source_dtype)
        qs,rs=score(q),score(R[1::2]);flags,p=conformal(snap,qs,rs)
        bad=ev.initial_modes==ev.unsupported_bin
        null_sorted,null_ids=rs.sort(descending=True)
        badrows=bad.nonzero().flatten()
        badbest=badrows[p[badrows].argmin()]
        ordered_p=p.sort().values
        bh_threshold=.05*torch.arange(1,len(p)+1,dtype=p.dtype)/len(p)
        details=[]
        ids=[1745,1773,1924] if ev.n==2048 else ([3262] if ev.n==4096 else [])
        qbest=arrays(q).argmin(1)
        for rowid in ids:
            c=int(qbest[rowid])
            real_u=ev.fixture.real_u
            # Oracle-space diagnostics follow fitting, calibration, and flags.
            # These values and mode labels are never inputs to niw_score.
            semantic=ev.G.semantic(ev.z)[rowid]
            real_rare=shared.geometry.oracle_modes(real_u)==ev.rare_bin
            nearest_even_u=(real_u[0::2]-semantic).norm(dim=1).min()
            nearest_odd_u=(real_u[1::2]-semantic).norm(dim=1).min()
            details.append(dict(row=rowid,flagged=bool(flags[rowid]),oracle_mode=int(ev.initial_modes[rowid]),
                semantic_position=semantic.tolist(),nearest_even_oracle_distance=float(nearest_even_u),
                nearest_odd_oracle_distance=float(nearest_odd_u),best_support_cell=c,
                even_cell_rows=int(snap.reference_counts[c]),odd_partition_rows=int(snap.real_calibration_counts[c]),
                posterior_degrees=float(metadata['posterior_degrees'][c]),
                posterior_scale_eigenvalues=metadata['posterior_covariance_eigenvalues'][c],
                score=float(qs[rowid]),p=float(p[rowid]),
                score_margin_above_null_max=float(qs[rowid]-null_sorted[0]),
                null_rows_at_or_above_query=int((rs>=qs[rowid]).sum()),
                real_rare_even_rows=int(real_rare[0::2].sum()),real_rare_odd_rows=int(real_rare[1::2].sum())))
        row=dict(fixture=case,detector=ev.detector(flags),
            head_input_dtypes=sorted(set(captured)),head_parameter_dtype=str(source_dtype),
            converted_feature_dtype=str(R.dtype),dictionary_rank=metadata['dictionary_rank'],
            support_rank=metadata['rank'],calibration_rows=len(rs),
            empirical_floor=1/(1+len(rs)),maximum_BH_p_at_five_percent_action_guard=.05*.05,
            best_unsupported_p=float(p[badbest]),worst_unsupported_p=float(p[bad].max()),
            minimum_flag_fraction_needed_for_best_unsupported=float(p[badbest]/.05),
            queries_above_maximum_null_score=int((qs>rs.max()).sum()),
            unsupported_above_maximum_null_score=int((qs[bad]>rs.max()).sum()),
            queries_needed_at_empirical_floor_for_any_BH_rejection=math.ceil(ev.n/(.05*(1+len(rs)))),
            minimum_ordered_p_to_BH_threshold_ratio=float((ordered_p/bh_threshold).min()),
            unsupported_min_score=float(qs[bad].min()),unsupported_max_score=float(qs[bad].max()),
            null_highest_scores=null_sorted[:6].tolist(),
            null_highest_original_real_rows=(2*null_ids[:6]+1).tolist(),rare_rows=details)
        result['cases'].append(row)
        print(json.dumps(dict(event='case',fixture=case,best_unsupported_p=row['best_unsupported_p'],
              minimum_flag_fraction_needed=row['minimum_flag_fraction_needed_for_best_unsupported'],
              null_highest_scores=row['null_highest_scores'],rare_rows=details)),flush=True)
    assert hashes=={str(p):sha(p) for p in paths}
    assert not torch.cuda.is_initialized()
    result['cuda_initialized']=False
    (ROOT/'margins.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
