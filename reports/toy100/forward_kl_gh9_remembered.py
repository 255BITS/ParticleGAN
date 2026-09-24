"""Post-rest remembered-data rescue for the frozen finite-GH9 output rule.

The frozen current-bank operator runs first and is returned byte-for-byte on
every non-rest path. Only on exact rest do remembered real coordinates enter
a deterministic GH9 donor search. The first strict actual GH9 improvement is
followed by one GH9 EM step, or the original cloud rests if none exists.
No target label, clock gain, random draw, or neural update enters this rule.
"""

import hashlib
from pathlib import Path

import torch

from reports.toy100 import (
    forward_kl_chunked as donor_module,
    forward_kl_gh9_stress as frozen,
    forward_kl_remembered_donor_rescue as rescue_module,
)
from reports.toy100.forward_kl_chunked import donor_values
from reports.toy100.forward_kl_free_filter import (
    quadrature, cross_entropy, em_centroids, gradient,
)
from reports.toy100.forward_kl_remembered_donor_rescue import remembered_first_descent


METHOD='forward_kl_gh9_remembered_v1'
FROZEN_SHA='adc5d6f6ded8e6cf297ee90aa0d540723becc632a601b93b5b9a4393cd344485'
DONOR_SHA='e0977838c7d5ae787f5194e653d1a8a09d5c7c0af7eb5f48b65af49a303b141f'
RESCUE_SHA='21828acbd6c823ae53fb0da343746442278eeeb4f68e8ece04e20fe85f346471'


def source_hashes():
    paths={
        'frozen':Path(frozen.__file__),
        'donor':Path(donor_module.__file__),
        'rescue':Path(rescue_module.__file__),
    }
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest()
            for name,path in paths.items()}
    if (hashes['frozen']!=FROZEN_SHA or hashes['donor']!=DONOR_SHA
            or hashes['rescue']!=RESCUE_SHA):
        raise RuntimeError('frozen forward-KL dependency source changed')
    return hashes


def _without_oracle_quality(row):
    result=dict(row)
    result.pop('initial_quality',None)
    result.pop('final_quality',None)
    return result


def optimize(history, bank, initial, width, sigma, *, means=None):
    """Return (12x2 clean output, JSON-compatible receipt) without mutation.

    `means=None` uses a fixed neutral dummy only to satisfy the frozen
    diagnostic receipt API, then drops its two quality fields. The output
    computation and all GH5/GH9 fields are independent of that dummy.
    Explicit means are diagnostic-only and preserve exact frozen receipts
    on non-rest paths, for archive parity checks.
    """
    source_hashes()
    without_quality=means is None
    diagnostic_means=(torch.zeros((8,2),dtype=torch.float32) if without_quality
                      else means)
    before_rng=torch.random.get_rng_state().clone()
    old_output,old_row=frozen.optimize(history,bank,initial,width,sigma,
                                       means=diagnostic_means)
    if not torch.equal(before_rng,torch.random.get_rng_state()):
        raise RuntimeError('frozen pure-output rule changed global Torch RNG')
    if old_row['selected']!='EXACT_REST':
        return old_output,(_without_oracle_quality(old_row) if without_quality
                           else old_row)
    original=initial.detach().double().clone()
    if not torch.equal(old_output,original):
        raise RuntimeError('frozen exact-rest output differs from input')
    target=history.detach().double()
    finite9=quadrature(target,width,9)
    variance=width**2+sigma**2
    donor,donor_row=remembered_first_descent(target,original,*finite9,variance,
        donor_values=donor_values,cross_entropy=cross_entropy)
    if donor_row['status']=='NO_SINGLE_REMEMBERED_DONOR':
        row=dict(old_row,remembered_search=donor_row)
        return old_output,(_without_oracle_quality(row) if without_quality else row)
    if donor_row['status']!='STRICT_FINITE_GH9_DONOR':
        raise RuntimeError('unexpected remembered donor search status')
    em,em_rows=em_centroids(donor,*finite9,variance,limit=1,audit=finite9)
    final=em if em_rows else donor
    after9=float(cross_entropy(*finite9,final,variance))
    before9=old_row['initial_audit9']
    tolerance=old_row['gh9_strict_tolerance']
    if (after9>donor_row['actual_cost']+tolerance or
            after9>=before9-tolerance or
            not torch.equal(before_rng,torch.random.get_rng_state())):
        raise RuntimeError('remembered donor/EM failed finite-GH9 or RNG guard')
    finite5=quadrature(target,width,5)
    after5=float(cross_entropy(*finite5,final,variance))
    grad5=gradient(*finite5,final,variance)
    grad9=gradient(*finite9,final,variance)
    row=dict(old_row,selected='REMEMBERED_GH9_DONOR',
        frozen_selection='EXACT_REST',remembered_search=donor_row,
        remembered_em=em_rows,final_cross_entropy=after5,
        final_audit9=after9,final_gradient_5_l2=float(grad5.norm()),
        final_gradient_9_l2=float(grad9.norm()),
        final_gradient_max_absolute_discrepancy=float((grad5-grad9).abs().max()),
        final_points=final.tolist(),
        max_output_displacement=float((final-original).norm(dim=1).max()))
    if not without_quality:
        row['final_quality']=frozen.quality(final,diagnostic_means)
    return final,(_without_oracle_quality(row) if without_quality else row)


def propose_target(history_real, current_bank, pre_clean_points, width, output_sigma):
    """Neural adapter API: target, oracle-free receipt, ephemeral GH9 rule.

    The rule is recomputed once after the frozen operator and is returned so
    the caller can check its *actual fitted output* against exactly the same
    cumulative finite-GH9 objective. It is never serialized into a receipt.
    """
    target,receipt=optimize(history_real,current_bank,pre_clean_points,width,
                            output_sigma,means=None)
    finite9=quadrature(history_real.detach().double(),width,9)
    variance=width**2+output_sigma**2
    before=float(cross_entropy(*finite9,pre_clean_points.detach().double(),variance))
    after=float(cross_entropy(*finite9,target,variance))
    tolerance=1e-10*max(1.,abs(before),abs(after))
    if (abs(before-receipt['initial_audit9'])>tolerance or
            abs(after-receipt['final_audit9'])>tolerance):
        raise RuntimeError('ephemeral GH9 rule differs from target receipt')
    return target,receipt,finite9
