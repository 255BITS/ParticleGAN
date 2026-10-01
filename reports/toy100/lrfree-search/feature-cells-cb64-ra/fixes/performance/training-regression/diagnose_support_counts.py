"""Decompose saved categorical counts by their unchanged support evidence."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
FIXES = ROOT.parent.parent
SNAPSHOTS = FIXES/'integration'/'review'/'training-regression'
PKG = FIXES/'pkg-CB64-RA2'
sys.path.insert(0, str(PKG))
from particlegan.feature_cells import FeatureCellSnapshot, Q


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(v):
    if isinstance(v, torch.Tensor): return v.tolist()
    if isinstance(v, dict): return {k:plain(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [plain(x) for x in v]
    return v


@torch.no_grad()
def characterize(snap, features):
    ids, _ = snap.assign(features)
    flags, p, score = snap.support(features)
    total = torch.bincount(ids, minlength=snap.cells)
    eligible = torch.bincount(ids[p > Q], minlength=snap.cells)
    nonflagged = torch.bincount(ids[~flags], minlength=snap.cells)
    return dict(rows=len(features), counts=total, eligible_counts=eligible,
        nonflagged_counts=nonflagged, flagged_counts=total-nonflagged,
        flagged_rows=int(flags.sum()), pointwise_eligible_rows=int((p > Q).sum()),
        flag_fraction=float(flags.double().mean()), eligible_fraction=float((p > Q).double().mean()),
        score_min=float(score.min()), score_median=float(score.median()), score_max=float(score.max())), (ids,flags,p,score)


@torch.no_grad()
def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = [Path(__file__), ROOT/'PROTOCOL.md', PKG/'particlegan'/'feature_cells.py',
        SNAPSHOTS/'diagnose_saved.py', SNAPSHOTS/'paired-count-rules.json']
    sources += [SNAPSHOTS/f'snapshot-{step:04d}.pt' for step in (1000,2000)]
    hashes = {str(p):sha(p) for p in sources}
    output = dict(scope='CPU saved reconstruction; support/count descriptive decomposition only',
        new_seeds=0, source_sha256=hashes, q=Q,
        real_control_scope='same heldout calibration sample compared with its own empirical null; in-sample reference',
        original_controller_has_unsupported_count_category=False, cases=[])
    for step in (1000,2000):
        payload = torch.load(SNAPSHOTS/f'snapshot-{step:04d}.pt', map_location='cpu', weights_only=False)
        snap = FeatureCellSnapshot.__new__(FeatureCellSnapshot)
        snap.__dict__.update(payload['snapshot'])
        real, real_items = characterize(snap, payload['real_features'][1::2])
        emitted, emitted_items = characterize(snap, payload['fake_features'])
        table, table_items = characterize(snap, payload['q'])
        assert torch.equal(table_items[1], payload['flags'])
        assert torch.equal(table_items[2], payload['pvalues'])
        comparison = payload['comparison']
        assert torch.equal(real['counts'], comparison['real_counts'])
        assert torch.equal(emitted['counts'], comparison['fake_counts'])
        rf = real['counts'].double()/real['rows']; ff = emitted['counts'].double()/emitted['rows']
        rs = real['eligible_counts'].double()/real['rows']
        fs = emitted['eligible_counts'].double()/emitted['rows']
        augmented_tv = float(((rs-fs).abs().sum()+((rf-rs)-(ff-fs)).abs().sum())*.5)
        balanced = ~comparison['excess'] & ~comparison['deficit']
        case = dict(step=step, checkpoint=payload['checkpoint'], checkpoint_sha256=payload['checkpoint_sha256'],
            identical_support_flags=True, identical_cell_counts=True, cells=snap.cells,
            real_calibration=real, emitted_fake=emitted, clean_table=table,
            original_count_comparison=comparison,
            full_cell_mass_tv=float((rf-ff).abs().sum()*.5),
            eligibility_augmented_cell_mass_tv=augmented_tv,
            supported_measure_mass_difference=float(rs.sum()-fs.sum()),
            original_count_discoveries=int((~balanced).sum()),
            no_discovery_cells=int(balanced.sum()),
            emitted_flagged_rows_in_no_discovery_cells=int(emitted['flagged_counts'][balanced].sum()),
            table_flagged_rows_in_no_discovery_cells=int(table['flagged_counts'][balanced].sum()),
            no_discovery_cells_with_a_supported_mass_deficit=int((balanced & (rs > fs)).sum()),
            emitted_pointwise_support_fraction_by_cell=torch.where(emitted['counts'] > 0,
                emitted['eligible_counts'].double()/emitted['counts'].clamp_min(1), torch.zeros(snap.cells)),
            real_pointwise_support_fraction_by_cell=torch.where(real['counts'] > 0,
                real['eligible_counts'].double()/real['counts'].clamp_min(1), torch.zeros(snap.cells)))
        output['cases'].append(case)
        print(json.dumps(dict(event='case',step=step,
            real_eligible_fraction=real['eligible_fraction'], emitted_eligible_fraction=emitted['eligible_fraction'],
            table_eligible_fraction=table['eligible_fraction'], emitted_flags=emitted['flagged_rows'],
            table_flags=table['flagged_rows'], count_discoveries=case['original_count_discoveries'],
            full_mass_tv=case['full_cell_mass_tv'], eligibility_augmented_tv=augmented_tv,
            emitted_flags_without_count_discovery=case['emitted_flagged_rows_in_no_discovery_cells'],
            table_flags_without_count_discovery=case['table_flagged_rows_in_no_discovery_cells'])), flush=True)
    output.update(cuda_initialized=torch.cuda.is_initialized(), sources_unchanged=hashes == {p:sha(p) for p in hashes})
    assert not output['cuda_initialized'] and output['sources_unchanged']
    (ROOT/'support-count-diagnosis.json').write_text(json.dumps(plain(output), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(event='complete',cuda_initialized=False,sources_unchanged=True)), flush=True)


if __name__ == '__main__': main()
