"""Generate the CPU saved-state regression report from recorded evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT/'state-diagnosis.json').read_text())
    counts = json.loads((ROOT/'support-count-diagnosis.json').read_text())
    assert not state['cuda_initialized'] and not counts['cuda_initialized']
    assert state['sources_unchanged'] and counts['sources_unchanged']
    assert all(v['finite'] for row in state['cases'] for v in row['model_tensor_summaries'].values())
    assert len(state['cases']) == 12 and len(counts['cases']) == 2
    final = [r for r in state['cases'] if r['step'] == 2000]
    summary = dict(status='DIAGNOSED_CPU_ONLY', adopted_patch=False,
        findings=['No accidental EMA serving, model/Adam nonfinite values, missing optimizer updates or detached prior gradients found.',
                  'RA2 toy gradients are sensitive to its larger kernel on the same saved data/noise; this does not qualify a smaller kernel.',
                  'Nearest-cell counts omit support location within each cell, hiding most support flags from categorical discovery.'],
        canonical_cuda_records=[dict(problem=r['problem'], variant=r['variant'],
            training_seconds=r['original_cuda_record']['training_seconds'], metrics=r['original_cuda_record']['metrics']) for r in final],
        gradient_comparisons=[dict(problem=r['problem'], variant=r['variant'], step=r['step'],
            generator_cosine=r['gradient_probe']['generator_gradient_cosine'],
            prior_cosine=r['gradient_probe']['prior_gradient_cosine'],
            opposed_prior_rows=r['gradient_probe']['prior_gradient_opposed_rows'],
            metrics=r['gradient_probe']['methods']) for r in state['cases']],
        count_comparisons=[{k:r[k] for k in ('step','full_cell_mass_tv','eligibility_augmented_cell_mass_tv',
            'original_count_discoveries','no_discovery_cells','emitted_flagged_rows_in_no_discovery_cells',
            'table_flagged_rows_in_no_discovery_cells','no_discovery_cells_with_a_supported_mass_deficit')} for r in counts['cases']],
        source_sha256={name:sha(ROOT/name) for name in ('PROTOCOL.md','diagnose_state.py',
            'diagnose_support_counts.py','state-diagnosis.json','support-count-diagnosis.json')},
        new_seeds=0, optimizer_updates=0, cuda_initialized=False, source_or_gate_changes=False)
    (ROOT/'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Saved learned-state regression', '',
        '## CUDA outcome recorded by the frozen learned harness', '',
        'These are the original CUDA measurements, not a CPU quality retest. All three toy variants fail the unchanged P≥.9, 25-mode, TV≤.1 gate. MNIST has no invented acceptance threshold.', '',
        '| Variant | Toy P | Modes | TV | Toy seconds | MNIST active FD | Active P | Active R | Class TV | MNIST seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    by = {(r['problem'],r['variant']):r for r in final}
    for variant in ('E22','CB64-RA','CB64-RA2'):
        tr, mr = by['toy',variant]['original_cuda_record'], by['mnist',variant]['original_cuda_record']
        t, m = tr['metrics'], mr['metrics']; a=m['active_embedding']
        lines.append(f"| {variant} | {t['precision']:.4f} | {t['coverage']} | {t['mass_tv']:.4f} | {tr['training_seconds']:.2f} | {a['embedding_frechet']:.5f} | {a['embedding_precision']:.4f} | {a['embedding_recall']:.4f} | {m['class_mass_tv']:.4f} | {mr['training_seconds']:.2f} |")
    lines += ['', '## Same-input conditional gradient comparison', '',
        'Fast training G/D weights, first128 prior rows, that update’s original generator real batch, and the same saved fold128/fold2 noise. Columns compare the existing latent kernel against zero latent jitter; both include the same toy output noise. No critic or optimizer update is performed. These 128-row coverages are diagnostic observations, not the 8192-sample acceptance run.', '',
        '| Variant | Step | G gradient cosine | Prior gradient cosine | Opposed prior rows /128 | Zero-jitter P / modes | Existing-kernel P / modes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in state['cases']:
        if r['problem'] != 'toy': continue
        g=r['gradient_probe']; c=g['methods']['no_latent_jitter']['conditional_128_row_output_metrics']; n=g['methods']['saved_variant_kernel']['conditional_128_row_output_metrics']
        lines.append(f"| {r['variant']} | {r['step']} | {g['generator_gradient_cosine']:.4f} | {g['prior_gradient_cosine']:.4f} | {g['prior_gradient_opposed_rows']} | {c['precision']:.4f} / {c['coverage']} | {n['precision']:.4f} / {n['coverage']} |")
    lines += ['', '## Same-snapshot count decomposition', '',
        'Existing score, null and Q=.05 remain fixed. Pointwise support here means the current p>Q parent eligibility. Its boundary uses odd calibration, so these augmented counts are descriptive and are not a valid new fixed-partition test. An eventual count boundary must be fitted on even real rows alone.', '',
        '| Step | Real eligible | Emitted fake eligible | Clean table eligible | Emitted / table BH flags | Count discoveries /64 | Emitted flags without count discovery | Table flags without count discovery | Full / augmented TV |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in counts['cases']:
        a,b,c=r['real_calibration'],r['emitted_fake'],r['clean_table']
        lines.append(f"| {r['step']} | {a['eligible_fraction']:.4f} | {b['eligible_fraction']:.4f} | {c['eligible_fraction']:.4f} | {b['flagged_rows']} / {c['flagged_rows']} | {r['original_count_discoveries']} | {r['emitted_flagged_rows_in_no_discovery_cells']} | {r['table_flagged_rows_in_no_discovery_cells']} | {r['full_cell_mass_tv']:.4f} / {r['eligibility_augmented_cell_mass_tv']:.4f} |")
    (ROOT/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
    report = '''# Learned regression diagnosis

## Findings

The remaining toy failure is a diffuse distribution with little mass near any of the 25 supported modes. Its single covered mode is the only mode exceeding the coverage mass threshold; outputs are not concentrated in a single mode. A structural blind spot is reproduced: ordinary count evidence assigns unsupported emitted points to the nearest existing cell and contains no support/unsupported category. Most unsupported rows therefore produce no categorical count discovery. The broad-flag isolation guard then prevents their repair.

No additional serving or optimizer implementation error was found in the 12 saved states. RA2 serves fast G/prior at toy1000/2000 because the prior tester is in drift (`last_decisive=1`). E22 serves EMA at both toy checkpoints; old CB serves EMA at2000. RA2's clean final precision is .2910 with fast weights and .3096 with EMA, both far below E22's .8203 clean EMA precision. Serving a different saved average cannot rescue this state. At1000 RA2 EMA is worse than fast (.1201 versus .1611). These comparisons contain no latent or output noise.

All model and Adam moment tensors are finite, every optimizer parameter's step counter equals the saved1000/2000 update, and prior moments remain active. The final toy prior has117 active last-gradient rows in each variant, all1024 rows remain exactly distinct, and A2's cumulative observed-row fraction is .1176 in every variant. The tested latent perturbation has an exactly identity derivative to the indexed prior rows. CUDA RNG bytes were never loaded into a CPU generator or trainer.

## LR and controller explanation

The final toy generator-network LR is6.640625e-5 in all three variants. RA2 prior LR=.0085 and critic LR=.0031874155 are2×old CB and4×E22, respectively. This follows the unchanged trainer rule `critic_scale=max(critic_tester.s,.75*prior_tester.s)`, followed by payoff damping. RA2's prior scale remains1 while old CB/E22 reach.5/.25; the critic LR floor therefore remains high even though RA2's own critic tester scale is4.6566e-10. This is consistent controller state, not a restored-LR mismatch. It is a feedback consequence of the unsettled prior, and it does not justify changing the LR recipe after seeing quality.

Learnable toy output sigma is floored at the recipe's .029 in all final states. The saved output-noise gradient is zero. Undefined stationary-window diagnostic entries are preserved explicitly as `NaN`/`Infinity` strings plus their source paths: the source uses NaN to exclude copied/unobserved rows and for undefined statistics. These placeholders are not nonfinite model or Adam moments. The initial default-Python SciPy/NumPy import error and the first strict-JSON serialization error are retained in failed logs; final scripts run with the frozen harness environment.

## Training-field sensitivity

The matched toy2000 conditional probe gives RA2 existing-jitter versus zero-jitter gradient cosine .0113 for G and .1485 for the prior, with60/128 prior rows having an opposed direction. E22's corresponding cosines are .9283/.6632 with19 opposed rows; old CB's tiny kernel gives .9998/.9986 with zero opposed rows. The larger kernel changes the training field, rather than merely broadening samples at evaluation. This is one saved batch and a fixed table enumeration, and does not establish that zero or smaller jitter is the correct learning rule. E22 and old CB both already fail the canonical toy gate.

The effect is problem-dependent. MNIST2000 G/prior cosines are .7104/.3474 for E22 and .7433/.4111 for RA2. RA2's canonical MNIST active FD=.81414 and recall=.7139 improve on old CB's1.48883/.3228, while E22 remains better at .54449/.8472. Large gradient differences alone cannot explain or rank learned quality universally.

## Within-cell support blind spot

The frozen CPU reconstruction at1000 has887 flagged clean rows and913 flagged emitted rows. Only6/64 cells have a categorical count discovery. Of those flagged rows,805 clean and788 emitted rows are in the58 cells without count discovery. The real calibration point-eligibility fraction is.9531, emitted fake is.1016 and clean table is.1250. At2000 the corresponding fractions are.9531/.1572/.2861, with710/860 clean/emitted flags and5 discoveries;688/820 flags are in cells without discovery.

Full categorical mass TV is.3096/.2734 at1000/2000, so full counts are not exactly balanced. The existing multiplicity-corrected count test detects few discrepancies, and cannot see the much larger support difference within those cells. Decomposing each cell descriptively by current eligibility increases TV to.8604/.8008. This uses no oracle mode label. Fifty-six and52 cells without count discovery still have lower eligible emitted mass than real calibration mass. Real calibration has zero BH flags against its own null; this is an in-sample calibration description, not an independent false-positive qualification.

The missing category is independent of the previously diagnosed parent/target accounting restriction. Simply recovering count-certified moves still leaves few actions and no signal for many same-cell holes. The support score's earlier rare false-positive failures also remain unresolved. No score, support law, Q, guard, partition or acceptance gate was changed in this diagnosis.

## Scope and next test

Only CPU reads, forward/backward probes and descriptive counts were run. There were no optimizer updates, extra seeds or CUDA contexts. The stability snapshots reuse its saved CPU projection and sampling; they do not reproduce CUDA projection RNG or the CUDA birth/death snapshot exactly (archived GPU flags888/711 versus reconstructed887/710). All source/checkpoint/data/noise hashes were verified unchanged.

The qualified next investigation is one real-only, even-fit inside/outside partition evaluated on untouched odd real and emitted fake counts, with the existing exact count law and multiplicity accounting. Its support boundary must not reuse the odd-calibration p threshold used for this descriptive table. Parent availability, full-reference targets, unique-parent and supported-deletion guarantees need independent checks. This report qualifies no new score or integration patch. The conservative accounting and lineage experiment proceeds separately.

See [LEADERBOARD.md](LEADERBOARD.md), [state-diagnosis.json](state-diagnosis.json), [support-count-diagnosis.json](support-count-diagnosis.json), and [PROTOCOL.md](PROTOCOL.md) for measurements, receipts and runnable commands.
'''
    (ROOT/'REPORT.md').write_text(report)
    generated = ['REPORT.md','LEADERBOARD.md','results.json']
    manifest = dict(status='DIAGNOSED_CPU_ONLY', adoption='NO_NEW_SCORE_OR_GATE_CHANGE',
        generated_sha256={name:sha(ROOT/name) for name in generated},
        input_sha256=summary['source_sha256'], cuda_initialized=False, optimizer_updates=0,
        qualifications='Conditional CPU diagnosis; unchanged frozen CUDA outcomes; no canonical CPU-to-GPU inference',
        failed_preparatory_logs=['failed-default-python.log','failed-strict-json.log'],
        root_only_gpu_execution_preserved=True)
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    freeze_names = [p for p in ROOT.iterdir() if p.is_file() and p.name != 'FROZEN.json']
    frozen = dict(scope='Completed saved-state/gradient/count diagnosis; immutable evidence',
        local_artifact_sha256={p.name:sha(p) for p in sorted(freeze_names)},
        adopted_patch=False, new_seeds=0, optimizer_updates=0, cuda_initialized=False)
    (ROOT/'FROZEN.json').write_text(json.dumps(frozen, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'], report_sha256=sha(ROOT/'REPORT.md'),
        leaderboard_sha256=sha(ROOT/'LEADERBOARD.md'), manifest_sha256=sha(ROOT/'manifest.json'),
        freeze_sha256=sha(ROOT/'FROZEN.json')), indent=2))


if __name__ == '__main__': main()
