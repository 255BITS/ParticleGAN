"""Render the measured scoped readout from compact certified results only."""
from pathlib import Path
import json

OUT=Path(__file__).resolve().parent


def main():
    results=json.loads((OUT/'results.json').read_text())
    diagnostics=json.loads((OUT/'current-component-diagnostics.json').read_text())
    provenance=json.loads((OUT/'provenance.json').read_text())
    verification=json.loads((OUT/'software-verification.json').read_text())
    arms=results['arms']
    tasks={arm:{r['task_id']:r for r in data['tasks']} for arm,data in arms.items()}
    def value(x):return 'unavailable' if x is None else f'{x:.6f}'
    def row(arm,task):return tasks[arm][task]
    def metric(arm,task,key):return row(arm,task)['final'].get(key)
    def short(arm,task):
        r=row(arm,task)
        if task=='gaussian1d_smoke':
            return f"{r['status']}; first confirmed {r['sustained_gate'].get('first_confirmed_step')}; {r['passing_checks']}/24 primary checks"
        if task=='gaussian1d_stability':
            g=r['sustained_gate']
            return f"{r['status']}; stationary {g['stationary_passes']}/72, shifted {g['shift_hold_passes']}/24, deadline {g['reacquisition_status']}"
        if task=='grid100':
            return f"{r['status']}; holdout precision {value(r['final'].get('precision'))}, {r['terminal_accuracy_passing_checks']}/5 terminal checks"
        return f"{r['status']}; covariance {value(r['final']['component_covariance_error'])}; {r['passing_checks']}/24, suffix {r['terminal_passing_suffix']}"
    lines=['# Round 5: component centers and kernel jitter','',
        '**Reject this exact prior-only routing replacement and retain local-v2’s scoped rare-density repair.** '
        f"The candidate completes **{arms['candidate']['outcomes'].get('PASS',0)} PASS / {arms['candidate']['outcomes'].get('FAIL',0)} FAIL**, "
        f"versus matched local-v2 **{arms['control']['outcomes'].get('PASS',0)} PASS / {arms['control']['outcomes'].get('FAIL',0)} FAIL**. "
        'All twelve jobs complete at their original full update budgets, including two 7k native runs. '
        f"Paid worker time is **{results['cost']['spent_seconds']:.6f} seconds**, with zero retries or remaining reservations. "
        '[PR376](https://github.com/255BITS/ParticleGAN/pull/376) is ready and unmerged.','',
        '## Complete matched results','',
        '| Original unchanged task | Exact local-v2 control | Prior-only transport |',
        '| --- | --- | --- |']
    order=['gaussian1d_smoke','gaussian1d_stability','vector_unequal_mass','vector_unequal_width','vector_two_broad','grid100']
    for task in order:lines.append(f'| {task} | {short("control",task)} | {short("candidate",task)} |')
    lines += ['', '[Complete metrics and final-window failures](results.json), '
        '[source, protocol and cost receipts](provenance.json), '
        '[current center/kernel and uncensored tail diagnostics](current-component-diagnostics.json), '
        'and [exact archived-control parity](predecessor-parity.json) bind these outcomes. '
        'This is a scoped diagnostic comparison; the single current goal leaderboard and original ordinary qualifications are unchanged.','',
        'The width forecast≤1.25 is observed, and its final average covariance also satisfies the original .85 endpoint bound. '
        'It has eight passing scheduled observations but only one terminal pass; five are required. '
        'All four modes remain present, so this gain is not the missing-mode averaging artifact of the rejected tail-moment package. '
        f"Its final counts are {metric('candidate','vector_unequal_width','component_counts')}; "
        f"mass TV {value(metric('candidate','vector_unequal_width','mass_tv'))}, "
        f"HQ {value(metric('candidate','vector_unequal_width','hq'))}, "
        f"core covariance {value(metric('candidate','vector_unequal_width','component_core_covariance_error'))}. "
        'The first narrow component still has full covariance error 2.349925 and spill .176923; the lower average is not uniform tail fidelity.','',
        'The required preservation forecast fails: unequal mass loses its sustained PASS. '
        f"Full covariance worsens {value(metric('control','vector_unequal_mass','component_covariance_error'))}→"
        f"{value(metric('candidate','vector_unequal_mass','component_covariance_error'))}, and the suffix falls 5→0. "
        f"Candidate counts {metric('candidate','vector_unequal_mass','component_counts')} retain every component; "
        f"mass TV {value(metric('candidate','vector_unequal_mass','mass_tv'))} passes. "
        'This is a shape/retention regression rather than total rare-mode disappearance. Broad remains PASS, '
        f"while its full covariance worsens {value(metric('control','vector_two_broad','component_covariance_error'))}→"
        f"{value(metric('candidate','vector_two_broad','component_covariance_error'))} and suffix becomes 23. No specialist passes are pooled.", '',
        'Gaussian final KS improves '
        f"{value(metric('control','gaussian1d_stability','cdf_ks'))}→{value(metric('candidate','gaussian1d_stability','cdf_ks'))}; "
        'the candidate endpoint passes every scalar bound. Nevertheless stationary passes drop 28/72→10/72, '
        'shifted hold drops 11/24→9/24, and deadline reacquisition fails. Both continuations restore their own exact '
        'eligible smoke state and complete all 5000 new updates with the original schedule and streams. Endpoint accuracy is not retention.','',
        '## Native and component diagnostics','',
        'Native full quality/coverage/accuracy remains FAIL in both arms. Independent holdout precision is '
        f"{value(metric('control','grid100','precision'))}→{value(metric('candidate','grid100','precision'))}; "
        'the original bound is .97. Undefined accuracy shape fields remain unavailable, never zeros or surrogate passes. '
        'The separate uncensored audit uses every saved final 20k clean/live draw, with nearest-cell assignment and population covariance. '
        'It supplies diagnostics, not a new gate or replacement for the original independent holdout.','',
        '| Native saved final-law diagnostic | Local-v2 | Prior-only |','| --- | ---: | ---: |']
    native={r['arm']:r for r in diagnostics['native_uncensored']}
    coverage_keys=['precision','modes','min_hq_mode_mass','mass_tv','min_cov_eig_ratio','max_cov_eig_ratio','min_radial_median_ratio','max_radial_median_ratio']
    for key in coverage_keys:
        lines.append(f"| Original 20k {key} | {value(row('control','grid100')['coverage_final'][key])} | {value(row('candidate','grid100')['coverage_final'][key])} |")
    for key in ['component_covariance_error','component_min_eigen_ratio','global_spill','max_component_spill','uncensored_radial_ks']:
        lines.append(f"| Uncensored {key} | {value(native['control'][key])} | {value(native['candidate'][key])} |")
    width_census={r['arm']:r for r in diagnostics['diagnostics'] if r['task_id']=='vector_unequal_width'}
    lines += ['', '| Unequal-width fixed-center diagnostic | Local-v2 | Prior-only |',
              '| --- | ---: | ---: |']
    for component in (0,1):
        for key in ('center_output_trace_over_target','conditional_jitter_trace_over_target'):
            a=width_census['control']['component_summary'][component][key]
            b=width_census['candidate']['component_summary'][component][key]
            lines.append(f'| Narrow component {component}: {key} | {value(a)} | {value(b)} |')
    lines += ['', 'Both narrow center clouds improve in this endpoint census, while their conditional kernel-jitter '
        'contributions increase. The average between-conditional-mean fraction falls .993170→.928961. '
        'This separates the measured contributor changes; it does not reconstruct or causally attribute the entire training history.']
    lines += ['', 'The [saved preselection census](saved-component-diagnostics.json) establishes that local-v2 vector tails '
        'are dominated by the center population: the two narrow center trace ratios are 5.122888/4.328984, '
        'against jitter .057369/.020522. The nonlinear quadrature mean between fraction is .993170. '
        'This supports investigating individual location motion, without attributing the entire training failure to G. '
        'The current census reports both arms and keeps fixed center assignment, assignment migration and nonlinear integration approximations explicit. '
        'Exact center outputs and finite quadrature covariance algebra are distinct from the approximate Gaussian conditional moments. '
        'Vector radial KS uses uncensored whitened nearest-cell radii against χ²₂; overlap can alter that reference, so it is diagnostic only.','',
        '**Stop this exact global routing candidate.** The width endpoint benefit is real under the matched law, '
        'but rare-density preservation and full temporal gates fail. Retain the source-bound local-v2 positives and their limitations. '
        'A possible useful question is how to preserve shared-map allocation while controlling individual center tails, '
        'but this study authorizes no second candidate, sweep, extension, seed repeat or promotion. '
        'Conditional hosts, images, words/rings and ordinary full Tier2 transfer are unmeasured. '
        'Original unsupported two-pole contracts retain their archived BLOCKED identity and are outside this six-task question.','',
        '## Source, verification and reproduction','',
        f"Both arms execute commit `{arms['control']['source_origin_commit']}` and digest `{arms['control']['source_digest']}`. "
        'Every exercised scientific file and frozen snapshot is verified unchanged. Later edits only clarify diagnostic captions '
        'or add publication/reproduction files. The sole consumed global recipe delta is '
        '`kinetic_transport_prior_only: false→true`, with local weight1, sliced weight1/32 directions, tail weight0 and backtracking off. '
        'Both use the exact winning FullDualNorm/rate configuration; bare historical BCAP Adam is not the control.','',
        'All six matched final stream registries, initial models/prior, fixed task laws and resources are equal. '
        'Vector data replay verifies all 1200 target batches per arm and their final streams. '
        'All three frozen scalar/vector/native adapters reuse one actual real tensor for D/G. '
        'No extra forward/draw, target oracle, learned width/weight change or clean-to-noisy serving switch enters the candidate. '
        'The five retained local-v2 tasks exactly reproduce their original complete metric trajectories, final model/optimizer tensors and streams; '
        'native local-v2 is a new diagnostic cell, not archived qualification reuse.','',
        f"[Verification](software-verification.json) records **94 distinct checks**, **{verification['actual_training_gifs']} actual-training GIFs** "
        f"and **{verification['saved_scalar_vector_metric_sets_reproduced']} reproduced scalar/vector primary metric sets**. "
        '[Compatible scorer controls](scorer-controls-reuse.json) retain five oracle PASS/five collapse FAIL under their original identities. '
        'No generated model samples are added during publication; rescoring recreates the frozen scorer’s deterministic target-reference draws. '
        'The 15 tiny software outer updates are a separate fixture cohort, not quality evidence. '
        'Memory refreshes use summaries only and all prior qualification/inventory/telemetry blobs remain byte-identical.','',
        'Main executed full reservations are **19440 seconds**, within the **21600-second track ceiling** after '
        'the declared 600-second saved-analysis and 120-second software allowances. '
        'Saved and current static diagnostics add zero optimizer updates or generated sampling draws; their measured method times remain separate from worker costs. '
        'The two initial diagnostic refusals occurred before updates and have no instrumented method timing; they remain inside the shared conservative allowance. '
        'GPU contention is accounting, not speed superiority. No retry, active subscription reservation, worker or monitor remains. '
        'Bulk stdout/JSONL/JUnit/checkpoints/tensor dumps remain on the artifact drive.','',
        '| Goal | Local-v2 actual training | Prior-only actual training |','| --- | --- | --- |']
    for task in order:lines.append(f'| {task} | [GIF](control-{task}.gif) | [GIF](candidate-{task}.gif) |')
    lines += ['', '[Media input receipts](media.json) certify the saved observations. Reproduce reporting without training:', '',
        '```sh',
        'PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/component_tails/round5/publish.py',
        'PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/component_tails/round5/analyze.py',
        'PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python -m experiments.forge compile --summaries-only',
        'PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/component_tails/round5/verify.py',
        '```','', 'The frozen declarations below retain the pre-run predictions and stopping rules.','',
        '## Preregistered rationale and protocol','']
    existing=(OUT/'README.md').read_text()
    marker='## Preregistered rationale and protocol\n'
    if marker in existing:original=existing.split(marker,1)[1].lstrip()
    else:original=existing.split('\n',1)[1].lstrip()
    (OUT/'README.md').write_text('\n'.join(lines)+'\n'+original)
    print(json.dumps({'event':'readout_rendered','outcomes':{a:d['outcomes'] for a,d in arms.items()}}))


if __name__=='__main__':main()
