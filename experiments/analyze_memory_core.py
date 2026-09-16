"""Refresh the core formulation leaderboard from completed evaluations only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.analyze_memory_scout import coverage
from experiments.memory_orbit_metrics import panels, DEFINITION


def analyze(source, out, baselines=(), handoff_only=False):
    if (source.parent/'INVALIDATED.md').exists():
        raise ValueError(f'Refusing invalidated queue: {source.parent}')
    rows = []
    done = source.parent/'done'
    completed = {json.loads(p.read_text())['name'] for p in done.glob('*.json')} if done.exists() else None
    baseline_files = {Path(p)/'summary.json' for p in baselines}
    for path in [*sorted(source.glob('*/summary.json')), *sorted(baseline_files)]:
        if (path.parent.parent.parent/'INVALIDATED.md').exists():
            raise ValueError(f'Refusing invalidated baseline/run: {path.parent}')
        historical = path in baseline_files
        if not historical and completed is not None and path.parent.name not in completed:
            continue
        row = json.loads(path.read_text())
        if historical:
            row['comparison_only'] = True
        row['source'] = str(path.parent)
        with np.load(path.parent/'trajectories.npz') as arrays:
            row['coverage'] = coverage(arrays['generated'][:, :256], arrays['real_clean'])
            if all(k in arrays for k in ('prefix8', 'prefix32', 'continuation_reference')):
                row['orbit_progress'] = panels(arrays)
        rows.append(row)
    # Warm fidelity is primary for handoff; cold performance remains a separate axis.
    rows.sort(key=lambda r: (-min(r['metrics'][f'prefix{n}']['fidelity_long']['reference_orbit_fraction']
                                 for n in (8, 32)),
                            -r['metrics']['generated_long']['circle_like_fraction']))
    lines = ['# Core memory GAN scouts', '',
             'Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.',
             'Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,',
             'signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.',
             'This new composite is a diagnostic; see continuous errors and particle coverage in results.json.', '',
             '| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def pct(x):
        return f'{100*x:.1f}%'
    def pair(a,b):
        return f'{pct(a)} / {pct(b)}'
    for row in rows:
        m = row['metrics']
        short, long = m['generated_256'], m['generated_long']
        warm = [pair(m[f'prefix{n}']['fidelity_256']['reference_orbit_fraction'],
                     m[f'prefix{n}']['fidelity_long']['reference_orbit_fraction']) for n in (8,32)]
        lines.append(f"| {row['name']} | {row['steps']} | {pair(short['circle_like_fraction'], long['circle_like_fraction'])} | "
                     f"{short['passing_cw']} / {short['passing_ccw']} | {pct(long['late_stopped_fraction'])} | {' | '.join(warm)} |")
    progress_rows = [r for r in rows if 'orbit_progress' in r]
    if progress_rows:
        progress_rows.sort(key=lambda r: -min(r['orbit_progress'][f'prefix{n}']['quality'] for n in (8, 32)))
        lines += ['', '## Continuous orbit progress (evaluation only)', '', DEFINITION, '',
                  'Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.', '',
                  '| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in progress_rows:
            p = row['orbit_progress']
            a, b = p['prefix8'], p['prefix32']
            lines.append(f"| {row['name']} | {p['cold']['quality']:.4f} | "
                         f"{a['quality']:.4f} / {b['quality']:.4f} | "
                         f"{a['quality_last256']:.4f} / {b['quality_last256']:.4f} | "
                         f"{pair(a['good_step_fraction'], b['good_step_fraction'])} | "
                         f"{a['longest_good_arc_turns_mean']:.3f} / {b['longest_good_arc_turns_mean']:.3f} |")
    lines += ['', '## Continuation errors (1,024 generated points)', '',
              '| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for n in (8,32):
            m = row['metrics'][f'prefix{n}']['fidelity_long']
            lines.append(f"| {row['name']} | {n} | {m['relative_radial_rmse']:.3f} | "
                         f"{m['signed_speed_mae']:.3f} | {pct(m['reference_direction_consistency'])} | "
                         f"{m['startup_error_relative']:.3f} | {m['position_error_first32']:.3f} / {m['position_error_last128']:.3f} |")
    lines += ['', '## Interpretation and next-run candidates', '']
    scouts = [r for r in rows if not r['config'].get('evaluate_checkpoint') and not r.get('comparison_only')]
    if scouts:
        cold = max(scouts, key=lambda r: r['metrics']['generated_long']['circle_like_fraction'])
        warm = max(scouts, key=lambda r: min(r['metrics'][f'prefix{n}']['fidelity_long']['reference_orbit_fraction'] for n in (8,32)))
        if cold['metrics']['generated_long']['circle_like_fraction'] > 0:
            lines.append(f"- Highest cold long-horizon circle rate so far: **{cold['name']}**. Check both passing directions and coverage before promotion.")
        else:
            lines.append('- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.')
        best_warm = min(warm['metrics'][f'prefix{n}']['fidelity_long']['reference_orbit_fraction'] for n in (8,32))
        if best_warm:
            lines.append(f"- Highest worst-prefix long-horizon fidelity so far: **{warm['name']}** ({pct(best_warm)}).")
        else:
            lines.append('- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.')
        finished = ((source.parent/'SEALED').exists() and
                    not any((source.parent/'pending').glob('*.json')) and
                    not any((source.parent/'running').glob('*.json')))
        lines.append('- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.'
                     if finished else '- Candidate ranking is provisional until the queue finishes; no automatic promotion or scientific success claim.')
    else:
        lines.append('Only historical/control results are available; new training comparisons are pending.')
    if handoff_only:
        lines[0] = '# Local memory handoff scouts (no full-rollout training)'
        lines += ['', '## Training cost', '',
                  '| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in scouts:
            cfg = row['config']
            lines.append(f"| {row['name']} | {cfg['point_examples_per_update']} | {cfg['max_prefix']} | "
                         f"{cfg['memory_dim']} | {cfg.get('predict_weight', 0):g} / {cfg.get('temporal_weight', 0):g} | "
                         f"{cfg.get('generator_calls_d_phase', cfg.get('training_generator_unroll', 1))} / "
                         f"{cfg.get('generator_calls_g_phase', cfg.get('training_generator_unroll', 1))} | "
                         f"{cfg.get('feedback_probability', 0):g} / {cfg.get('feedback_strength', 1):g} | {row['seconds_per_update']:.4f} |")
        if any(r['config'].get('feedback_judge_memory') in ('clean', 'mixed') for r in scouts):
            lines += ['', '## Adversarial memory exploration', '',
                      '| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Legacy auxiliaries disabled |',
                      '|---|---|---|---|---:|---|']
            for row in scouts:
                c = row['config']
                lines.append(f"| {row['name']} | {c.get('feedback_judge_memory', 'shared')} | "
                             f"{c.get('g_memory_adapter', 'none')} | {c.get('feedback_backprop', False)} | "
                             f"{c.get('reader_calls_d_phase', '?')} / {c.get('reader_calls_g_phase', '?')} | "
                             f"{c.get('adversarial_only', False)} |")
            lines += ['', 'Clean judging: G reads the generated-write state, while both candidate scores',
                      'and B-cap use the same real-history memory, strictly before the target.',
                      'The proposal adapter uses two point-reader passes and stores no private state.',
                      'Reader calls include those internal passes; G calls count complete G evaluations.']
        if any(r['config'].get('local_pair_weight') or r['config'].get('feedback_judge_memory') == 'mixed'
               or r['config'].get('feedback_strength_distribution', 'fixed') != 'fixed' for r in scouts):
            lines += ['', '## Local transition and recovery scouts', '',
                      '| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |',
                      '|---|---:|---|---:|']
            for row in scouts:
                c = row['config']
                weight = c.get('feedback_shared_weight', .5) if c.get('feedback_judge_memory') == 'mixed' else float(c.get('feedback_judge_memory') == 'shared')
                lines.append(f"| {row['name']} | {weight:g} | {c.get('feedback_strength_distribution', 'fixed')} / "
                             f"{c.get('feedback_strength', 1):g} | {c.get('local_pair_weight', 0):g} |")
            lines += ['', 'Mixed judging averages separate clean/shared GAN losses and their default B-cap penalties.',
                      'The optional pair head judges two consecutive generated points against a real pair,',
                      'conditioned on real memory before both points. Its branch has one generated write',
                      'and is independent of point-loss exploration. Point/pair losses and penalties are',
                      'convexly weighted; the prior regularizer is applied once. No third generated point.',
                      'Uniform replacements range from zero to the configured maximum; mild_full selects',
                      'the mild value or one (default 25% full). Each update reuses its strengths in D and G.']
        if any(r['config'].get('slow_dim') or r['config'].get('g_memory_adapter', 'none') != 'none'
               or r['config'].get('stability_g_weight') or r['config'].get('stability_d_weight') for r in scouts):
            lines += ['', '## Local memory dynamics settings', '',
                      '| Run | Slow coordinates / rate | G adapter bottleneck | Repair target / weight / noise | Stability D / G weight / max gain |',
                      '|---|---:|---:|---|---|']
            for row in scouts:
                c = row['config']
                lines.append(f"| {row['name']} | {c.get('slow_dim', 0)} / {c.get('slow_rate', 1):g} | "
                             f"{c.get('adapter_bottleneck', 16) if c.get('g_memory_adapter', 'none') != 'none' else 'off'} | "
                             f"{c.get('repair_target', 'raw')} / {c.get('repair_weight', 0):g} / {c.get('repair_noise', .05):g} | "
                             f"{c.get('stability_d_weight', 0):g} / {c.get('stability_g_weight', 0):g} / {c.get('stability_max_gain', 1.1):g} |")
        if any(r['config'].get('g_state_dim', 0) for r in scouts):
            lines += ['', '## G-owned observation recurrence', '',
                      '| Run | G state size | State update reads D | G reads D | Prefix GRU updates / phase |',
                      '|---|---:|---|---|---:|']
            for row in scouts:
                c = row['config']
                lines.append(f"| {row['name']} | {c.get('g_state_dim', 0)} | {c.get('g_state_reads_d', False)} | "
                             f"{c.get('g_use_d_memory', True)} | {c.get('g_recurrence', {}).get('prefix_state_updates_per_phase', 0)} |")
            lines += ['', 'G state encodes real observations with full real-prefix BPTT, then at most one generated write.',
                      'State advances once per observation; proposal/final reads share the same state.',
                      'D owns M, G owns S. Both start at zero for cold evaluation. G has no MSE objective.',
                      'Memory access controls are separately trained; interventions alone do not establish comparative benefit.']
        if any(r['config'].get('local_objectives') for r in scouts):
            lines += ['', '## Local history and future objectives', '',
                      '| Run | Mismatch weight / donors | Recovery noise / probability | Future weight / offsets | Query bands |',
                      '|---|---|---|---|---:|']
            for row in scouts:
                c = row['config']
                lines.append(f"| {row['name']} | {c.get('mismatch_weight', 0):g} / {c.get('mismatch_kind', 'nearest')} | "
                             f"{c.get('recovery_noise', 0):g} / {c.get('recovery_probability', .5):g} | "
                             f"{c.get('future_weight', 0):g} / {c.get('future_offsets', [])} | {c.get('future_query_bands', 0)} |")
            lines += ['', 'Mismatch training ranks real continuations from other histories with the existing point head;',
                      'its D loss and default B-cap are normalized by 1+weight. G point loss stays unchanged.',
                      'Recovery perturbs prefix observations read by G; the pair judge retains the clean reference.',
                      'Future queries read identical prefix memory independently, with fixed z and explicit offsets.',
                      'Their joint GAN uses real future targets, no generated writes, and default exact B-cap.',
                      'Future weight convexly mixes this branch with the existing GAN; prior regularization stays once.']
        lines += ['', 'Scouts use local point GAN losses and optionally a two-point transition GAN. There is no full generated training',
                  'rollout or cold/warm path loss. Configured feedback adds at most one',
                  'generated write before each target; configs control G gradients through that write.',
                  'Longer rollouts are evaluation only.',
                  'Dense scouts use four points per episode; the older handoff_only trainer used one.',
                  'Architecture, context, corruption and optional D-only local auxiliary losses are explicit',
                  'config changes. Mismatched-history ranking changes the point head negative examples when enabled. Compare measured cost too.',
                  'Optional G repair trains a stateless read adapter. Local stability adds two parallel',
                  'one-step feedback branches per enabled phase, with detached prefix anchors and particles.',
                  'These local branches do not feed into another generated prediction; writer updates remain D-only.',
                  'No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.']
    else:
        lines += ['', 'The 10k control has more training than the 2k scouts. Conditional-head capacity and penalty domains',
              'differ from the old path-only control; active-loss weights are normalized. All new scouts share',
              'the same initialized modules, particles, data stream, optimizer defaults, and 10k schedule.',
              'All evaluation episodes are fixed across formulations; no seed sweeps.']
    lines += ['', 'Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).', '']
    out.mkdir(parents=True, exist_ok=True)
    # Readers never observe partially written JSON or Markdown.
    for name, content in [('results.json', json.dumps(rows, indent=2, allow_nan=False)+'\n'),
                          ('leaderboard.md', '\n'.join(lines))]:
        temporary = out/(name+'.tmp')
        temporary.write_text(content)
        temporary.replace(out/name)
    print(f'Reported {len(rows)} completed runs: {out}/leaderboard.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.out)


if __name__ == '__main__':
    main()
