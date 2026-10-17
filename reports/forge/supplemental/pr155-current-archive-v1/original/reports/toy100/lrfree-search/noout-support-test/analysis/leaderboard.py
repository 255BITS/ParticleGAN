"""Leaderboard rows from run directories: verdict, last-5 worst terminal metrics, holdout numbers and the S2 margin (holdout-deconvolved centre).
usage: python leaderboard.py LABEL=RUN_DIR ...   (runs/<label>-<task>)"""
import sys, json, math
def row(name, run):
    try:
        v = json.load(open(f'{run}/native-noisy/verdict.json'))['accuracy']
        res = json.load(open(f'{run}/result.json'))
    except Exception as e:
        return f'{name:22s} (not finished: {e.__class__.__name__})'
    term = v['terminal_checks'][-5:]
    m = [t['metrics'] for t in term]
    prec = min(x['precision'] for x in m); ctr = max(x['center_rms_sigma'] for x in m); tr = max(x['abs_cov_trace_bias'] for x in m); ks = max(x['radial_ks'] for x in m)
    h = v['holdout_metrics']; xh = math.sqrt(max(0., h['center_rms_sigma'] ** 2 - .0435 ** 2))
    return (f"{name:22s} {res['status']:4s} checks {res['passing_checks']:2d}/{res['observations']} streak {res['final_streak']:2d} | last5 worst prec {prec:.4f} ctr {ctr:.3f} |tr| {tr:.3f} ks {ks:.3f} | "
            f"holdout prec {h['precision']:.4f} ctr {h['center_rms_sigma']:.3f} (x_hat {xh:.3f}) ks {h['radial_ks']:.3f} tv {h['mass_tv']:.3f}")
for a in sys.argv[1:]:
    n, r = a.split('=', 1); print(row(n, r))
