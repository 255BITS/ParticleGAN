"""Print compact tables from the jsonl files: python summarize.py [diagnose|arms|crosscheck]."""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = ["particles", "core_eig", "minor_major", "out_rms_over_sigma", "linear_out_rms_over_sigma", "z_rare_rms",
         "z_rare_min_pair", "jac_op", "jac_op_big", "rare_spill", "D_real_rare", "D_fake_rare", "gradD_real_rare",
         "gradD_fake_rare", "inward_pull_real", "D_center_minus_1sigma_ring", "D_real_big"]


def diagnose():
    print("median over the last 8 observations; zero_obs = observations with no particle on the rare component")
    print("| arm | zero_obs | " + " | ".join(PROBE) + " |")
    for line in open(HERE / "diagnose.jsonl"):
        r = json.loads(line)
        obs = r["observations"]
        cells = []
        for k in PROBE:
            v = [o[k] for o in obs[-8:] if isinstance(o.get(k), (int, float))]
            cells.append(f"{statistics.median(v):.3g}" if v else "-")
        print(f"| {r['arm']} | {sum(o['particles'] == 0 for o in obs)} | " + " | ".join(cells) + " |")


def arms():
    rows = [json.loads(line) for line in open(HERE / "arms.jsonl")]
    rows.sort(key=lambda r: (-r["sustained"], -r["passing_suffix"], len(r["failing"]), r["live"]["sw1_normalized"]))
    print("| arm | kind | verdict (suffix) | sw1 | mass_tv | core eig (min) | rare core eig | min_mass_ratio | spill | rare mass | failing |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        v = r["live"]
        print(f"| {r['arm']} | {r['kind']} | {'SUST' if r['sustained'] else 'fail'} ({r['passing_suffix']}) | "
              f"{v['sw1_normalized']:.3f} | {v['mass_tv']:.3f} | {v['component_core_min_eigen_ratio']:.2f} | "
              f"{v['component_core_eigen_ratios'][3]:.2f} | {v['min_mass_ratio']:.2f} | {v['max_component_spill']:.3f} | "
              f"{v['component_mass'][3]:.3f} | {', '.join(f.replace('component_', '') for f in r['failing']) or '-'} |")


def crosscheck():
    rows = [json.loads(line) for line in open(HERE / "crosscheck.jsonl")]
    arms = sorted({r["arm"] for r in rows})
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    by = {(r["task"], r["arm"]): r for r in rows}
    print("| task | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    for t in tasks:
        cells = []
        for a in arms:
            r = by.get((t, a))
            if r is None or r["status"] == "ERROR":
                cells.append("ERROR" if r else "-")
                continue
            fail = ", ".join(f.replace("component_", "") for f in r["failing"]) or "-"
            cells.append(f"{'SUST' if r['sustained'] else 'fail'} ({r['passing_suffix']}) sw1 {r['live']['sw1_normalized']:.3f}; {fail}")
        print(f"| {t} | " + " | ".join(cells) + " |")
    print("| **passes** | " + " | ".join(str(sum(by[(t, a)].get("sustained", False) for t in tasks if (t, a) in by)) + "/" + str(len(tasks)) for a in arms) + " |")


if __name__ == "__main__":
    {"diagnose": diagnose, "arms": arms, "crosscheck": crosscheck}[sys.argv[1] if len(sys.argv) > 1 else "arms"]()
