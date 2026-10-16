"""Per-cell rerun table: verdict, per-update time vs the earlier run, and GPU co-tenancy.

Reads only saved files. Earlier timed-out runs have no timestamps, so their per-update time is
bounded: between 900 / (last_logged_step + interval) (lower) and 900 / last_logged_step (upper).
Usage: analyze.py STATE_ROOT > table.json
"""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/ml2/hypergan/dig2-20261016/wallclock-reruns"
FILL = "/ml2/hypergan/attempt-fill-20261015/attempts"
UUID = {"GPU-72c1b506-891d-b8bc-b353-e020585e1c47": 0, "GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69": 1}
# Long-lived residents seen before any rerun started: nano-work-server, desktop portal, nautilus,
# zenity (GPU 0) and llama-server (GPU 1).
RESIDENTS = {"34929", "12273", "2415818", "4024976", "3406406"}
# Retained (non-fill) earlier evidence, read from the PR399 receipts in reports/forge/test-performance-evidence.
RETAINED = {
    "bcap-default-baseline-direction-v1": {"seconds": 900.239, "last_step": 19168, "interval": 833,
                                           "evidence": "pr399-retained-pr399-paired-tier1-12.json (INCOMPLETE, raw timeout)"},
    "generator-confidence-lr-bcap-direction-v1": {"seconds": 857.132, "completed": 20001,
                                                  "evidence": "pr399-confidence-lr-tier1-6.json (PASS, queue_seconds)"},
}


def observations(path, stamped):
    out = []
    for line in open(path, errors="replace"):
        t = None
        if stamped and "\t" in line:
            head, line = line.split("\t", 1)
            t = json.loads(head)["t"]
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") == "observation" and "step" in event:
            out.append((t, int(event["step"]), event))
    return out


rows = []
for spec_path in sorted(glob.glob(f"{ROOT}/specs/*.json")):
    name = os.path.basename(spec_path)[:-5]
    spec = json.load(open(spec_path))
    directory = f"{ROOT}/attempts/{name}"
    row = {"cell": name, "recipe": spec["candidate_id"], "task": spec["task"], "limit": spec["timeout_seconds"],
           "earlier_status": spec["rerun_of"]["status"], "earlier_attempt": spec["rerun_of"]["attempt_id"],
           "key": spec["rerun_of"]["key"]}
    # Earlier run.
    earlier = f"{FILL}/{spec['rerun_of']['attempt_id']}/run.log"
    if os.path.exists(earlier):
        obs = observations(earlier, stamped=False)
        steps = [s for _, s, _ in obs]
        interval = (steps[-1] - steps[-2]) if len(steps) > 1 else steps[-1] if steps else None
        seconds = spec["rerun_of"]["seconds"]
        row.update(earlier_seconds=seconds, earlier_last_step=steps[-1] if steps else 0,
                   earlier_spu_low=(seconds / (steps[-1] + interval)) if steps else None,
                   earlier_spu_high=(seconds / steps[-1]) if steps else None, earlier_evidence=earlier)
    elif spec["candidate_id"] in RETAINED:
        r = RETAINED[spec["candidate_id"]]
        if "completed" in r:
            row.update(earlier_seconds=r["seconds"], earlier_last_step=r["completed"],
                       earlier_spu_low=r["seconds"] / r["completed"], earlier_spu_high=r["seconds"] / r["completed"])
        else:
            row.update(earlier_seconds=r["seconds"], earlier_last_step=r["last_step"],
                       earlier_spu_low=r["seconds"] / (r["last_step"] + r["interval"]),
                       earlier_spu_high=r["seconds"] / r["last_step"])
        row["earlier_evidence"] = r["evidence"]
    # Rerun.
    if os.path.exists(f"{directory}/result.json"):
        result = json.load(open(f"{directory}/result.json"))
        obs = observations(f"{directory}/run.log", stamped=True)
        lease = [json.loads(l) for l in open(f"{directory}/lease.json")]
        start, end = lease[0]["start"], lease[-1].get("end")
        gpu = lease[0]["leased_gpu"]
        row.update(status=result["status"], reason=result.get("reason"), updates=result.get("updates"),
                   worker_seconds=result["seconds"], lease_seconds=(end - start) if end else None, gpu=gpu,
                   last_step=obs[-1][1] if obs else 0, observations=len(obs),
                   confirmed_checks=(result.get("grade") or {}).get("evaluator_result", {}).get("confirmed_checks"))
        if len(obs) > 1:
            row["solo_spu"] = (obs[-1][0] - obs[0][0]) / (obs[-1][1] - obs[0][1])
        # Co-tenancy on the leased GPU.
        utils, others = [], set()
        for line in open(f"{directory}/gpu-samples.log"):
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            m = re.search(rf"(?:^|;){gpu}, (\d+) %", parts[1])
            if m:
                utils.append(int(m.group(1)))
            for pid, uuid, mem in re.findall(r"(\d+), (GPU-[0-9a-f-]+), (\d+) MiB", parts[1]):
                if UUID.get(uuid) == gpu and pid not in RESIDENTS:
                    others.add(pid)
        row.update(gpu_util_mean=sum(utils) / len(utils) if utils else None,
                   nonresident_pids_on_gpu=len(others))
        if row.get("solo_spu") and row.get("earlier_spu_low"):
            row["slowdown_low"] = row["earlier_spu_low"] / row["solo_spu"]
            row["slowdown_high"] = row["earlier_spu_high"] / row["solo_spu"]
    rows.append(row)
json.dump(rows, sys.stdout, indent=1, sort_keys=True)
print()
