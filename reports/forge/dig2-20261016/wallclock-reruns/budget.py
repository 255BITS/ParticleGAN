"""GPU-seconds charged to this lead: lease seconds of finished cells plus declared limits of claimed, unfinished ones.

Usage: budget.py ROOT [ROOT ...]  (prints an integer)
"""
import glob
import json
import os
import sys

total = 0.0
for root in sys.argv[1:]:
    for directory in glob.glob(f"{root}/attempts/*"):
        spec = json.load(open(f"{root}/specs/{os.path.basename(directory)}.json"))
        lease = f"{directory}/lease.json"
        if os.path.exists(f"{directory}/result.json") and os.path.exists(lease):
            rows = [json.loads(line) for line in open(lease)]
            if "end" in rows[-1]:
                total += rows[-1]["end"] - rows[0]["start"]
                continue
        total += spec["timeout_seconds"]
print(int(total + 0.999))
