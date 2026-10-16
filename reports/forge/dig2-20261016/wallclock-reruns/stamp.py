"""Prefix every stdin line with a wall-clock timestamp so per-update time can be measured from run.log."""
import json
import sys
import time

for line in sys.stdin:
    sys.stdout.write(json.dumps({"t": round(time.time(), 3)}) + "\t" + line)
    sys.stdout.flush()
