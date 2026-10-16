"""Write the PROPOSED discriminator_stability revision 10 (ring16 at 6,400 updates). Not activated.

    python reports/forge/dig2-20261016/ring16-cap-make-view-proposal.py

The file goes to configs/forge/view-proposals/, which no loader reads, so the live view (revision 9),
the publication and every recorded score are unchanged. To adopt it, the user copies revision 9 to
configs/forge/view-history/discriminator_stability-v9.json, moves this file to
configs/forge/views/discriminator_stability.json, and updates the tests that pin revision 9.
"""
import json
from copy import deepcopy
from pathlib import Path

live = json.loads(Path("configs/forge/views/discriminator_stability.json").read_text())
assert live["revision"] == 9
view = deepcopy(live)
view["revision"] = 10
hits = [a for a in view["assignments"] if a["task"] == "ring16_acquisition"]
assert len(hits) == 1 and hits[0]["qualification_tier"] == 1 and hits[0]["importance"] == "required"
hits[0]["task"] = "ring16_acquisition_6400"
view["policy_change_reason"] = (
    "User-authorised ring16 cap change: the required Tier 1 ring16 question becomes ring16_acquisition_6400 "
    "(6400 updates, each recipe's own schedule over the full cap, unchanged gate, same 16.67-update check "
    "cadence). ring16_acquisition (1600 updates) and every receipt graded under it keep their identity; "
    "revision 9 remains in view history. Evidence: reports/forge/dig2-20261016/ring16-cap.md.")
out = Path("configs/forge/view-proposals/discriminator_stability-v10-ring16-6400.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(view, indent=2) + "\n")
print(out)
