"""Write the scoped research tasks for dig2 clock-audit-measured (reproduction source)."""
import json
from copy import deepcopy
from pathlib import Path

T = Path("configs/forge/tasks")

clock = json.loads((T / "clockfree_audit_measurement_v1.json").read_text())
v2 = deepcopy(clock)
v2["id"] = "clockfree_audit_measured_v2-clockdig"
v2["description"] = (
    "PROPOSED revision of clockfree_audit_measurement_v1 (needs user/reviewer sign-off; not a counted gate). "
    "Same probe and four measured comparisons. Measured parity decides: any differing comparison fails. "
    "The v1 declaration table is a pre-screen: a step-schedule entry is overridden when the probe had power "
    "to see a step/horizon-driven version of that schedule; a fixed internal counter entry is overridden "
    "under internal_counter_policy=measured when the counter is saved learning state. Continuous policies, "
    "extensions and lazy-penalty counters keep their declarations.")
v2["evaluation"]["audit_rule"] = "measured_parity_v2"
v2["evaluation"]["internal_counter_policy"] = "measured"
v2["evaluation"]["rule_status"] = "proposed_requires_sign_off"
v2["evaluation"]["previous_task"] = "clockfree_audit_measurement_v1"
(T / f"{v2['id']}.json").write_text(json.dumps(v2, indent=2, sort_keys=True) + "\n")

ring = json.loads((T / "ring16_acquisition.json").read_text())
r = deepcopy(ring)
r["id"] = "ring16_acquisition_6400-clockdig"
r["execution"]["steps"] = 6400
r["execution"]["host_definition"]["steps"] = 6400
r["execution"]["original_schedule_horizon"] = 6400
# Same evaluation cadence as the original (one observation per 16.67 updates).
r["evaluation"]["observations"] = 384
r["resources"]["timeout_seconds"] = 600
r["description"] = (
    "Scoped research variant of ring16_acquisition for dig2 clock-audit-measured (not the counted task and "
    "not the ring16-cap lead's revision): identical target, gates, sustained rule and evaluation cadence, "
    "with a 6,400-update cap and a 6,400-update task-owned schedule horizon (384 observations).")
r["research_artifacts"] = {**ring.get("research_artifacts", {}), "previous_task": "configs/forge/tasks/ring16_acquisition.json"}
(T / f"{r['id']}.json").write_text(json.dumps(r, indent=2) + "\n")
print(v2["id"], r["id"])
