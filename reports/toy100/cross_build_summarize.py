"""Read cross-build gate receipts into one table. No training."""

import json
import math
from pathlib import Path
import sys


def _load(path):
    return json.loads(Path(path).read_text())


def ring_counts(mode_hold):
    obs = mode_hold["result"]["observations"]
    steps = sorted({math.ceil(i * 1200 / 24) for i in range(1, 25)})
    by = {row["step"]: row for row in obs}
    rows = [by[step] for step in steps]
    ok = [row["modes"] == 8 and row["hq"] >= .9 for row in rows]
    suffix = 0
    for flag in reversed(ok):
        if not flag:
            break
        suffix += 1
    final = rows[-1]
    return dict(passing=f"{sum(ok)}/24", suffix=suffix,
                final_modes=final["modes"], final_hq=round(final["hq"], 4),
                min_modes=min(row["modes"] for row in rows))


def stay_suffix(payload):
    late = [row for row in payload.get("diagnostic") or [] if row["step"] > 1200]
    suffix = 0
    for row in reversed(late):
        if row["modes"] == 8 and row["hq"] >= .9:
            suffix += 1
        else:
            break
    return suffix


def first_divergence(traces):
    """traces: list of (label, path). Returns the first INIT or STATE mismatch."""
    loaded = []
    for label, path in traces:
        rows = []
        for line in Path(path).read_text().splitlines():
            if line:
                rows.append(json.loads(line))
        loaded.append((label, rows))
    if len(loaded) < 2:
        return None
    keys = ("z", "g", "d", "d_loss", "g_loss")
    limit = min(len(rows) for _, rows in loaded)
    for index in range(limit):
        base = loaded[0][1][index]
        for label, rows in loaded[1:]:
            other = rows[index]
            if (base.get("event"), base.get("run"), base.get("step")) != (
                    other.get("event"), other.get("run"), other.get("step")):
                return dict(index=index, reason="event alignment",
                            left=loaded[0][0], right=label,
                            left_row={k: base.get(k) for k in ("event", "run", "step")},
                            right_row={k: other.get(k) for k in ("event", "run", "step")})
            differ = [key for key in keys if base.get(key) != other.get(key)]
            if differ:
                return dict(index=index, event=base.get("event"), run=base.get("run"),
                            step=base.get("step"), fields=differ,
                            left=loaded[0][0], right=label,
                            left_loss=(base.get("d_loss"), base.get("g_loss")),
                            right_loss=(other.get("d_loss"), other.get("g_loss")),
                            left_z_sum=base.get("z_sum"), right_z_sum=other.get("z_sum"))
    return dict(index=None, matched_rows=limit, labels=[label for label, _ in loaded])


def first_op(paths):
    loaded = [(label, [json.loads(line) for line in Path(path).read_text().splitlines() if line])
              for label, path in paths]
    if len(loaded) < 2:
        return None
    limit = min(len(rows) for _, rows in loaded)
    for index in range(limit):
        base = loaded[0][1][index]
        for label, rows in loaded[1:]:
            other = rows[index]
            if base.get("op") != other.get("op") or _out_hash(base) != _out_hash(other):
                return dict(index=index, left=loaded[0][0], right=label,
                            left_op=base.get("op"), right_op=other.get("op"),
                            left_hash=_out_hash(base), right_hash=_out_hash(other),
                            left_sum=_out_sum(base), right_sum=_out_sum(other))
    if any(len(rows) != limit for _, rows in loaded):
        return dict(index=limit, reason="op-count mismatch",
                    counts={label: len(rows) for label, rows in loaded})
    return dict(index=None, matched_ops=limit)


def _out_hash(row):
    out = row.get("out")
    if isinstance(out, dict):
        return out.get("hash")
    return json.dumps(out, sort_keys=True)


def _out_sum(row):
    out = row.get("out")
    if isinstance(out, dict):
        return out.get("sum")
    return None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                "reports/toy100/continuous-evidence/cross-build-repro")
    methods = ("reachstall", "delayg05", "delayed_g125")
    builds = ("cu130", "cpu214", "cpu213")
    table = []
    for method in methods:
        for build in builds:
            base = root / build / method
            row = dict(method=method, build=build)
            decl = base / "cold" / "declaration.json"
            if decl.is_file():
                body = _load(decl)
                row["torch"] = body.get("torch")
                row["cpu"] = body.get("cpu")
            warm = base / "warm" / "summary.json"
            if warm.is_file():
                summary = _load(warm)
                for name, key in (("identity", "warm_identity"), (method, "warm_method")):
                    local = (summary.get(name) or {}).get("local") or {}
                    if local:
                        row[key] = f"{local.get('passing_checks')}/{local.get('checks')}"
                        row[key + "_min_hq"] = local.get("min_hq")
                        row[key + "_min_modes"] = local.get("min_modes")
            traj = base / "cold" / "trajectory.json"
            if traj.is_file():
                verdict = _load(traj)["verdict"]
                row["cold_traj"] = verdict.get("status")
            ring = base / "cold" / "mode_hold.json"
            if ring.is_file():
                row["cold_ring"] = ring_counts(_load(ring))
                row["cold_ring_verdict"] = _load(ring)["verdict"]["status"]
            stay = base / "stay" / "stay.json"
            if stay.is_file():
                summary = _load(stay)["summary"]
                row["stay"] = f"{summary.get('passing')}/{summary.get('checks')}"
                row["stay_final"] = summary.get("final")
                row["stay_min_modes"] = summary.get("min_modes")
                row["stay_min_hq"] = summary.get("min_hq")
                row["stay_suffix"] = stay_suffix(_load(stay))
                row["stay_failing"] = summary.get("failing_steps")
            table.append(row)
    divergences = {}
    for method in methods:
        cold = []
        stay = []
        for build in builds:
            cpath = root / build / method / "cold" / "state.jsonl"
            spath = root / build / method / "stay" / "state.jsonl"
            if cpath.is_file():
                cold.append((build, cpath))
            if spath.is_file():
                stay.append((build, spath))
        divergences[method] = dict(
            cold=first_divergence(cold) if cold else None,
            stay=first_divergence(stay) if stay else None,
        )
    print(json.dumps(dict(table=table, divergences=divergences), indent=2, default=float))


if __name__ == "__main__":
    main()
