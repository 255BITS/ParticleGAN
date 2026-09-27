"""Old-init (runs_oldinit/) vs new-init (runs/) comparison for INIT_RERUN.md.

Usage: python compare_init.py            prints markdown: new leaderboard, delta table, rank changes
Scoring and ranking are summarize.py's. Reference rows (REF formulations, ref:ka2-constant) are not ranked;
their place "ref@N" is where they would sit in the same ordering (just above ranked arm N).
"""
import summarize as S

KEYS = lambda r: (r["arrival"] is None, r["fails"], r["departures"],
                  r["arrival"] if r["arrival"] is not None else 1e9)


def load(dirname):
    S.RUNS = (S.HERE / dirname).resolve()
    rows = S.load_arms()
    ref = S.load_reference()
    if ref:
        rows.append(ref)
    return rows


def places(rows):
    out, rank = {}, 0
    for r in sorted(rows, key=KEYS):
        if r["status"] == "reference":
            out[r["arm"]] = f"ref@{rank + 1}"
        else:
            rank += 1
            out[r["arm"]] = str(rank)
    return out


def d(new, old, spec="{:+d}"):
    if new is None or old is None:
        return "n/a" if new is None and old is None else f"{S.fmt(old)} -> {S.fmt(new)}"
    return spec.format(new - old)


def main():
    new, old = load("runs"), load("runs_oldinit")
    board = S.table([dict(r) for r in new])
    pn, po = places(new), places(old)
    on = {r["arm"]: r for r in old}
    lines = ["| arm | place old -> new | prehold old -> new | arrival old -> new | departures old -> new "
             "| fails outside transit old -> new (Δ) | max grad-norm old -> new | final HQ old -> new |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(new, key=KEYS):
        o = on.get(r["arm"])
        if o is None:
            continue
        g = lambda x: S.fmt(x["gmax"], "{:.2f}", "n/a")
        lines.append("| " + " | ".join([
            r["arm"], f"{po[r['arm']]} -> {pn[r['arm']]}",
            f"{o['pre_pass']} -> {r['pre_pass']}",
            f"{S.fmt(o['arrival'], '{}', 'none')} -> {S.fmt(r['arrival'], '{}', 'none')}",
            f"{o['departures']} -> {r['departures']}",
            f"{o['fails']} -> {r['fails']} ({r['fails'] - o['fails']:+d})",
            f"{g(o)} -> {g(r)}", f"{o['final_hq']:.3f} -> {r['final_hq']:.3f}"]) + " |")
    moves = []
    for a in pn:
        if a in po and pn[a].isdigit() and po[a].isdigit():
            moves.append((int(po[a]) - int(pn[a]), a, po[a], pn[a]))
    moves.sort()
    print("## New-init leaderboard\n\n" + board + "\n\n## Old vs new (every arm, new-init order)\n\n"
          + "\n".join(lines) + "\n\n## Rank changes (positive = moved up)\n\n"
          + "\n".join(f"- {a}: {o} -> {n} ({m:+d})" for m, a, o, n in sorted(moves, key=lambda t: -abs(t[0]))))


if __name__ == "__main__":
    main()
