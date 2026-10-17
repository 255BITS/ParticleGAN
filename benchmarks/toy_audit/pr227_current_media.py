"""Animate PR227's qualified guided/rotated curves without loading any model.

This supplemental renderer verifies the original reviewers' entire file-hash
manifests. It reads already measured clean FAST held-out paired games; it never
trains, reconstructs a checkpoint, recomputes a metric, or substitutes a judge.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .render import COLORS, frame, plt, save


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "b370cb7a49429e67f106e41752abea078327c5d0"
RUN_ROOT = Path("/ml2/hypergan/ParticleGAN-convergence-toy-develop/runs")
ARMS = ("ordinary_native_game", "particle_native_game", "neutral_particle_native_game")
LABELS = ("ordinary", "original particles", "neutral particles")
JUDGES = tuple(f"{arm}@{step}" for arm in ARMS[:2] for step in (800, 6400))
STEPS = tuple(sorted({0, 5120, *range(200, 6401, 200)}))
STATE_STEPS = tuple(sorted({802, *STEPS}))
PROTOCOLS = {
    "guided_pair": {
        "archive": "routed-convergence-guided-pair-v1",
        "schema": "routed_convergence_guided_pair",
        "title": "PR227 · guided pair (guidance 3)",
        "claim": "6400 result: original gap reproduced; H/b initialization support",
        "scientific_status": "INITIALIZATION_SUPPORT",
        "curve_sha256": "036d04da137bc14848f3046fc225b90483ba9cf26eddb291045611778fda04d9",
    },
    "rotated_teacher": {
        "archive": "routed-convergence-rotated-v1",
        "schema": "routed_convergence_rotated",
        "title": "PR227 · rotated teacher",
        "claim": "6400 result: original-gap reproduction FAIL; H/b support not established",
        "scientific_status": "GAP_REPRODUCTION_FAIL",
        "curve_sha256": "93d52f0ea36129e2814c258ec36574629ed7deab27a6196cf1732b14ade40752",
    },
}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=None)
def committed_bytes(relative):
    return subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)


def hash_map_digest(mapping):
    """The held reviewer's base.digest for a dictionary of string SHA values.

    That function hashes sorted keys followed by JSON-encoded values. No tensor
    deserialization, model import, reviewer execution or optimizer work occurs.
    """
    result = hashlib.sha256()
    for key in sorted(mapping):
        require(isinstance(key, str) and isinstance(mapping[key], str), "non-string SHA manifest")
        result.update(key.encode())
        result.update(json.dumps(mapping[key], sort_keys=True).encode())
    return result.hexdigest()


def file_manifest(directory, receipt):
    manifest = {}
    for relative in ("receipt.json", "data.pt", "common-judge-curves.jsonl"):
        manifest[relative] = sha(directory / relative)
    for relative, expected in receipt["source_archive"].items():
        path = directory / "source" / relative
        require(path.resolve().is_relative_to((directory / "source").resolve()), "source path escapes archive")
        actual = sha(path)
        require(actual == expected, f"archived source changed: {relative}")
        manifest["source/" + relative] = actual
    require(set(receipt["arms"]) == set(ARMS), "unexpected arm selection")
    for arm, record in receipt["arms"].items():
        require(record["status"] == "complete", f"incomplete arm: {arm}")
        states = record["checkpoints"]
        require(len(states) == len(STATE_STEPS) and {s["step"] for s in states} == set(STATE_STEPS),
                f"missing or duplicate states: {arm}")
        require({p.name for p in (directory / arm).glob("step-*.pt")} == {s["file"] for s in states},
                f"undeclared checkpoint: {arm}")
        for state in states:
            require(state["file"] == f"step-{state['step']:04d}.pt", f"noncanonical state file: {arm}")
            relative = arm + "/" + state["file"]
            actual = sha(directory / relative)
            require(actual == state["sha256"], f"checkpoint changed: {relative}")
            manifest[relative] = actual
        relative = arm + "/trace.jsonl"
        manifest[relative] = sha(directory / relative)
    return manifest


def curve_values(raw):
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    require(len(rows) == len(ARMS) * len(STEPS), "curve must contain all 102 observations")
    values, seen = {}, set()
    for row in rows:
        arm, step = row["arm"], row["step"]
        require(arm in ARMS and type(step) is int and step in STEPS, "unknown arm or checkpoint step")
        require((arm, step) not in seen, "duplicate curve observation")
        seen.add((arm, step))
        require(set(row["scores"]) == set(JUDGES), "missing or substituted common judge")
        for judge in JUDGES:
            value = row["scores"][judge]["test"]["paired_game"]
            require(type(value) in (float, int) and math.isfinite(value), "nonfinite or nonnumeric paired game")
            values[(arm, step, judge)] = value
    require(seen == {(arm, step) for arm in ARMS for step in STEPS}, "incomplete curve grid")
    return values


def validate(protocol, directory):
    directory = Path(directory).resolve()
    config = PROTOCOLS[protocol]
    stem = f"docs/e22_routed_convergence_{protocol}"
    docs = {kind: committed_bytes(stem + suffix) for kind, suffix in
            (("card", "_v1.json"), ("results", "_results.json"))}
    card, results = (json.loads(docs[key]) for key in ("card", "results"))
    receipt, review = (read(directory / name) for name in ("receipt.json", "independent-review.json"))
    input_hashes = {name: sha(directory / name) for name in
                    ("receipt.json", "independent-review.json", "independent-qualified-summary.json")}
    require(receipt["schema"] == config["schema"] + "_execution_v1" and receipt["status"] == "complete",
            "wrong or incomplete execution receipt")
    require(review["schema"] == config["schema"] + "_independent_review_v1"
            and review["status"] == "qualified" and review["qualified"] is True, "unqualified review")
    require(receipt["qualification_credit"] == review["qualification_credit"] == results["qualification_credit"] == "none",
            "standalone protocol qualification scope changed")
    require(receipt["contract"] == card and receipt["task"] == card["task_id"], "source/card law mismatch")
    require(receipt["runtime"]["device"] == "cpu" and receipt["runtime"]["threads"] == 1,
            "wrong original runtime cohort")
    require(card["execution"]["steps_per_arm"] == 6400 and card["evaluation"]["endpoints"] == [5120, 6400],
            "changed budget or endpoints")
    require(review["quality_updates"] == dict.fromkeys(ARMS, 6400), "incomplete original training budget")
    require(review["curve_counts"] == dict.fromkeys(ARMS, 34)
            and review["state_counts"] == dict.fromkeys(ARMS, 35), "review observation count changed")
    require(review["execution_receipt_sha256"] == input_hashes["receipt.json"], "review/receipt hash mismatch")
    require(results["review_sha256"] == input_hashes["independent-review.json"], "committed review hash mismatch")
    summary_key = "source_summary_sha256" if protocol == "guided_pair" else "summary_artifact_sha256"
    require(results[summary_key] == input_hashes["independent-qualified-summary.json"], "qualified summary hash mismatch")
    if protocol == "rotated_teacher":
        require(results["execution_receipt_sha256"] == input_hashes["receipt.json"], "committed receipt hash mismatch")
    source_hashes = receipt["bindings"]["source_hashes"]
    require(source_hashes == review["source_hashes"] == results["source_hashes"], "qualified source binding differs")
    for relative, expected in source_hashes.items():
        require(sha_bytes(committed_bytes(relative)) == expected, f"committed PR227 source changed: {relative}")
        require(receipt["source_archive"].get(relative) == expected, "missing proposal source snapshot")
    native = {name.removeprefix("native/"): value for name, value in receipt["source_archive"].items()
              if name.startswith("native/")}
    require(hash_map_digest(native) == receipt["bindings"]["native_source_hash"]
            == review["native_source_digest"] == results["native_source_digest"], "native source digest mismatch")
    require(receipt["data_digest"] == review["data_digest"] == results["data_digest"], "qualified data digest mismatch")
    require(receipt["judges"] == review["judges"] and set(review["judges"]) == set(JUDGES),
            "qualified critic identities differ")
    manifest = file_manifest(directory, receipt)
    require(manifest["data.pt"] == receipt["data_file_sha256"], "data file changed")
    require(hash_map_digest(manifest) == review["artifact_manifest_digest"], "qualified artifact manifest changed")
    if protocol == "rotated_teacher":
        require(review["artifact_manifest_digest"] == results["artifact_manifest_digest"], "committed manifest differs")
    require(manifest["common-judge-curves.jsonl"] == config["curve_sha256"], "qualified curve stream changed")
    if protocol == "guided_pair":
        docs["original_plot_receipt"] = committed_bytes(stem + "_plot.json")
        plot = json.loads(docs["original_plot_receipt"])
        for role, original_path in plot["input_paths"].items():
            name = Path(original_path).name
            actual = sha(directory / name)
            require(actual == plot["input_sha256"][role], f"original guided plot input changed: {name}")
            input_hashes[name] = actual
        execution_complete = read(directory / "execution-completion.json")
        review_complete = read(directory / "independent-review.json.completion.json")
        require(execution_complete["complete"] is True and review_complete["complete"] is True,
                "incomplete guided durable completion receipts")
        require(execution_complete["receipt_sha256"] == input_hashes["receipt.json"]
                and review_complete["review_sha256"] == input_hashes["independent-review.json"],
                "guided completion hash mismatch")
        require(plot["critics"] == review["judges"] and plot["data_digest"] == review["data_digest"],
                "original plot law differs")
        require(plot["checkpoint_steps"] == list(STEPS) and plot["pool"] == "test"
                and plot["metric"] == "paired_game" and plot["head"] == "clean FAST"
                and plot["smoothing"] is False, "original plot observation law differs")
    values = curve_values((directory / "common-judge-curves.jsonl").read_bytes())
    for arm in ARMS:
        for step in (5120, 6400):
            for judge in JUDGES:
                actual = values[(arm, step, judge)]
                for owner in (receipt, review, results):
                    expected = owner["endpoint_scores"][f"{arm}@{step}"][judge]["test"]["paired_game"]
                    require(math.isclose(actual, expected, abs_tol=2e-7, rel_tol=2e-7), "qualified endpoint differs")
    gates = review["gates"]
    require(gates == results["gates"], "committed scientific gates changed")
    guided = protocol == "guided_pair"
    require(gates["original_particle_gap_reproduced"] is guided and gates["H_b_support_gate"] is guided,
            "recorded guided/rotated scientific finding differs")
    input_hashes.update({name: manifest[name] for name in ("data.pt", "common-judge-curves.jsonl")})
    metadata = {
        "protocol": protocol, "task_id": receipt["task"], "archive": str(directory),
        "fresh_execution_status": "COMPLETE", "provenance_review_status": "QUALIFIED",
        "scientific_status": config["scientific_status"], "qualification_credit": "none",
        "input_sha256": input_hashes, "committed_docs_sha256": {name: sha_bytes(raw) for name, raw in docs.items()},
        "source_hashes": source_hashes,
        "native_bindings": {key: value for key, value in receipt["bindings"].items() if key != "source_hashes"},
        "source_archive_digest": hash_map_digest(receipt["source_archive"]),
        "verified_artifact_manifest_digest": hash_map_digest(manifest), "verified_artifact_files": len(manifest),
        "data_digest": review["data_digest"], "judges": review["judges"], "runtime": receipt["runtime"],
        "serving": "clean FAST model; original fixed private Gaussian critic-input panels; held-out test pool",
        "original_review_checks": review["checks"], "frame_steps": list(STEPS),
        "observations_per_arm": 34, "plotted_scalar_count": len(values), "interpolation": False,
        "excluded_software_replay_state_step": 802, "gates": gates,
        "endpoint_test_paired_game": {str(step): {judge: {arm: values[(arm, step, judge)] for arm in ARMS}
                                                 for judge in JUDGES} for step in (5120, 6400)},
        "accepted_structural_moves": {arm: receipt["arms"][arm]["coverage"]["moves"] for arm in ARMS},
    }
    return values, metadata, manifest


def render(protocol, values, path):
    config = PROTOCOLS[protocol]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), dpi=100)
    fig.subplots_adjust(left=.085, right=.975, bottom=.19, top=.765, hspace=.72, wspace=.24)
    limits = {}
    for judge in JUDGES:
        all_values = [values[(arm, step, judge)] for arm in ARMS for step in STEPS]
        lo, hi = min(all_values), max(all_values)
        pad = max(.06, .065 * (hi - lo))
        limits[judge] = [lo - pad, hi + pad]
    frames = []
    for t, step in enumerate(STEPS):
        for ax, judge in zip(axes.flat, JUDGES):
            ax.clear()
            for arm, label, color in zip(ARMS, LABELS, COLORS):
                x = STEPS[:t + 1]
                ax.plot(x, [values[(arm, s, judge)] for s in x], color=color,
                        marker="o", markersize=2.1, linewidth=1.35, label=label)
                ax.scatter([step], [values[(arm, step, judge)]], color=color, s=20, zorder=3)
            ax.axvline(5120, color="#999999", linestyle=":", linewidth=.8)
            ax.axvline(6400, color="#999999", linestyle=":", linewidth=.8)
            ax.set_xlim(-80, 6500)
            ax.set_ylim(*limits[judge])
            ax.set_xticks([0, 1600, 3200, 4800, 6400])
            ax.set_xlabel("actual training update")
            ax.set_ylabel("held-out paired game")
            critic_arm, critic_step = judge.split("@")
            title = "ordinary critic" if critic_arm == ARMS[0] else "original-particle critic"
            ax.set_title(f"{title} · fixed at update {critic_step}", fontsize=9)
            ax.grid(alpha=.2)
            ax.text(.02, -.42, " / ".join(f"{values[(arm, step, judge)]:.3f}" for arm in ARMS),
                    transform=ax.transAxes, fontsize=8, color="#444444")
        fig.suptitle(f"{config['title']} · actual checkpoint {step} / 6400", y=.975, fontsize=13)
        legend = fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center",
                            bbox_to_anchor=(.5, .925), ncol=3, frameon=False, fontsize=9)
        note = fig.text(.5, .846, config["claim"], ha="center", fontsize=10)
        footer = fig.text(.06, .024,
                          "Lower game is better within each fixed critic. Numbers follow legend order. Clean FAST; test pool.\n"
                          "34 actual observations; no interpolated measurements. Dotted updates: 5120 / 6400. No Forge credit.",
                          fontsize=8, linespacing=1.5)
        frames.append(frame(fig))
        legend.remove()
        note.remove()
        footer.remove()
    plt.close(fig)
    media = save(frames, path)
    media.update(poster_sha256=sha(path.with_suffix(".png")), fixed_y_limits=limits,
                 frame_steps=list(STEPS), width=frames[0].width, height=frames[0].height)
    return media


def write_report(directory, records):
    lines = [
        "# PR227 supplemental guided-pair and rotated-teacher training media", "",
        "These GIFs use every retained common-critic observation from the two expanded PR227 protocols. "
        "Each frame advances to an actual saved update; all three arms share the same four fixed critics. "
        "The plots show the already measured held-out paired native RpGAN game on clean FAST outputs "
        "under the original private critic-input panels. Lower is better within a fixed critic. "
        "Both archives passed their independent provenance review; that review does not make every scientific gate pass.", "",
        "| Protocol | Full-budget scientific result | Actual frames |", "|---|---|---|",
        "| Guided pair, guidance 3 | Original gap reproduced; H/b initialization support | 34, updates 0–6400 |",
        "| Rotated teacher | Original-gap reproduction FAIL; H/b support not established | 34, updates 0–6400 |", "",
        "## What the guided-pair problem verifies", "",
        "The fixed guided execution combines conditional and unconditional halves at guidance 3 while keeping "
        "the rotated teacher exactly reachable. It compares ordinary adapters, original routed particles, "
        "and particles with only fresh H/b initialization neutralized. At update 6400 original particles "
        "score worse than ordinary under all four critics; neutral particles score better than both. "
        "This supports initialization sensitivity in this specific fixture. At update 5120 original "
        "particles still beat ordinary under the two early critics, so the final finding does not describe every update.", "",
        "![Guided-pair actual training](guided_pair.gif)", "",
        "## What the rotated-teacher problem verifies", "",
        "The fixed teacher rotates the down span to a declared low-overlap basis while preserving its weight "
        "row Gram and exact reachability. It tests acquisition of that span and whether neutralizing H/b "
        "explains an original-particle disadvantage. That disadvantage is not reproduced consistently: "
        "the final ordinary and original-particle endpoint critics disagree. The failed reproduction and "
        "inapplicable gap-reduction support remain failed findings, even though neutral particles beat ordinary "
        "under all four critics at both endpoints.", "",
        "![Rotated-teacher actual training](rotated_teacher.gif)", "",
        "## Boundaries and reproduction", "",
        "Neither fixture isolates the unique cause of full Supra behavior. Absolute games across the two "
        "protocols use different target scales and critics and cannot be subtracted as a causal effect size. "
        "Weight-space overlap does not preserve activation covariance; the rotated toy's chance overlap "
        "also differs from the historical wide host. Both particle controls retain trainable banks, routers "
        "and nonzero code paths, but no structural moves were accepted. These standalone diagnostics grant "
        "no learned-MoG, Forge, robustness or default-promotion credit.", "",
        "The supplemental [receipt](receipt.json) pins PR227 source commit `" + SOURCE_COMMIT + "`, "
        "the committed cards/results, exact source archives, native source, data, critic identities, original "
        "execution/review hashes and both complete artifact manifests. All checkpoint files were hashed "
        "without deserialization. No training, replay update, model loading or metric recomputation occurred. "
        "The software-only 802 recovery states are excluded from these 34-point training curves. "
        "Existing audit catalog, media manifest and original PR227 GIF remain unchanged.", "",
        "```sh", "python -m benchmarks.toy_audit.pr227_current_media --validate-only",
        "python -m benchmarks.toy_audit.pr227_current_media", "```", "",
        "Default external archives:", "",
    ]
    for record in records:
        lines.append(f"- `{record['archive']}`")
    (directory / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guided-archive", type=Path, default=RUN_ROOT / PROTOCOLS["guided_pair"]["archive"])
    parser.add_argument("--rotated-archive", type=Path, default=RUN_ROOT / PROTOCOLS["rotated_teacher"]["archive"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/toy_audit/pr227_current_media")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for archive in (args.guided_archive, args.rotated_archive):
        require(not args.output.resolve().is_relative_to(archive.resolve()), "output must not alter original archive")
    protected = {relative: sha(ROOT / relative) for relative in
                 ("reports/toy_audit/media/media.json", "reports/toy_audit/media/pr227.gif", "reports/toy_audit/media/pr227.png")}
    records = []
    for protocol, archive in (("guided_pair", args.guided_archive), ("rotated_teacher", args.rotated_archive)):
        values, record, manifest = validate(protocol, archive)
        print(json.dumps({"protocol": protocol, "status": "qualified_hash_chain_verified",
                          "frames": len(STEPS), "artifact_files": len(manifest),
                          "scientific_status": record["scientific_status"]}), flush=True)
        if not args.validate_only:
            record["media"] = render(protocol, values, args.output / f"{protocol}.gif")
            require(file_manifest(archive, read(archive / "receipt.json")) == manifest, "renderer altered original evidence")
            for name, expected in record["input_sha256"].items():
                require(sha(archive / name) == expected, "renderer altered input receipt")
        records.append(record)
    require(all(sha(ROOT / relative) == expected for relative, expected in protected.items()), "original media changed")
    if args.validate_only:
        return
    receipt = {"schema": "toy_audit_pr227_current_supplemental_media_v1", "source_commit": SOURCE_COMMIT,
               "renderer_sha256": sha(Path(__file__)), "style_helper_sha256": sha(ROOT / "benchmarks/toy_audit/render.py"),
               "training_updates": 0, "software_replay_updates": 0, "model_states_loaded": 0, "metrics_recomputed": 0,
               "original_media_preserved_sha256": protected, "records": records}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    write_report(args.output, records)
    print(json.dumps({"receipt": str(args.output / "receipt.json"),
                      "gif_sha256": {r["protocol"]: r["media"]["sha256"] for r in records}}), flush=True)


if __name__ == "__main__":
    main()
