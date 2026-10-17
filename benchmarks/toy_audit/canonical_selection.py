"""List canonical audit entries while retaining every named scientific control.

This metadata selection never runs a toy, rewrites a gate or reuses training.
Only the explicitly proved shared-law groups collapse in the default list.
Other cases remain separate without claiming their laws were compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports/toy_audit/canonical-selection.json"
LAW_KEYS = ("kind", "means", "covariances", "masses", "radius_min", "radius_max",
            "turns", "noise", "scale_start", "scale_end", "scale_ramp_end")
PROVED_GROUPS_SHA256 = "08f6417d1e861bfb4aa0a86520f9ef41565b94f64bfc47dae277b68e26686c01"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def law_fingerprint(case, proof, banks):
    spec = case.get("spec", {})
    if proof["kind"] == "image_templates":
        pattern = spec.get("pattern")
        if pattern not in banks:
            raise ValueError(f"unproved image law for {case['id']}: {pattern}")
        bank = banks[pattern]
        if spec.get("modes") != bank["modes"]:
            raise ValueError(f"different image mode count for {case['id']}")
        # Architecture, threshold and optimizer fields remain per-case controls.
        return digest(dict(bank_sha256=bank["sha256"], shape=bank["shape"], dtype="<f4",
                           masses=[1 / bank["modes"]] * bank["modes"],
                           conditioning="none", real_noise_std=spec.get("noise_std"),
                           real_noise="independent Gaussian, clipped to [0,1]"))
    if proof["kind"] == "vector_sampler":
        if spec.get("kind") not in ("gaussian_mixture", "spiral", "annulus"):
            raise ValueError(f"unproved vector law for {case['id']}")
        fields = {key: spec[key] for key in LAW_KEYS if key in spec}
        if spec.get("scale_start", 1.) != spec.get("scale_end", 1.):
            fields["steps"] = spec["steps"]  # target_scale uses the update budget.
        return digest(fields)
    raise ValueError(f"unsupported law proof: {proof['kind']}")


def validate(manifest, catalog):
    if manifest.get("format") != "toy_canonical_selection_v1":
        raise ValueError("unsupported canonical-selection format")
    cases = catalog["cases"]
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate catalog IDs")
    records = manifest["records"]
    retained = {row["catalog_id"]: row for row in records}
    if len(retained) != len(records):
        raise ValueError("duplicate retained IDs")
    if set(retained) != set(by_id):
        raise ValueError(f"unknown/missing retained IDs: unknown={sorted(set(retained) - set(by_id))}, "
                         f"missing={sorted(set(by_id) - set(retained))}")
    expected, representatives = {}, set()
    for group in manifest["groups"]:
        canonical = group["canonical_id"]
        members = group["members"]
        if canonical not in members or len(members) < 2 or len(set(members)) != len(members):
            raise ValueError("each alias group needs unique members and its representative")
        if canonical in representatives:
            raise ValueError("conflicting canonical representatives")
        representatives.add(canonical)
        for name in members:
            if name not in by_id:
                raise ValueError(f"unknown group ID: {name}")
            if name in expected:
                raise ValueError(f"conflicting aliases for {name}")
            if law_fingerprint(by_id[name], group["proof"], manifest["template_banks"]) != group["proof"]["law_sha256"]:
                raise ValueError(f"different or unproved law in alias group: {name}")
            expected[name] = canonical
    for name, row in retained.items():
        if row["canonical_id"] != expected.get(name, name):
            raise ValueError(f"conflicting or unproved alias for {name}")
        required_role = "keep" if row["canonical_id"] == name else "control"
        if row["retention"] != required_role:
            raise ValueError(f"invalid retention role for {name}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(f"missing scientific keep/control reason for {name}")
        if row["original_name"] != by_id[name]["name"]:
            raise ValueError(f"immutable catalog name changed for {name}")
        for field, original in (("original_rating", "rating"), ("original_status", "status"), ("original_media", "media")):
            if row[field] != by_id[name].get(original):
                raise ValueError(f"immutable catalog {original} changed for {name}")
    if digest({key: manifest[key] for key in ("groups", "template_banks", "law_source_sha256")}) != PROVED_GROUPS_SHA256:
        raise ValueError("registered law proofs changed; unproved aliases are not allowed")
    if digest(catalog) != manifest["catalog"]["content_sha256"]:
        raise ValueError("catalog content changed; review and bind a new selection")
    canonical_count = sum(row["catalog_id"] == row["canonical_id"] for row in records)
    counts = manifest["counts"]
    if (len(records), canonical_count, len(records) - canonical_count, len(manifest["groups"])) != (
            counts["original_cases"], counts["canonical_problem_entries"], counts["linked_control_cases"], counts["proved_shared_law_groups"]):
        raise ValueError("selection counts conflict with retained cases")
    return by_id, retained


def load(root=ROOT, manifest_path=MANIFEST):
    root, manifest_path = Path(root), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    catalog_path = root / manifest["catalog"]["path"]
    catalog = json.loads(catalog_path.read_text())
    validate(manifest, catalog)
    if hashlib.sha256(catalog_path.read_bytes()).hexdigest() != manifest["catalog"]["file_sha256"]:
        raise ValueError("catalog bytes changed; provenance binding is stale")
    for name, expected in manifest["law_source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"law source changed: {name}")
    return manifest, catalog


def list_cases(manifest, catalog, case_id=None):
    _, retained = validate(manifest, catalog)
    if case_id is not None and case_id not in retained:
        raise ValueError(f"unknown retained case ID: {case_id}")
    return [dict(row) for row in manifest["records"] if case_id is None or row["catalog_id"] == case_id]


def list_problems(manifest, catalog, problem_id=None):
    validate(manifest, catalog)
    groups = {group["canonical_id"]: group for group in manifest["groups"]}
    selected = []
    for row in manifest["records"]:
        name = row["catalog_id"]
        if row["canonical_id"] != name:
            continue
        linked = [item["catalog_id"] for item in manifest["records"] if item["canonical_id"] == name]
        group = groups.get(name)
        selected.append(dict(canonical_id=name, display_name=group["label"] if group else row["display_name"],
                             retained_cases=linked,
                             equivalence="proved shared law" if group else "separately retained; global equivalence unproved"))
    if problem_id is not None and problem_id not in {row["canonical_id"] for row in selected}:
        raise ValueError(f"unknown canonical problem ID: {problem_id}")
    return [row for row in selected if problem_id is None or row["canonical_id"] == problem_id]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("list", "cases"), default="list")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--id", help="Exact canonical problem ID or retained case ID for the selected view")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest, catalog = load(args.root, args.manifest)
        rows = (list_problems if args.command == "list" else list_cases)(manifest, catalog, args.id)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            if args.command == "list":
                print(f"{row['canonical_id']}\t{row['display_name']}\t" + ", ".join(row["retained_cases"]))
            else:
                print(f"{row['catalog_id']}\t{row['display_name']}\t{row['retention']}\t{row['canonical_id']}\t{row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
