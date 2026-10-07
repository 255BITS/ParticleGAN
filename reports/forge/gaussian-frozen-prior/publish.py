"""Publish frozen-cohort saved observations using the existing media scorer."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.toy_audit.gaussian_frozen_prior import PROTOCOL, declaration
from experiments.forge.contracts import atomic_json, file_hash

source = ROOT / "reports/forge/bcap-past-extrapolation/publish.py"
spec = importlib.util.spec_from_file_location("past_media", source)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
publisher.PROTOCOL = PROTOCOL
publisher.declaration = declaration
publisher.DEST = Path(__file__).resolve().parent
original_render = publisher.render_gif


def render_control(case, frames, destination, **options):
    for frame in frames:
        for view in frame["views"]:
            view["title"] = "Frozen initial prior: " + view["title"]
            view["caption"] = "Fixed initial MoG; actual live GPU G/D outputs; all scheduled checks determine the grade."
    return original_render(case, frames, destination, **options)


publisher.render_gif = render_control

if __name__ == "__main__":
    publisher.main()
    path = publisher.DEST / "results.json"
    results = json.loads(path.read_text())
    results["cohort"] = "frozen_initial_prior"
    for row in results["results"]:
        row.update(cohort="frozen_initial_prior", prior=declaration()["prior"], prior_unchanged=True)
    atomic_json(path, results)
    path = publisher.DEST / "verification.json"
    verification = json.loads(path.read_text())
    verification["wrapper_source_sha256"] = file_hash(Path(__file__))
    atomic_json(path, verification)
