"""Publish capped-prior saved samples with the existing media/scoring host."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.toy_audit import gaussian_prior_magnitude as study

spec = importlib.util.spec_from_file_location("_capped_prior_publication_host",
    ROOT / "reports/forge/bcap-past-extrapolation/publish.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)
publication.PROTOCOL = study.PROTOCOL
publication.declaration = study.declaration
publication.DEST = Path(__file__).resolve().parent

if __name__ == "__main__":
    publication.main()
