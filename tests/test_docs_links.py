"""Root documentation links resolve and point newcomers to one current status page."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "RESEARCH_GLOSSARY.md", "EXPERIMENTATION.md", "FINDINGS.md")
LINK = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)\)")
INVENTORY = ROOT / "reports/forge/technique-inventory.md"
BANNER = "<!-- superseded-banner: generated -->"


def _anchors(path):
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    return {re.sub(r"[^\w\- ]", "", m.strip().lower()).replace(" ", "-")
            for m in re.findall(r"^#{1,6} (.*)$", text, re.M)}


def _links(path):
    return LINK.findall(re.sub(r"```.*?```", "", path.read_text(), flags=re.S))


@pytest.mark.parametrize("name", DOCS)
def test_root_doc_relative_links_and_anchors_resolve(name):
    path = ROOT / name
    broken = []
    for target in _links(path):
        if re.match(r"[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (path.parent / file_part) if file_part else path
        if not dest.exists() or (anchor and dest.suffix == ".md" and anchor not in _anchors(dest)):
            broken.append(target)
    assert not broken, broken


def test_glossary_expands_or_marks_every_heavily_used_acronym():
    glossary = (ROOT / "RESEARCH_GLOSSARY.md").read_text()
    for name in ("BISPF", "SBNP", "BCAP", "K3P", "KA2", "E22", "UCD", "DV12", "hq"):
        assert re.search(rf"^\| \*\*{name}\*\*", glossary, re.M), name


def test_superseded_banners_link_to_the_single_current_inventory():
    bannered = [p for p in (ROOT / "reports").rglob("*.md")
                if "attempts" not in p.parts and BANNER in p.read_text(errors="replace")]
    assert len(bannered) >= 10
    for path in bannered + [ROOT / "FINDINGS.md"]:
        targets = [(path.parent / t).resolve() for t in _links(path) if t.endswith("technique-inventory.md")]
        assert INVENTORY.resolve() in targets, path
