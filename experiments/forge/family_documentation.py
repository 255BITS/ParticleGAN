"""Maintained, descriptive family metadata; never a scientific grading input."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from urllib.parse import urlsplit

from .contracts import identifier, read_json


DOCUMENTATION_DIRECTORY = Path("configs/forge/family-documentation")
TAG_VOCABULARY = Path("configs/forge/family-tags.json")
FAMILY_REGISTRY = Path("configs/forge/trainer-families.json")
TRAINING_DETAILS = (
    ("loss", "Adversarial loss"),
    ("optimizer", "Optimizer"),
    ("learning_rates", "Learning rates and annealing"),
    ("gradient_clipping", "Parameter-gradient clipping"),
    ("critic_penalty", "Critic penalties and anchors"),
    ("damping_and_guards", "Damping and update guards"),
    ("noise", "Training and sampling noise"),
    ("averaging", "Parameter averaging and serving"),
)
_TAG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_PINNED_GITHUB_SOURCE = re.compile(r"/[^/]+/[^/]+/blob/[0-9a-f]{40}/.+")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _texts(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    for item in value:
        _text(item, label)
    return value


def _https_url(value, label):
    _text(value, label)
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            or any(character.isspace() for character in value)):
        raise ValueError(f"{label} must be an HTTPS URL")
    return parsed


def validate_documentation(root: Path, value: dict, family_id: str) -> dict:
    """Validate maintained prose and resolvable source navigation, not its claims."""
    if not isinstance(value, dict):
        raise ValueError(f"family documentation for {family_id} must be an object")
    required = {"overview", "symbols", "pseudocode", "training_details", "configuration_notes",
                "references", "sources"}
    if set(value) != required:
        raise ValueError(f"family documentation for {family_id} needs exactly {sorted(required)}")
    _text(value["overview"], "technique overview")
    _text(value["symbols"], "pseudocode symbols")
    _texts(value["pseudocode"], "pseudocode")
    if any("```" in line for line in value["pseudocode"]):
        raise ValueError("pseudocode cannot contain Markdown code fences")
    _texts(value["configuration_notes"], "configuration notes")
    details = value["training_details"]
    if not isinstance(details, dict) or set(details) != {key for key, _ in TRAINING_DETAILS}:
        raise ValueError("training details need every declared training characteristic")
    for key, _ in TRAINING_DETAILS:
        _text(details[key], "training detail " + key)
    references = value["references"]
    if not isinstance(references, list):
        raise ValueError("references must be a list; use an empty list for repository-specific techniques")
    urls = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"title", "url", "component"}:
            raise ValueError("references need a title, URL and supported component")
        _text(reference["title"], "reference title")
        _text(reference["component"], "reference component")
        _https_url(reference["url"], "reference URL")
        urls.append(reference["url"])
    if len(urls) != len(set(urls)):
        raise ValueError("references must not repeat a URL")
    for source in _texts(value["sources"], "implementation sources"):
        if "://" in source:
            parsed = _https_url(source, "implementation source")
            if (parsed.netloc != "github.com" or not _PINNED_GITHUB_SOURCE.fullmatch(parsed.path)
                    or parsed.query):
                raise ValueError("historical sources must use a commit-pinned GitHub blob URL")
        else:
            path = Path(source)
            resolved = (root / path).resolve()
            if (path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve())
                    or not resolved.is_file()):
                raise ValueError(f"implementation source must be an existing repository file: {source}")
    if len(value["sources"]) != len(set(value["sources"])):
        raise ValueError("implementation sources must not repeat a path")
    return deepcopy(value)


def load_family_documentation(root: Path, family_ids, load) -> tuple[dict, dict]:
    """Load current and historical reports through the caller's provenance loader.

    Small historical/synthetic report roots predate editorial documentation. If
    the directory is absent, render an explicit unavailable notice. Once the
    directory exists, incomplete coverage is an error rather than a silent gap.
    """
    root = Path(root)
    if not (root / DOCUMENTATION_DIRECTORY).is_dir():
        return {}, {}
    if not (root / FAMILY_REGISTRY).is_file() or not (root / TAG_VOCABULARY).is_file():
        raise ValueError("family documentation needs its family registry and tag vocabulary")
    vocabulary = load(TAG_VOCABULARY)
    definitions = vocabulary.get("tags") if isinstance(vocabulary, dict) else None
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("family tag vocabulary needs nonempty tag definitions")
    for tag, definition in definitions.items():
        if not isinstance(tag, str) or not _TAG.fullmatch(tag):
            raise ValueError("family tags must be lowercase hyphenated names")
        _text(definition, "family tag definition")
    registry = load(FAMILY_REGISTRY)
    registered = {}
    for family in registry.get("families", []) + registry.get("historical_families", []):
        family_id = identifier(family.get("id"), "documented trainer family")
        if family_id in registered:
            raise ValueError("documented trainer family ids must be distinct")
        tags = family.get("tags")
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"family {family_id} needs a nonempty tag list")
        if len(tags) != len(set(tags)) or tags != sorted(tags):
            raise ValueError(f"family {family_id} tags must be unique and lexically sorted")
        if any(not _TAG.fullmatch(tag) for tag in tags):
            raise ValueError("family tags must be lowercase hyphenated names")
        if set(tags) - definitions.keys():
            raise ValueError(f"family {family_id} has undefined tags: {sorted(set(tags) - definitions.keys())}")
        registered[family_id] = family
    documentation = {}
    for family_id in family_ids:
        identifier(family_id, "documented trainer family")
        path = DOCUMENTATION_DIRECTORY / (family_id + ".json")
        if family_id not in registered or not (root / path).is_file():
            raise ValueError(f"missing registered family documentation for {family_id}")
        documentation[family_id] = {
            "tags": list(registered[family_id]["tags"]),
            "documentation": validate_documentation(root, load(path), family_id),
        }
    return documentation, dict(sorted(definitions.items()))


def validate_family_documentation(root: Path) -> dict:
    """Validate all registered editorial metadata without rendering or training."""
    root = Path(root)
    if not (root / DOCUMENTATION_DIRECTORY).is_dir():
        return {"status": "unavailable", "families": 0, "tags": 0}
    if not (root / FAMILY_REGISTRY).is_file():
        raise ValueError("family documentation needs its family registry and tag vocabulary")
    registry = read_json(root / FAMILY_REGISTRY)
    families = [family["id"] for family in registry.get("families", []) + registry.get("historical_families", [])]
    documentation, tags = load_family_documentation(root, families, lambda path: read_json(root / path))
    return {"status": "valid", "families": len(documentation), "tags": len(tags)}
