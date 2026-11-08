"""Exercise the actual lint gate without changing repository sources."""
from scripts.check_lint import PATHS, main


def _workspace(root):
    for name in PATHS:
        (root / name).mkdir()
    return root / "scripts" / "probe.py"


def test_lint_gate_rejects_new_findings_and_requires_resolved_exemptions_to_shrink(tmp_path, capsys):
    probe = _workspace(tmp_path)
    args = ["--root", str(tmp_path)]
    assert main([*args, "--update-baseline"]) == 0
    probe.write_text("answer = unknown_name\n")
    assert main(args) == 1
    assert "NEW:" in capsys.readouterr().out
    assert main([*args, "--update-baseline"]) == 0
    assert main(args) == 0
    probe.unlink()
    assert main(args) == 1
    assert "RESOLVED:" in capsys.readouterr().out
    assert main([*args, "--update-baseline"]) == 0
    assert main(args) == 0


def test_lint_baseline_survives_unrelated_line_moves(tmp_path):
    probe = _workspace(tmp_path)
    source = "from math import sin\nfrom math import sin\n"
    probe.write_text(source)
    args = ["--root", str(tmp_path)]
    assert main([*args, "--update-baseline"]) == 0
    probe.write_text("\n" * 10 + source)
    assert main(args) == 0
