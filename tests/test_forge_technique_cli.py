from experiments.forge.__main__ import parser


def test_inventory_cli_defaults_discover_all_techniques_through_all_tiers():
    args = parser().parse_args(["inventory", "run"])
    assert args.stage == "run"
    assert args.view == "discriminator_stability"
    assert args.through_tier == 3
    assert args.gpus == "0,1"
    assert str(args.campaign) == "configs/forge/campaigns/technique-inventory.json"


def test_techniques_cli_regeneration_selects_compute_without_execution_options():
    args = parser().parse_args([
        "techniques", "--device", "cuda", "--output", "reports/forge/technique-inventory"
    ])
    assert args.command == "techniques"
    assert args.device == "cuda"
    assert str(args.output) == "reports/forge/technique-inventory"


def test_inventory_compiles_boards_once_after_all_attempts(monkeypatch, tmp_path):
    from experiments.forge import __main__ as cli, knowledge, queue, technique_inventory

    events = []

    class FakeQueue:
        def __init__(self, root, *, report_root, on_completion):
            self.on_completion = on_completion

    def run(root, queue_root, *, queue, **options):
        for _ in range(3):
            events.append("receipt")
            if queue.on_completion:
                queue.on_completion()
        return {"stage": "drained"}

    monkeypatch.setattr(queue, "Queue", FakeQueue)
    monkeypatch.setattr(technique_inventory, "run_inventory", run)
    monkeypatch.setattr(knowledge, "compile_memory", lambda root: events.append("compile"))
    monkeypatch.setattr(cli, "emit", lambda result: None)
    cli.main(["--root", str(tmp_path), "inventory", "run", "--gpus", "cpu"])
    assert events == ["receipt", "receipt", "receipt", "compile"]
