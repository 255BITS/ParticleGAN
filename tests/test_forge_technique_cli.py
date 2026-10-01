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
