"""Keep the public README illustration aligned with the selected default and its evidence."""
import hashlib
import json
from pathlib import Path

from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_readme_animation_uses_the_current_default():
    receipt = json.loads((ROOT / 'reports/readme-100gaussians/summary-current.json').read_text())
    expected = json.loads(json.dumps(get_recipe().to_dict()))
    assert receipt['recipe'] == expected, 'Regenerate the README animation when promoting a new default'
    assert receipt['completed_steps'] == expected['total_steps']
    assert receipt['seed'] == 0
    assert receipt['task']['initialization'] == {'generator': 0, 'critic': 1, 'prior': 2}


def test_readme_animation_preserves_its_recorded_verdict_and_gif():
    receipt = json.loads((ROOT / 'reports/readme-100gaussians/summary-current.json').read_text())
    animation = ROOT / '100gaussians.gif'
    assert hashlib.sha256(animation.read_bytes()).hexdigest() == receipt['render']['gif_sha256']
    assert receipt['render']['observations_sha256'] == receipt['observations_sha256']
    assert receipt['render']['frame_steps'] == receipt['media_steps']
    assert receipt['media_steps'][0] == 0
    assert receipt['media_steps'][-1] == receipt['completed_steps']
    assert len(receipt['terminal_checks']) == receipt['gate']['terminal_checks']
    passed = all(row['modes'] == 100 and row['hq'] >= .95 for row in receipt['terminal_checks'])
    assert receipt['status'] == ('PASS' if passed else 'FAIL')
    assert receipt['render']['status'] == receipt['status']
