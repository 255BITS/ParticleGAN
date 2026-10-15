from copy import deepcopy
from pathlib import Path
import json
import pytest
from experiments.forge.source_bound_board import validate_board, main_board_rows
from experiments.forge.family_reports import render_leaderboard, render_family

ROOT = Path(__file__).resolve().parents[1]


def test_actual_source_rows_keep_benchmark_and_check_units_separate():
    data = json.loads((ROOT/'reports/forge/source-bound-board.json').read_text())
    validate_board(data)
    candidate, control = data['rows']
    assert candidate['benchmark_counts'] == control['benchmark_counts'] == dict(passed=1,failed=1,run=2,required_unique_tasks=29,untested=27)
    assert sum(v['status'] is None for v in candidate['benchmark_tasks'].values()) == 27
    assert candidate['whole_gauntlet_score'] is None
    assert candidate['retention_phases']['stationary']['passed'] == 69
    assert control['retention_phases']['stationary']['passed'] == 35
    assert all(p['recorded_task_tier']==2 for row in data['rows'] for p in row['retention_phases'].values())
    altered=deepcopy(data);altered['rows'][0]['benchmark_counts']['passed']=69
    with pytest.raises(ValueError,match='observation'):validate_board(altered)
    altered=deepcopy(data);altered['rows'][0]['retention_phases']['reacquisition']['recorded_task_tier']=3
    with pytest.raises(ValueError,match='Tier2'):validate_board(altered)


def test_rendered_main_board_matches_machine_source_and_preserves_selection():
    publication=json.loads((ROOT/'reports/forge/technique-inventory.json').read_text());before=deepcopy(publication)
    text=render_leaderboard(ROOT,publication,ROOT/'reports/forge/technique-inventory.md')
    assert publication==before
    assert text.index('| Family / view | Tier 1') < text.index('G-only coherent mean')
    row=next(line for line in text.splitlines() if line.startswith('| ↳ †') and 'G-only coherent mean' in line)
    assert '[1(*)/6]' in row and '[0(*)/21]' in row and '[0(*)/2]' in row and '[1(*)/29]' in row
    family=next(f for f in publication['family_progress']['families'] if f['id']=='bcap-pure')
    detail=render_family(ROOT,publication,family)
    assert '1 of 2 benchmark tasks' in detail and '27 of 29 required unique benchmark tasks' in detail
    assert '69 of 72 observation checks' in detail and '35 of 72 observation checks' in detail
    assert '20 of 24 observation checks' in detail and '12 of 24 observation checks' in detail
    assert '**FAIL**' in detail and 'not separate Tier3 benchmark passes' in detail
    assert 'required-view instances' in text and 'not distinct training runs' in text
