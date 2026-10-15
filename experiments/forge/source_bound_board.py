"""Visible source-bound benchmark rows; never alter selected qualification rows."""
from pathlib import Path
from .contracts import read_json, file_hash

BOARD_SOURCE = Path('reports/forge/source-bound-board.json')


def validate_board(data):
    if data.get('schema_version') != 1 or data.get('qualification_input') is not False:
        raise ValueError('source-bound board must be explicitly nonqualifying')
    view = data['recorded_view']
    required = {a['task']: a['qualification_tier'] for a in view['assignments'] if a['importance'] == 'required'}
    if view['revision'] != 9 or len(required) != 29 or required.get('gaussian1d_smoke') != 1 or required.get('gaussian1d_stability') != 2:
        raise ValueError('recorded benchmark tier/unit contract differs')
    for row in data['rows']:
        cells = row['benchmark_tasks']
        if set(cells) != set(required) or row['whole_gauntlet_score'] is not None or row['qualification_input'] is not False:
            raise ValueError('every required benchmark must be measured or explicitly null')
        measured = {k: v for k, v in cells.items() if v['status'] is not None}
        if set(measured) != {'gaussian1d_smoke', 'gaussian1d_stability'}:
            raise ValueError('a partial pair cannot fill untested cells')
        if any(v['tier'] != required[k] or v['status'] not in ('PASS', 'FAIL', None) for k, v in cells.items()):
            raise ValueError('recorded task verdict/tier changed')
        if cells['gaussian1d_smoke']['status'] != 'PASS' or cells['gaussian1d_stability']['status'] != 'FAIL':
            raise ValueError('the original acquisition PASS/retention FAIL must remain visible')
        if row['benchmark_counts'] != dict(passed=1, failed=1, run=2, required_unique_tasks=29, untested=27):
            raise ValueError('observation checks cannot be counted as benchmark tasks')
        phases = row['retention_phases']
        if any(p['recorded_task'] != 'gaussian1d_stability' or p['recorded_task_tier'] != 2 for p in phases.values()):
            raise ValueError('all phases belong to the same recorded Tier2 benchmark')
        for name, total in [('stationary', 72), ('shift_hold', 24)]:
            p = phases[name]
            if p['unit'] != 'observation_checks' or p['checks'] != total or not 0 <= p['passed'] <= total:
                raise ValueError('phase fraction has incorrect units or bounds')
    return data


def main_board_rows(root, page):
    path = Path(root) / BOARD_SOURCE
    if not path.exists():
        return []
    data = validate_board(read_json(path))
    from .confidence_coherence_scores import validate_original_rows, main_board_rows as paired_rows
    validate_original_rows(root, data)
    lines = ['## Source-bound benchmark rows', '',
        'These rows are visible on the main board but do not replace a selected family configuration. '
        'Benchmark-task totals and within-task observation checks have different units. '
        'Untested benchmark cells are null in the machine data; no whole-gauntlet score is inferred.', '',
        '| Source-bound candidate / matched control | Benchmark tasks passed / run | Required benchmark tasks untested | Recorded Tier2 stationary checks | Recorded Tier2 reacquisition | Recorded Tier2 shifted-hold checks | Original retention benchmark |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in data['rows']:
        p = row['retention_phases']; label = row['label']; target = row['readout']
        import os
        link = os.path.relpath(Path(root) / target, Path(page).parent)
        lines.append(f"| [{label}]({link}) | **1 of 2 benchmark tasks** | **27 of 29 required unique benchmark tasks** | {p['stationary']['passed']} of 72 observation checks | {p['reacquisition']['status']} (phase) | {p['shift_hold']['passed']} of 24 observation checks | **FAIL** |")
    lines += ['', 'Recorded discriminator_stability revision9: acquisition is Tier1 and the complete stability benchmark is Tier2. '
        'Stationary retention, deadline reacquisition and shifted hold are phases of that same Tier2 task; '
        'they are not separate Tier3 benchmark passes. The proposed revision10 display split is not applied retrospectively.', '',
        f'Machine row source: [source-bound-board.json](source-bound-board.json), SHA256 `{file_hash(path)}`.', '']
    lines += paired_rows(root, page)
    lines += ['## Selected family rows', '']
    return lines
