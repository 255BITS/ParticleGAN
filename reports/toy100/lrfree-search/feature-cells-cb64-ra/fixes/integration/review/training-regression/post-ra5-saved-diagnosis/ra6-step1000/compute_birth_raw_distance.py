"""Recompute selected-reference distances directly; retain float32 cdist evidence."""
import hashlib
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
source = here / 'birth-raw-annotation.json'
output = here / 'birth-raw-distance-correction.json'
assert not output.exists()
original = source.read_bytes()
receipt = json.loads(original)
rows = [dict(row=r['row'], model=r['model'],
    recorded_float32_cdist_distance=r['distance_to_nearest_even_reference'],
    selected_even_reference_row=r['nearest_even_reference_row'],
    direct_coordinate_distance=math.dist(r['point'], r['nearest_even_reference_point']))
    for r in receipt['records']]
assert source.read_bytes() == original
output.write_text(json.dumps(dict(status='DIRECT_DISTANCE_CLARIFICATION', records=rows,
    input_sha256=hashlib.sha256(original).hexdigest(),
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Direct norm to the reference selected by the saved float32 cdist calculation; no exact-match claim.',
    inputs_unchanged=True, frozen_writes=0, torch_imported=False), indent=2) + '\n')
print(json.dumps(rows, indent=2))
