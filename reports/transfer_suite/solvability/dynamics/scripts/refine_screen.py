import json
from pathlib import Path
import search

cards=json.loads(Path('/tmp/pr36-stress-solvability/refinement_cards.json').read_text())
report=search.initialize()
for card in cards:
    search.run(report,card,search.TASKS[0])
print('COMPLETE refine_screen',len(report['rows']),flush=True)
