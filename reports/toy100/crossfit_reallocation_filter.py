"""The unchanged 44-check gate, rebound to a separately frozen cross-fit rule."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100 import reallocation_state_filter as gate
from reports.toy100.crossfit_reallocation import METHOD,crossfit_reallocation


if __name__=='__main__':
    sources=('reports/toy100/crossfit_reallocation_filter.py',
             'reports/toy100/crossfit_reallocation.py')+gate.SOURCES
    with patch.object(gate,'SOURCES',sources),patch.object(gate,'METHOD',METHOD),\
         patch.object(gate,'reallocation_smoothed_candidate',crossfit_reallocation):
        gate.main()
