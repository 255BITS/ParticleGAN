"""Same frozen filter order for the one declared two-player Armijo arm."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reports.toy100 import alternating_linesearch_probe as controller
from reports.toy100.alternating_two_player_linesearch import alternating_two_player_linesearch,METHOD


if __name__=='__main__':
    options=dict(controller.OPTIONS,g_armijo=.875)
    files=[Path(__file__),ROOT/'reports/toy100/alternating_two_player_linesearch.py']
    with patch.object(controller,'alternating_linesearch',alternating_two_player_linesearch),\
         patch.object(controller,'METHOD',METHOD),patch.object(controller,'OPTIONS',options),\
         patch.object(controller,'CANDIDATE_KEY','dg_linesearch'),patch.object(controller,'ADAPTER_FILES',files),\
         patch.object(controller,'CHANGE','Retain verified D rule; replace unchecked G norm bound by actual G Armijo c7/8, the same scalar quadratic curvature margin.25'):
        controller.main()
