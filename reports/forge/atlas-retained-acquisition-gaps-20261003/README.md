# Retained acquisition and stability audit

Overlap and intensity both completed their original terminal-five gates. Early passing reads were followed by losses before five consecutive passes; neither lost a gate after confirmation. The ring failed its disjoint hold through precision loss while retaining all eight modes.

[Chronology and exact interpretation](GAPS.md) explains each question, unchanged inequality and selected public sampling law. [Portable numeric receipt](gaps.json) retains the original source, grades, 19 consumed file pins and complete observation chronology. This is an audit of saved observations, with zero models, draws, scoring or training updates. It does not add qualification or change an original verdict.

The original 26-slot cohort remains **7 PASS, 11 FAIL and 8 BLOCKED**. Overlap confirms acquisition at step 850 after a stable suffix beginning 650; intensity confirms at 525 after a suffix beginning 425. The ring drops to HQ .89453125 at 1407 against its .9 threshold, with eight modes and cover 1. Passing checkpoints establish no guarantee about behavior between those checkpoints.

Root independently replayed the [stdlib audit](audit.py) in a new external directory and reproduced `gaps.json` byte for byte, SHA256 `44cb50568d59208530a264cd6ad84ced0e49e3ffdb7bce6dab558630a165682a`. See [publication verification](publication-proof.json). The source-bound raw JSON archive is local to this machine; reproducing the audit requires those pinned inputs at their recorded paths. Copy `audit.py` into a fresh external directory without `gaps.json` and run `python -B audit.py`. It refuses to overwrite an existing receipt.

This chronology was frozen before the separate grid output-kernel endpoint attempt. The prepared-endpoint wording at the end of GAPS.md describes that earlier cut; it is not a completion claim for that later experiment. The [original full-run report and GIFs](../atlas-current-gpu-diagnostics-native-v2-20261003/README.md) retain the training trajectories.
