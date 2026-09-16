"""Completed-only handoff report with the previous no-trajectory baseline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.analyze_memory_core import analyze

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = Path(__file__).resolve().parents[1]/'runs/memory_path/core_round1/runs/handoff_only'
    manifest = args.source.parent/'report_baselines.json'
    baselines = [Path(p) for p in json.loads(manifest.read_text())] if manifest.exists() else [baseline]
    analyze(args.source, args.out, baselines=baselines, handoff_only=True)
