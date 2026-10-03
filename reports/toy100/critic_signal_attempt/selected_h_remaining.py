"""Run the exact archived screen with explicitly archived host-only plumbing."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critic_signal
import critic_signal_screen as screen
from selected_h_extension import extended_signal_policy

# This runs in spawned workers too, before screen.worker imports signal_policy.
critic_signal.signal_policy = extended_signal_policy
_original_sources = screen.source_hashes

def sources():
    values = _original_sources()
    for name in ('selected_h_extension.py', 'selected_h_remaining.py'):
        rel = 'reports/toy100/' + name
        values[rel] = screen.sha(ROOT / rel)
    return values

screen.source_hashes = sources

if __name__ == '__main__':
    screen.main()
