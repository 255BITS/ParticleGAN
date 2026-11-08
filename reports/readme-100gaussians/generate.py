"""Generate the README animation from the current public default (record + render).

Historical seed-1234 runs and their original renderer are preserved in Git history;
see README.md and summary-hero.json. No historical recipe is substituted here.
"""
from record import main

if __name__ == "__main__":
    raise SystemExit(main())
