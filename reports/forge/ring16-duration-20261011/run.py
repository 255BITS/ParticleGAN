"""Entry point; execute.py retains the byte-exact completed training runner."""
import sys
from execute import main, load_protocol, sha, write


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "publish":
        from publish import publish, HERE, ROOT
        if len(sys.argv) != 2:
            raise SystemExit("For custom paths, call publish.py --raw PATH --output PATH")
        publish(ROOT / "runs/api/ring16-duration-20261011", HERE)
    else:
        main()
