"""Launch a live simulator with expert and three-generator controllers."""
import argparse
from http.server import ThreadingHTTPServer
import logging
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.gym_lander_live import CONTROLLER_LABELS, LiveLander, handler_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/gym/lunar_lander/adversarial/best.pt")
    parser.add_argument("--controllers-manifest", default="reports/gym/lunar_lander_control/controllers.json")
    parser.add_argument("--state-controllers-manifest", default="reports/gym/lunar_lander_state_control/controllers.json",
                        help="Merge state-only controllers; its comparable validation winner sets the default")
    parser.add_argument("--sparse-controllers-manifest", default="reports/gym/lunar_lander_sparse_action/controllers.json",
                        help="Merge sparse-action controllers; its comparable validation winner sets the default")
    parser.add_argument("--gan-controllers-manifest", default="reports/gym/lunar_lander_gan_control/controllers.json",
                        help="Merge GAN controllers; its validation-selected GAN winner takes default priority")
    parser.add_argument("--controller", choices=tuple(CONTROLLER_LABELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=291000)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--log", default="results/gym/lunar_lander/live_controller_server.log")
    args = parser.parse_args()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.log)])
    torch.set_num_threads(1)
    lander = LiveLander(args.checkpoint, args.device, args.seed,
                        controllers_manifest=args.controllers_manifest, controller=args.controller,
                        state_controllers_manifest=args.state_controllers_manifest,
                        sparse_controllers_manifest=args.sparse_controllers_manifest,
                        gan_controllers_manifest=args.gan_controllers_manifest)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(lander))
        logging.info("PLAY http://localhost:%s controller=%s checkpoint=%s device=%s",
                     args.port, lander.controller, lander.checkpoint, args.device)
        logging.info("%s; real simulator.step(action)", lander.snapshot()["mode"])
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        lander.close()


if __name__ == "__main__":
    main()
