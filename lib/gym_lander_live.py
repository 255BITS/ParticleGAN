"""Compare expert and three-generator controllers in the actual Box2D world."""
import base64
import json
import logging
import os
from pathlib import Path
import struct
import threading
import uuid
import zlib

import numpy as np
import torch

from experiments.train_gym_transition import load_checkpoint
from lib.gym_data import GYM_VERSION, terrain_context


OFF_ACTION = np.array([-1., 0.], dtype=np.float32)
CONTROLLER_LABELS = {"gan_joint": "GAN · joint discriminator",
                     "gan_marginals": "GAN · joint + marginal discriminators",
                     "joint": "GAN · pretrained joint fine-tune",
                     "original": "GAN · original world-model prototype",
                     "expert": "Non-GAN · heuristic expert",
                     "imitation": "Non-GAN fine-tune · pretrained imitation",
                     "state_probes": "Non-GAN · state-only probes",
                     "state_auxiliary": "Non-GAN · state-only auxiliary",
                     "sparse_probes": "Non-GAN · five-label-episode probes",
                     "sparse_auxiliary": "Non-GAN · five-label-episode auxiliary"}
NEW_GAN_CONTROLLERS = {"gan_joint", "gan_marginals"}
GAN_DEFAULT_CONTROLLERS = NEW_GAN_CONTROLLERS | {"joint"}
STATE_CONTROLLERS = {"state_probes", "state_auxiliary", "sparse_probes", "sparse_auxiliary"} | NEW_GAN_CONTROLLERS
CONTROLLER_MODES = {
    "expert": "Gym heuristic(state) → action",
    "original": "Existing E(st, previous action) → z → G2",
    "imitation": "E_control(st, previous action) → z → G2 · imitation training",
    "joint": "E_control(st, previous action) → z → G2 · joint G1/G2/G3 training",
    "state_probes": "E(st, terrain) → z → G2 → at · action training with detached G1/G3 probes",
    "state_auxiliary": "E(st, terrain) → z → G2 → at · action + state + successor training",
    "sparse_probes": "E(st, terrain) → z → G2 → at · sparse action labels, detached G1/G3 probes",
    "sparse_auxiliary": "E(st, terrain) → z → G2 → at · sparse action labels + abundant state/successor pairs",
    "gan_joint": "E(st, terrain) → z → G2 → at · joint masked-observation GAN",
    "gan_marginals": "E(st, terrain) → z → G2 → at · joint + marginal masked-observation GAN",
}
CONTROLLER_NOTES = {
    "expert": "The built-in heuristic supplies actions directly. The same real simulator advances every controller.",
    "original": "The original paired encoder was trained with current actions. Reusing it with previous actions is the untrained control prototype.",
    "imitation": "E_control and G2 learned expert action targets. G1, G3, the paired encoder, and the prior stayed frozen.",
    "joint": "Expert action imitation trains E_control and G2 alongside the state, action, and outcome generators, paired encoder, prior, and discriminators.",
    "state_probes": "Trained from scratch. E and the MoG prior learn actions through G2. G1 reconstructs st and G3 predicts st+1 from detached z; their losses cannot change E or the prior. There is no discriminator.",
    "state_auxiliary": "Trained from scratch. Action, state reconstruction, and successor prediction losses all train the shared E and MoG prior. There is no action input, previous-action input, or discriminator. G3 predicts successors associated with expert behavior, not arbitrary alternative commands.",
    "sparse_probes": "Trained from scratch with action labels from five expert episodes. E, G2, and the prior learn from those labels. G1/G3 train on state/successor pairs from all 47 expert episodes using detached z; their losses cannot change E or the prior. No discriminator or action/history input.",
    "sparse_auxiliary": "Trained from scratch with action labels from five expert episodes and state/successor pairs from all 47. G1/G3 losses also train E and the prior. This reduces explicit action labels; the transitions still come from expert behavior. No discriminator or action/history input. G3 cannot receive alternative commands.",
    "gan_joint": "Trained from scratch with a joint discriminator throughout. G1 → st, G2 → at, and G3 → st+1 share E and the MoG prior. Only five expert episodes provide action labels; both real and generated records receive matching observation masks. State and successor pairs come from all 47 expert episodes. The live controller reads only state and terrain.",
    "gan_marginals": "Trained from scratch with joint and marginal discriminators throughout. G1, G2, G3, E, and the MoG prior learn together. Five expert episodes provide explicit action labels; state/successor observations come from all 47. The same missing-action mask is applied to real and generated discriminator inputs. Live inference reads only state and terrain.",
}


def rgb_png_url(rgb):
    """Encode the simulator RGB frame with the standard library."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("Expected RGB simulator frame")
    def chunk(kind, data):
        return struct.pack("!I", len(data))+kind+data+struct.pack("!I", zlib.crc32(kind+data) & 0xffffffff)
    rows = b"".join(b"\x00"+row.tobytes() for row in rgb)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows, 1)) + chunk(b"IEND", b""))
    return "data:image/png;base64,"+base64.b64encode(png).decode("ascii")


@torch.no_grad()
def encoder_action(bundle, state, previous_action, terrain):
    """Return physical G2 command and route ID using only E and G2."""
    device, scaler = bundle["device"], bundle["scaler"]
    s, a, c = [torch.as_tensor(x, dtype=torch.float32, device=device).reshape(1, -1)
               for x in (state, previous_action, terrain)]
    encoded = bundle["E"](torch.cat([scaler.state(s), scaler.action(a)], 1), c, bundle["prior"])
    # G2's raw branch is tanh-bounded in physical command units. The full
    # generator subsequently standardizes this value only for reconstruction/D.
    action = bundle["G"].branches[1](torch.cat([encoded.codes[:, 0], c], 1)).tanh()
    result = action[0].cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("G2 produced a nonfinite command")
    return result, int(encoded.indices[0, 0])


class LiveLander:
    def __init__(self, checkpoint, device="cpu", seed=291000, render=True,
                 controllers_manifest=None, controller=None, state_controllers_manifest=None,
                 sparse_controllers_manifest=None, gan_controllers_manifest=None):
        import gymnasium as gym
        if gym.__version__ != GYM_VERSION:
            raise RuntimeError(f"Requires gymnasium=={GYM_VERSION}")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.device = device
        self.controller_paths = {"expert": None, "original": str(Path(checkpoint))}
        default_controller = "original"
        self.default_selection = "Original prototype · control comparison results unavailable"
        for manifest_path in (controllers_manifest, state_controllers_manifest, sparse_controllers_manifest, gan_controllers_manifest):
            if not manifest_path or not Path(manifest_path).exists():
                continue
            manifest = json.loads(Path(manifest_path).read_text())
            for name, specification in manifest["controllers"].items():
                if name not in CONTROLLER_LABELS:
                    raise ValueError(f"Unknown controller in manifest: {name}")
                path = specification.get("checkpoint")
                if name == "expert" or path:
                    self.controller_paths[name] = str(path) if path else None
            default_controller = manifest["default_controller"]
            if manifest_path == gan_controllers_manifest and default_controller not in GAN_DEFAULT_CONTROLLERS:
                raise ValueError("GAN manifest default must be a GAN-trained controller")
            self.default_selection = "Validation-selected learned controller: " + CONTROLLER_LABELS[default_controller]
            if manifest_path == gan_controllers_manifest:
                self.default_selection = "Validation-selected GAN controller: " + CONTROLLER_LABELS[default_controller]
        self.bundles = {}
        self._select_controller(controller or default_controller)
        self.env = gym.make("LunarLander-v3", continuous=True, enable_wind=False,
                            render_mode="rgb_array" if render else None)
        self.render = render
        self.lock = threading.RLock()
        self.reset(seed)

    def _select_controller(self, controller):
        if controller not in self.controller_paths:
            raise ValueError(f"Controller unavailable: {controller}")
        if controller != "expert" and controller not in self.bundles:
            if controller == "original":
                bundle = load_checkpoint(self.controller_paths[controller], self.device)
            elif controller in NEW_GAN_CONTROLLERS:
                from lib.gym_gan_control import load_gan_control_checkpoint
                bundle = load_gan_control_checkpoint(self.controller_paths[controller], self.device)
            elif controller in STATE_CONTROLLERS:
                from lib.gym_state_control import load_state_control_checkpoint
                bundle = load_state_control_checkpoint(self.controller_paths[controller], self.device)
            else:
                from lib.gym_control import load_control_checkpoint
                bundle = load_control_checkpoint(self.controller_paths[controller], self.device)
            if bundle["G"] is None or bundle.get("E_control", bundle.get("E")) is None:
                raise ValueError("Choose a three-generator checkpoint with an encoder")
            self.bundles[controller] = bundle
        self.controller = controller
        self.bundle = self.bundles.get(controller)
        self.checkpoint = self.controller_paths[controller]

    def switch_controller(self, controller, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer between 0 and 4294967295")
        with self.lock:
            self._select_controller(controller)
            return self.reset(seed)

    def reset(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer between 0 and 4294967295")
        with self.lock:
            self.seed = seed
            self.state, _ = self.env.reset(seed=seed)
            self.terrain = terrain_context(self.env)
            self.action = OFF_ACTION.copy()
            self.previous_action = OFF_ACTION.copy()
            self.episode_id = uuid.uuid4().hex
            self.step_count = 0
            self.reward = self.total_return = 0.
            self.terminated = self.truncated = False
            self.component_id = None
            self.termination_reason = None
            logging.info("RESET episode=%s seed=%s controller=%s initial_previous_action=%s",
                         self.episode_id, seed, self.controller, self.action.tolist())
            return self.snapshot()

    def step(self, episode_id, expected_step):
        with self.lock:
            if episode_id != self.episode_id or type(expected_step) is not int or expected_step != self.step_count:
                raise RuntimeError("State changed; refresh before stepping")
            if not (self.terminated or self.truncated):
                self.previous_action = self.action.copy()
                if self.controller == "expert":
                    from gymnasium.envs.box2d.lunar_lander import heuristic
                    self.action = np.asarray(heuristic(self.env.unwrapped, self.state), dtype=np.float32)
                    self.component_id = None
                elif self.controller == "original":
                    self.action, self.component_id = encoder_action(self.bundle, self.state, self.previous_action, self.terrain)
                elif self.controller in STATE_CONTROLLERS:
                    from lib.gym_state_control import state_control_action
                    self.action, self.component_id = state_control_action(self.bundle, self.state, self.terrain)
                else:
                    from lib.gym_control import control_action
                    self.action, self.component_id = control_action(self.bundle, self.state, self.previous_action, self.terrain)
                self.state, self.reward, self.terminated, self.truncated, _ = self.env.step(self.action)
                self.step_count += 1
                self.total_return += float(self.reward)
                if self.terminated or self.truncated:
                    from lib.gym_control_evaluation import termination_reason
                    self.termination_reason = termination_reason(self.env, self.terminated, self.truncated)
                    logging.info("END episode=%s controller=%s steps=%s return=%.3f reason=%s terminated=%s truncated=%s",
                                 self.episode_id, self.controller, self.step_count, self.total_return,
                                 self.termination_reason, self.terminated, self.truncated)
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            status = ({"successful_landing": "Successful landing", "crash": "Crashed",
                       "out_of_bounds": "Out of bounds", "time_limit": "Time limit reached"}
                      .get(self.termination_reason, "Paused · press Play or Step"))
            result = dict(episode_id=self.episode_id, seed=self.seed, step=self.step_count,
                state=self.state.tolist(), terrain=self.terrain.tolist(), action=self.action.tolist(),
                previous_action=self.previous_action.tolist(), reward=float(self.reward),
                terminated=bool(self.terminated), truncated=bool(self.truncated),
                done=bool(self.terminated or self.truncated), status=status, component_id=self.component_id,
                controller=self.controller, controllers=[dict(id=name, label=CONTROLLER_LABELS[name])
                    for name in CONTROLLER_LABELS if name in self.controller_paths],
                termination_reason=self.termination_reason, mode=CONTROLLER_MODES[self.controller],
                input_kind="state_only" if self.controller in STATE_CONTROLLERS else (
                    "expert" if self.controller == "expert" else "state_previous_action"),
                note=CONTROLLER_NOTES[self.controller], default_selection=self.default_selection,
                checkpoint=self.checkpoint)
            result["return"] = self.total_return
            if self.render:
                result["frame"] = rgb_png_url(self.env.render())
            return result

    def close(self):
        self.env.close()


def handler_for(lander):
    from http.server import BaseHTTPRequestHandler
    html = Path(__file__).with_suffix(".html").read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def send(self, value, status=200, content_type="application/json"):
            payload = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/":
                self.send(html, content_type="text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send(lander.snapshot())
            else:
                self.send({"error": "Not found"}, 404)

        def do_POST(self):
            # Same-origin local UI commands only; state-changing calls use JSON.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send({"error": "Expected application/json"}, 415)
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://"+self.headers.get("Host", ""):
                self.send({"error": "Origin mismatch"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/reset":
                    result = lander.reset(body["seed"])
                elif self.path == "/api/controller":
                    result = lander.switch_controller(body["controller"], body["seed"])
                elif self.path == "/api/step":
                    result = lander.step(body["episode_id"], body["step"])
                else:
                    self.send({"error": "Not found"}, 404)
                    return
                self.send(result)
            except (ValueError, KeyError, TypeError) as error:
                self.send({"error": str(error)}, 400)
            except RuntimeError as error:
                self.send({"error": str(error)}, 409)

        def log_message(self, fmt, *args):
            if len(args) < 2 or str(args[1]) != "200":
                logging.info("HTTP "+fmt, *args)
    return Handler
