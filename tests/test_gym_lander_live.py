"""Verify the live feedback loop against a separate actual simulator."""
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")
pytest.importorskip("Box2D")

from lib.gym_data import make_env
from lib.gym_lander_live import LiveLander, OFF_ACTION, encoder_action, handler_for, rgb_png_url
from lib.gym_transition import GymTransitionEncoder, GymTransitionGenerator, GymTransitionScaler
from particlegan import get_recipe


@pytest.fixture
def bundle():
    torch.set_num_threads(1)
    torch.manual_seed(45)
    scaler = GymTransitionScaler(torch.zeros(6), torch.ones(6), torch.zeros(2), torch.ones(2))
    return dict(device=torch.device("cpu"), scaler=scaler,
        G=GymTransitionGenerator(scaler, z_dim=4, width=8).eval(),
        E=GymTransitionEncoder(z_dim=4, width=8).eval(),
        prior=get_recipe("mog", z_dim=4, num_particles=8).make_prior(device="cpu"))


@contextmanager
def lander_fixture(monkeypatch, bundle):
    monkeypatch.setattr("lib.gym_lander_live.load_checkpoint", lambda *args: bundle)
    lander = LiveLander("test.pt", render=False, seed=53)
    try:
        yield lander
    finally:
        lander.close()


def test_encoder_previous_action_drives_real_simulator(monkeypatch, bundle):
    with lander_fixture(monkeypatch, bundle) as lander:
        reference = make_env()
        try:
            reference_state, _ = reference.reset(seed=53)
            previous_action = OFF_ACTION.copy()
            for step in range(8):
                expected_action, route = encoder_action(bundle, reference_state, previous_action, lander.terrain)
                reference_state, reward, terminated, truncated, _ = reference.step(expected_action)
                actual = lander.step(lander.episode_id, step)
                np.testing.assert_array_equal(actual["previous_action"], previous_action)
                np.testing.assert_array_equal(actual["action"], expected_action)
                np.testing.assert_array_equal(actual["state"], reference_state)
                assert actual["component_id"] == route
                assert actual["reward"] == reward
                assert actual["terminated"] == terminated and actual["truncated"] == truncated
                previous_action = expected_action
            # Inference must not populate model gradients or train any parameters.
            assert all(p.grad is None for m in (bundle["G"], bundle["E"], bundle["prior"]) for p in m.parameters())
        finally:
            reference.close()


def test_http_steps_once_stops_at_end_and_reset_invalidates_stale_commands(monkeypatch, bundle):
    with lander_fixture(monkeypatch, bundle) as lander:
        lander.env._max_episode_steps = 2
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(lander))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request(path, body=None):
            req = Request(f"http://127.0.0.1:{server.server_port}"+path,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=5) as response:
                return json.load(response)
        try:
            initial = request("/api/state")
            assert initial["step"] == 0
            body = dict(episode_id=initial["episode_id"], step=0)
            first = request("/api/step", body)
            assert first["step"] == 1
            with pytest.raises(HTTPError) as error:
                request("/api/step", body)
            assert error.value.code == 409
            final = request("/api/step", dict(episode_id=first["episode_id"], step=1))
            assert final["truncated"] and final["done"] and final["step"] == 2
            assert final["termination_reason"] == "time_limit"
            again = request("/api/step", dict(episode_id=final["episode_id"], step=2))
            assert final == again
            reset = request("/api/reset", dict(seed=53))
            assert reset["step"] == 0 and reset["episode_id"] != initial["episode_id"]
            assert reset["state"] == initial["state"]
            with pytest.raises(HTTPError):
                request("/api/step", body)
            expert = request("/api/controller", dict(controller="expert", seed=53))
            assert expert["controller"] == "expert" and expert["step"] == 0
            assert expert["checkpoint"] is None and expert["component_id"] is None
            assert expert["episode_id"] != reset["episode_id"]
            with pytest.raises(HTTPError):
                request("/api/step", dict(episode_id=reset["episode_id"], step=0))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def test_expert_switch_preserves_reset_world_and_uses_actual_heuristic(monkeypatch, bundle):
    from gymnasium.envs.box2d.lunar_lander import heuristic
    with lander_fixture(monkeypatch, bundle) as lander:
        initial = lander.snapshot()
        lander.step(lander.episode_id, 0)
        switched = lander.switch_controller("expert", 53)
        assert switched["state"] == initial["state"]
        assert switched["return"] == 0 and switched["termination_reason"] is None
        reference = make_env()
        try:
            state, _ = reference.reset(seed=53)
            for step in range(8):
                action = heuristic(reference.unwrapped, state)
                state, reward, _, _, _ = reference.step(action)
                actual = lander.step(lander.episode_id, step)
                np.testing.assert_array_equal(actual["action"], action)
                np.testing.assert_array_equal(actual["state"], state)
                assert actual["reward"] == reward and actual["component_id"] is None
        finally:
            reference.close()
        episode_id = lander.episode_id
        with pytest.raises(ValueError):
            lander.switch_controller("missing", 53)
        assert lander.controller == "expert" and lander.episode_id == episode_id
        restored = lander.switch_controller("original", 53)
        assert restored["state"] == initial["state"] and restored["step"] == 0


def test_manifest_selects_trained_control_encoder_and_lazy_loads(monkeypatch, bundle, tmp_path):
    import lib.gym_control as control
    calls = []
    control_bundle = dict(bundle, E_control=bundle["E"])
    def load(path, device):
        calls.append(path)
        return control_bundle
    monkeypatch.setattr(control, "load_control_checkpoint", load)
    monkeypatch.setattr("lib.gym_lander_live.load_checkpoint", lambda *args: bundle)
    manifest = tmp_path / "controllers.json"
    manifest.write_text(json.dumps(dict(default_controller="joint", controllers={
        "expert": {"checkpoint": None}, "original": {"checkpoint": "original.pt"},
        "imitation": {"checkpoint": "imitation.pt"}, "joint": {"checkpoint": "joint.pt"}})))
    lander = LiveLander("original.pt", render=False, seed=53, controllers_manifest=manifest)
    try:
        assert lander.controller == "joint" and calls == ["joint.pt"]
        expected_action, expected_route = control.control_action(control_bundle, lander.state, OFF_ACTION, lander.terrain)
        result = lander.step(lander.episode_id, 0)
        np.testing.assert_array_equal(result["action"], expected_action)
        assert result["component_id"] == expected_route
        lander.switch_controller("imitation", 53)
        assert calls == ["joint.pt", "imitation.pt"]
        lander.switch_controller("joint", 53)
        assert calls == ["joint.pt", "imitation.pt"]
    finally:
        lander.close()


@pytest.mark.parametrize("default_controller", ["state_probes", "sparse_probes", "sparse_auxiliary", "gan_joint", "gan_marginals"])
def test_state_only_controllers_merge_manifest_and_ignore_previous_action(monkeypatch, bundle, tmp_path, default_controller):
    import lib.gym_state_control as control
    import lib.gym_gan_control as gan_control
    state_bundle = dict(bundle, E=control.GymStateEncoder(z_dim=4, width=8).eval())
    monkeypatch.setattr(control, "load_state_control_checkpoint", lambda *args: state_bundle)
    monkeypatch.setattr(gan_control, "load_gan_control_checkpoint", lambda *args: state_bundle)
    monkeypatch.setattr("lib.gym_lander_live.load_checkpoint", lambda *args: bundle)
    old_manifest = tmp_path / "old.json"
    old_manifest.write_text(json.dumps(dict(default_controller="original", controllers={
        "expert": {"checkpoint": None}, "original": {"checkpoint": "original.pt"},
        "imitation": {"checkpoint": "imitation.pt"}, "joint": {"checkpoint": "joint.pt"}})))
    new_manifest = tmp_path / "new.json"
    new_manifest.write_text(json.dumps(dict(default_controller="state_auxiliary", controllers={
        "state_probes": {"checkpoint": "probes.pt"}, "state_auxiliary": {"checkpoint": "auxiliary.pt"}})))
    sparse_manifest = tmp_path / "sparse.json"
    sparse_manifest.write_text(json.dumps(dict(default_controller="sparse_probes" if default_controller.startswith("gan_") else default_controller, controllers={
        "sparse_probes": {"checkpoint": "sparse_probes.pt"},
        "sparse_auxiliary": {"checkpoint": "sparse_auxiliary.pt"}})))
    gan_manifest = None
    if default_controller.startswith("gan_"):
        gan_manifest = tmp_path / "gan.json"
        gan_manifest.write_text(json.dumps(dict(default_controller=default_controller, controllers={
            "gan_joint": {"checkpoint": "gan_joint.pt"}, "gan_marginals": {"checkpoint": "gan_marginals.pt"}})))
    lander = LiveLander("original.pt", render=False, seed=53, controllers_manifest=old_manifest,
                        state_controllers_manifest=new_manifest, sparse_controllers_manifest=sparse_manifest,
                        gan_controllers_manifest=gan_manifest)
    reference = make_env()
    try:
        initial = lander.snapshot()
        assert initial["controller"] == default_controller
        assert initial["input_kind"] == "state_only"
        expected_controllers = {
            "expert", "original", "imitation", "joint", "state_probes", "state_auxiliary",
            "sparse_probes", "sparse_auxiliary"}
        if gan_manifest:
            expected_controllers |= {"gan_joint", "gan_marginals"}
            assert "Validation-selected GAN controller" in initial["default_selection"]
        assert {c["id"] for c in initial["controllers"]} == expected_controllers
        state, _ = reference.reset(seed=53)
        for step in range(4):
            expected_action, route = control.state_control_action(state_bundle, state, lander.terrain)
            # Deliberately alter the previous-command value; state-only E must
            # still produce exactly the action obtained from state + terrain.
            lander.action = np.array([.99, -.99], dtype=np.float32)
            state, reward, _, _, _ = reference.step(expected_action)
            result = lander.step(lander.episode_id, step)
            np.testing.assert_array_equal(result["action"], expected_action)
            np.testing.assert_array_equal(result["state"], state)
            assert result["reward"] == reward and result["component_id"] == route
        switched = lander.switch_controller("state_probes", 53)
        assert switched["state"] == initial["state"] and switched["step"] == 0
        assert switched["episode_id"] != initial["episode_id"]
        original = lander.switch_controller("original", 53)
        assert original["input_kind"] == "state_previous_action"
    finally:
        reference.close()
        lander.close()


def test_gan_manifest_rejects_non_gan_default_but_keeps_history(monkeypatch, bundle, tmp_path):
    import lib.gym_control as control
    monkeypatch.setattr(control, "load_control_checkpoint", lambda *args: dict(bundle, E_control=bundle["E"]))
    manifest = tmp_path / "gan.json"
    value = dict(default_controller="imitation", controllers={
        "joint": {"checkpoint": "joint.pt"}, "imitation": {"checkpoint": "imitation.pt"}})
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="GAN manifest default"):
        LiveLander("original.pt", render=False, gan_controllers_manifest=manifest)
    value["default_controller"] = "joint"
    manifest.write_text(json.dumps(value))
    lander = LiveLander("original.pt", render=False, gan_controllers_manifest=manifest)
    try:
        assert lander.controller == "joint"
        history = lander.switch_controller("imitation", 53)
        assert history["controller"] == "imitation"
        assert "Non-GAN" in next(c["label"] for c in history["controllers"] if c["id"] == "imitation")
    finally:
        lander.close()


def test_rgb_frame_png_roundtrip():
    import base64
    import io
    Image = pytest.importorskip("PIL.Image")
    pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    url = rgb_png_url(pixels)
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    np.testing.assert_array_equal(np.asarray(image), pixels)
