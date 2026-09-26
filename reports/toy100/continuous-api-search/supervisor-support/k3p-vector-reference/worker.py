"""One declared public-v0.8.0 K3P unequal-mass reference. No training on import."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from preflight import verify
    verification = verify(BUNDLE)
    declaration = json.loads((BUNDLE / "declaration.json").read_text())
    manifest = json.loads((BUNDLE / "bundle-sha256.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "declaration.json", declaration)
    dump(args.output / "source-preflight.json", verification)
    with zipfile.ZipFile(args.output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest["files"]:
            archive.write(BUNDLE / name, name)
        archive.write(BUNDLE / "bundle-sha256.json", "bundle-sha256.json")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    if any(name == "particlegan" or name.startswith("particlegan.") for name in sys.modules):
        raise RuntimeError("run in a fresh process: ParticleGAN was already imported")
    sys.path.insert(0, str(BUNDLE / "source"))
    import torch
    from particlegan import GANTrainer, get_recipe, BatchDistanceDiscriminator
    from particlegan.training import input_noise_std, output_noise_std
    from particlegan.recipes import learning_rate_scales
    import frozen_host as host
    from execution_contract import EXECUTION, host_cuda, serial_step, validate_checkpoint

    identity = {"bundle_manifest_sha256": hashlib.sha256((BUNDLE / "bundle-sha256.json").read_bytes()).hexdigest(),
                "recipe": declaration["recipe"], "fixture_sha256": declaration["fixture"]["sha256"],
                "evaluation": declaration["evaluation"]}
    trainer = stream = None
    started = time.monotonic()
    observations = []
    runtime = {}

    def checkpoint():
        return dict(schema=1, execution=EXECUTION, identity=identity,
                    trainer=trainer.state_dict(), data_rng=stream.get_state())

    def restore_checkpoint(envelope):
        validate_checkpoint(envelope, identity)
        trainer.load_state_dict(envelope["trainer"])
        stream.set_state(envelope["data_rng"].cpu())

    def assert_cpu_defaults():
        assert str(torch.get_default_device()) == "cpu", "learner factory default must stay CPU"
        assert torch.Generator is original_generator, "host generator routing leaked into learner"

    original_generator = torch.Generator
    try:
        assert_cpu_defaults()
        assert torch.cuda.is_available(), "CUDA is required; no CPU score fallback"
        expected_runtime = declaration["runtime_expected"]
        assert str(torch.__version__) == expected_runtime["torch"]
        assert torch.version.cuda == expected_runtime["cuda"]
        assert torch.cuda.get_device_name(0) == expected_runtime["gpu"]
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        imported = {}
        for name, module in tuple(sys.modules.items()):
            if name == "particlegan" or name.startswith("particlegan."):
                path = Path(module.__file__).resolve()
                relative = str(path.relative_to(BUNDLE / "source"))
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                assert actual == declaration["release"]["source_sha256"][relative]
                imported[name] = {"path": str(path), "sha256": actual}
        runtime = dict(torch=str(torch.__version__), torch_revision=torch.version.git_version,
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(0), deterministic=True, tf32=False,
                       threads=1, interop_threads=1, imported_package=imported,
                       default_device_during_learner="cpu", serial_backward=True,
                       environment={k: os.environ.get(k) for k in
                                    ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                                     "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        dump(args.output / "runtime.json", runtime)
        cfg = host.resolve(declaration["spec"])
        assert cfg == declaration["spec"] and cfg["steps"] == 1200
        recipe = get_recipe(total_steps=1200, num_particles=256, z_dim=4, batch_size=128)
        assert json.loads(json.dumps(recipe.to_dict())) == declaration["recipe"]
        assert input_noise_std(recipe, 120) == 0 and output_noise_std(recipe, 240) == .029
        torch.manual_seed(0)
        prior = recipe.make_prior(init_std=.5, generator=torch.Generator(device="cpu").manual_seed(0))
        generator = host.SimpleMLPGenerator(cfg["z_dim"], cfg["hidden"], cfg["layers"], 2)
        card = declaration["card"]
        assert card["implementation"] == "shared_batch_feature_v1"
        critic = BatchDistanceDiscriminator(in_dim=2, hidden_dim=card["width"],
                    n_hidden=card["layers"], scales=tuple(card["kernel_scales"]),
                    beta=card["softplus_beta"], eps=card["eps"])
        fixture = torch.load(BUNDLE / "initial-values.pt", map_location="cpu", weights_only=True)
        groups = [list(generator.parameters()) + list(prior.parameters()), list(critic.parameters())]
        assert len(groups) == len(fixture) == 2
        with torch.no_grad():
            for params, values in zip(groups, fixture):
                assert len(params) == len(values)
                for parameter, value in zip(params, values):
                    assert parameter.device.type == value.device.type == "cpu"
                    assert parameter.dtype == value.dtype == torch.float32 and parameter.shape == value.shape
                    parameter.copy_(value)
        actual = [[dict(shape=list(p.shape), sha256=hashlib.sha256(p.detach().contiguous().numpy().tobytes()).hexdigest())
                   for p in group] for group in groups]
        assert actual == declaration["fixture"]["initial_parameter_groups"]
        models_cpu = {name: host.digest(model.state_dict()) for name, model in
                      (("G", generator), ("D", critic), ("prior", prior))}
        assert models_cpu == declaration["expected_initial_model_hashes"]
        generator, critic, prior = generator.cuda(), critic.cuda(), prior.cuda()
        stream = torch.Generator(device="cuda:0").manual_seed(0)
        assert_cpu_defaults()
        trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                    latent_generator=torch.Generator(device="cuda:0").manual_seed(1),
                    penalty_generator=torch.Generator(device="cuda:0").manual_seed(2),
                    optimizer_options={"foreach": False, "fused": False})
        assert not trainer.opt_g.state and not trainer.opt_d.state, "released Adam state must start lazy/empty"
        initial = checkpoint()
        validate_checkpoint(initial, identity)
        torch.save(initial, args.output / "initial-state.pt")
        dump(args.output / "initial.json", dict(models_cpu=models_cpu, parameter_groups=actual,
             complete=host.digest(initial), adam_state_empty=True,
             streams={name: host.digest(value) for name, value in initial["trainer"]["streams"].items()},
             data_rng=host.digest(initial["data_rng"])))

        def optimizer_proof():
            result = {}
            for role, opt in (("generator", trainer.opt_g), ("critic", trainer.opt_d)):
                entries = []
                for group in opt.param_groups:
                    assert group["capturable"] is False and group["foreach"] is False and group["fused"] is False
                    for parameter in group["params"]:
                        state = opt.state.get(parameter)
                        assert state, "missing native Adam state after actual update"
                        entries.append(dict(shape=list(parameter.shape), parameter_device=str(parameter.device),
                            step_device=str(state["step"].device), step=float(state["step"]),
                            exp_avg_device=str(state["exp_avg"].device), exp_avg_sq_device=str(state["exp_avg_sq"].device)))
                result[role] = entries
            dump(args.output / f"optimizer-device-proof-{trainer.completed_steps:04d}.json", result)
            assert all(p["step_device"] == "cpu" and p["parameter_device"] == p["exp_avg_device"] == p["exp_avg_sq_device"] == "cuda:0"
                       for entries in result.values() for p in entries), "unexpected Adam placement; no repair permitted"

        @torch.no_grad()
        def measure(ema=False):
            model, table = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
            modes = [(m, m.training) for root in (model, table) for m in root.modules()]
            try:
                model.eval(); table.eval()
                with torch.random.fork_rng(devices=[0]), host_cuda(torch):
                    torch.manual_seed(402)  # Preserved caller namespace; output epsilon explicitly uses2303.
                    latent = table.sample(4096, generator=torch.Generator(device="cuda:0").manual_seed(990))[0]
                    fake = trainer._generate(model, latent, output_noise_std(recipe, trainer.completed_steps),
                                             torch.Generator(device="cuda:0").manual_seed(2303))
                    return host.score_samples(fake, cfg, trainer.completed_steps)
            finally:
                for module, flag in modes:
                    module.training = flag

        def real_batch(step):
            with host_cuda(torch):
                return host.sample_target(cfg, cfg["batch"], stream, step)

        with (args.output / "metrics.jsonl").open("w", buffering=1) as obs, (args.output / "learning-rates.jsonl").open("w", buffering=1) as lr:
            for step in range(1, 1201):
                real = real_batch(step)
                assert_cpu_defaults()
                with serial_step(torch):
                    stats = trainer.step(real, generator_real=lambda: real_batch(step),
                                         collect_stats=step in declaration["evaluation"]["observations"])
                assert_cpu_defaults()
                if step == 1 or step == 1200:
                    optimizer_proof()
                network_scale, prior_scale = learning_rate_scales(step - 1, recipe)
                for optimizer, base, roles in zip((trainer.opt_g, trainer.opt_d), trainer.initial_lrs, trainer.roles):
                    for group, rate, role in zip(optimizer.param_groups, base, roles):
                        assert group["lr"] == rate * (prior_scale if role == "prior" else network_scale)
                rate_row = dict(step=step, **host.rates(trainer), input_noise=input_noise_std(recipe, step - 1),
                                output_noise=output_noise_std(recipe, step - 1),
                                penalty=trainer.penalty.diagnostics())
                lr.write(json.dumps(rate_row, allow_nan=False) + "\n")
                if step in declaration["evaluation"]["observations"]:
                    before = host.digest([trainer.state_dict(), stream.get_state()])
                    point = dict(step=step, **measure(), ema=measure(True), seconds=time.monotonic() - started)
                    assert host.digest([trainer.state_dict(), stream.get_state()]) == before
                    assert_cpu_defaults()
                    observations.append(point); obs.write(json.dumps(point, allow_nan=False) + "\n")
                    print(json.dumps(point), flush=True)
        verdict = host.test_verdict(cfg, dict(live=observations[-1], observations=observations))
        assert verdict["convergence"]["complete"]
        status = "PASS" if verdict["passed"] and verdict["convergence"]["passing_suffix"] >= 5 else "FAIL"
        metrics = dict(verdict=verdict, final=observations[-1], updates=trainer.completed_steps)
        torch.save(checkpoint(), args.output / "final-state.pt")
    except Exception as error:
        status = "ERROR"
        metrics = dict(error=repr(error), traceback=traceback.format_exc(), observations=len(observations))
        if trainer is not None and stream is not None:
            try:
                torch.save(checkpoint(), args.output / "error-state.pt")
            except Exception as checkpoint_error:
                metrics["checkpoint_error"] = repr(checkpoint_error)
    row = dict(candidate=declaration["candidate"], gate=declaration["gate"], status=status,
               seconds=time.monotonic() - started, metrics=metrics, artifact=str(args.output.resolve()))
    dump(args.output / "result.json", row)
    dump(args.output / "artifact-sha256.json", {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in args.output.rglob("*") if p.is_file() and p.name != "artifact-sha256.json"})
    print(json.dumps(row), flush=True)
    if status == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
