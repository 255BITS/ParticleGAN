"""Sealed public K3P v0.8.0 recovery-ring reference. No training on import."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
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
    args.output = args.output.resolve()
    if args.output == BUNDLE or BUNDLE in args.output.parents:
        raise ValueError("output must be outside the sealed bundle")
    from preflight import verify
    verification = verify(BUNDLE)
    declaration = json.loads((BUNDLE / "declaration.json").read_text())
    manifest = json.loads((BUNDLE / "bundle-sha256.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = dict(manifest["files"])
    source_hashes["bundle-sha256.json"] = hashlib.sha256((BUNDLE / "bundle-sha256.json").read_bytes()).hexdigest()
    dump(args.output / "declaration.json", {**declaration, "status": "DECLARED_BEFORE_EXECUTION",
         "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes})
    dump(args.output / "source-preflight.json", verification)
    with zipfile.ZipFile(args.output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes:
            archive.write(BUNDLE / name, name)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    trainer = stream = means = frozen = None
    started = time.monotonic()
    observations = []
    torch = host = None
    identity = {"bundle_manifest_sha256": source_hashes["bundle-sha256.json"],
                "recipe": declaration["recipe"], "host": declaration["host"],
                "evaluation": declaration["evaluation"]}

    try:
        if any(n == "particlegan" or n.startswith("particlegan.") for n in sys.modules):
            raise RuntimeError("fresh process required: ParticleGAN is already imported")
        sys.path.insert(0, str(BUNDLE / "source"))
        import torch
        from particlegan import GANTrainer, get_recipe
        from particlegan.training import input_noise_std, output_noise_std
        from particlegan.recipes import learning_rate_scales
        import frozen_host as host
        from execution_contract import EXECUTION, serial_step, validate_checkpoint

        original_generator = torch.Generator

        def assert_public_imports():
            imported = {}
            for name, module in tuple(sys.modules.items()):
                if name == "particlegan" or name.startswith("particlegan."):
                    path = Path(module.__file__).resolve()
                    relative = str(path.relative_to(BUNDLE / "source"))
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    assert actual == declaration["release"]["source_sha256"][relative]
                    imported[name] = {"path": str(path), "sha256": actual}
            assert Path(host.__file__).resolve() == BUNDLE / "source/frozen_host.py"
            assert not any(n.startswith(("benchmarks.", "particlegan.ka2", "particlegan.precision", "particlegan.game_update")) for n in sys.modules)
            return imported

        def assert_defaults():
            assert str(torch.get_default_device()) == "cpu"
            assert torch.Generator is original_generator
            assert torch.get_default_dtype() == torch.float32

        assert_defaults()
        assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
        expected = declaration["runtime_expected"]
        assert str(torch.__version__) == expected["torch"]
        assert torch.version.cuda == expected["cuda"]
        assert torch.cuda.get_device_name(0) == expected["gpu"]
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == expected["cublas_workspace"]
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ[name] == "1", name
        torch_root = Path(torch.__file__).resolve().parent
        runtime_sources = {}
        for name, wanted in expected["source_hashes"].items():
            path = torch_root / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == wanted, "pinned Torch source differs: " + name
            runtime_sources[name] = {"path": str(path), "sha256": actual}
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        runtime = dict(torch=str(torch.__version__), torch_revision=torch.version.git_version,
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), executable=sys.executable,
            deterministic=True, tf32=False, threads=1, interop_threads=1,
            imported_package=assert_public_imports(), pinned_torch_sources=runtime_sources,
            default_factory_device="cpu", execution=EXECUTION,
            environment={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        dump(args.output / "runtime.json", runtime)
        recipe = get_recipe(total_steps=4600)
        assert json.loads(json.dumps(recipe.to_dict())) == declaration["recipe"]
        assert input_noise_std(recipe, 460) == 0 and output_noise_std(recipe, 920) == .029

        def make_trainer():
            # Exact original ring order: each network initializes on CPU before its move.
            assert_defaults()
            torch.manual_seed(0)
            generator = host.SimpleMLPGenerator(recipe.z_dim, 96, 3, 2).to("cuda:0")
            critic = host.SimpleMLPDiscriminator(2, 96, 3, 3).to("cuda:0")
            value = GANTrainer(recipe, generator, critic, seed=0,
                               optimizer_options={"foreach": False, "fused": False})
            assert not value.opt_g.state and not value.opt_d.state, "release Adam must begin lazy/empty"
            return value

        trainer = make_trainer()
        stream = torch.Generator(device="cuda:0").manual_seed(0)
        means = host.ring_means().to("cuda:0")
        original_means = means.clone()

        def envelope(value=trainer):
            result = dict(schema=1, execution=EXECUTION, identity=identity,
                          trainer=value.state_dict(), data_rng=stream.get_state(), means=means.clone())
            validate_checkpoint(result, identity)
            return result

        def save_checkpoint(name):
            value = envelope()
            torch.save(value, args.output / name)
            return value

        def optimizer_proof(value, label):
            proof = {}
            for role, optimizer in (("generator", value.opt_g), ("critic", value.opt_d)):
                entries = []
                for group in optimizer.param_groups:
                    assert group["capturable"] is False and group["foreach"] is False and group["fused"] is False
                    for parameter in group["params"]:
                        state = optimizer.state.get(parameter)
                        assert state is not None and {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                        entries.append(dict(shape=list(parameter.shape), parameter_device=str(parameter.device),
                            dtype=str(parameter.dtype), gradient_device=None if parameter.grad is None else str(parameter.grad.device),
                            gradient_dtype=None if parameter.grad is None else str(parameter.grad.dtype),
                            step_device=str(state["step"].device), step_dtype=str(state["step"].dtype), step_shape=list(state["step"].shape), step=float(state["step"]),
                            exp_avg_device=str(state["exp_avg"].device), exp_avg_sq_device=str(state["exp_avg_sq"].device),
                            exp_avg_dtype=str(state["exp_avg"].dtype), exp_avg_sq_dtype=str(state["exp_avg_sq"].dtype)))
                proof[role] = entries
            dump(args.output / ("optimizer-device-proof-" + label + ".json"), proof)
            assert all(r["step_device"] == "cpu" and r["step_shape"] == [] and r["step"] == value.completed_steps
                       and r["parameter_device"] == r["exp_avg_device"] == r["exp_avg_sq_device"] == "cuda:0"
                       and r["gradient_device"] in (None, "cuda:0") and r["gradient_dtype"] in (None, "torch.float32")
                       and r["dtype"] == r["exp_avg_dtype"] == r["exp_avg_sq_dtype"] == r["step_dtype"] == "torch.float32"
                       for entries in proof.values() for r in entries), "unexpected native Adam placement; no relocation or repair allowed"

        initial = host.state_receipt(trainer, stream, means)
        expected_initial = declaration["expected_initial_fixture"]
        matches = {k:initial[k] == wanted for k,wanted in expected_initial.items()}
        dump(args.output / "initial.json", initial)
        dump(args.output / "initialization-audit.json", dict(matches=matches, adam_state_empty=True,
             scope=declaration["initial_comparison_scope"]))
        save_checkpoint("initial-state.pt")
        torch.save({name:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    for name,model in (("G",trainer.G),("D",trainer.D),("prior",trainer.prior))}, args.output / "initial-models-cpu.pt")
        assert all(matches.values()), "public ring initialization/RNG mismatch; no repair or seed retry"

        @torch.no_grad()
        def measure(value, ema=False):
            isolated = torch.Generator(device="cuda:0").manual_seed(9)
            score = host.diversity(value.sample(4096, ema=ema, generator=isolated), means)
            assert 0 <= score["hq"] <= 1 and 0 <= score["modes"] <= 8
            return score

        def own_frozen_digest():
            state = frozen.state_dict()
            # Process-global RNG belongs to the running process, not the frozen model.
            return host.digest({k:v for k,v in state.items() if k not in ("cpu_rng", "cuda_rng")})

        frozen_digest = None
        with (args.output / "metrics.jsonl").open("w", buffering=1) as obs, \
             (args.output / "learning-rates.jsonl").open("w", buffering=1) as lr, \
             (args.output / "state-hashes.jsonl").open("w", buffering=1) as states:
            for step in range(1, 4601):
                indices = torch.randint(0, 8, (2048,), device="cuda:0", generator=stream)
                real = means[indices] + .07 * torch.randn(2048, 2, device="cuda:0", generator=stream)
                assert_defaults()
                with serial_step(torch):
                    stats = trainer.step(real, generator_real=real, collect_stats=step % 10 == 0)
                assert_defaults()
                assert trainer.completed_steps == step
                if step in (1, 4600):
                    optimizer_proof(trainer, f"main-{step:04d}")
                network, prior_scale = learning_rate_scales(step - 1, recipe)
                for optimizer, base_rates, roles in zip((trainer.opt_g,trainer.opt_d),trainer.initial_lrs,trainer.roles):
                    for group,rate,role in zip(optimizer.param_groups,base_rates,roles):
                        assert group["lr"] == rate * (prior_scale if role == "prior" else network)
                rate_row = dict(step=step, **host.rates(trainer), input_noise=input_noise_std(recipe,step-1),
                                output_noise=output_noise_std(recipe,step-1), penalty=trainer.penalty.diagnostics(),
                                accepted_updates=step, field_evaluations=1)
                lr.write(json.dumps(rate_row, allow_nan=False) + "\n")
                if step % 10 == 0:
                    before = host.digest(envelope())
                    point = dict(step=step, **measure(trainer), ema=measure(trainer,True),
                                 learning_rates=host.rates(trainer), penalty=trainer.penalty.diagnostics(),
                                 input_noise=input_noise_std(recipe,step-1), output_noise=output_noise_std(recipe,step-1),
                                 evaluation_output_noise=output_noise_std(recipe,step),
                                 losses={k:float(v) for k,v in stats.items() if isinstance(v,torch.Tensor)},
                                 seconds=time.monotonic()-started)
                    if frozen is not None:
                        assert frozen.completed_steps == 2400
                        point["frozen"] = dict(step=step, **measure(frozen),
                                               evaluation_output_noise=output_noise_std(recipe,frozen.completed_steps))
                        assert own_frozen_digest() == frozen_digest
                    assert host.digest(envelope()) == before, "observation changed main learner/caller state"
                    observations.append(point)
                    obs.write(json.dumps(point, allow_nan=False) + "\n")
                    if step % 100 == 0:
                        states.write(json.dumps(host.state_receipt(trainer,stream,means), allow_nan=False) + "\n")
                        print(json.dumps(point, allow_nan=False), flush=True)
                if step == 2400:
                    saved = save_checkpoint("change-2400-state.pt")
                    before_control = host.digest(envelope())
                    frozen = make_trainer()
                    validate_checkpoint(saved, identity)
                    frozen.load_state_dict(saved["trainer"])
                    assert host.digest(envelope()) == before_control, "frozen construction/load changed main state"
                    assert host.digest(frozen.state_dict()) == host.digest(saved["trainer"])
                    optimizer_proof(frozen, "frozen-2400")
                    frozen_digest = own_frozen_digest()
                    means.copy_(original_means + means.new_tensor([1.,0.]))
                    print(json.dumps(dict(event="shift", after_step=2400, absolute_offset=[1.,0.])), flush=True)
        assert len(observations) == 460
        assert_public_imports()
        save_checkpoint("final-state.pt")
        segments = [host.segment(observations,0,2400), host.segment(observations,2400,4600)]
        controls = [p["frozen"] for p in observations if p["step"] > 2400]
        assert len(controls) == 220
        metrics = dict(segments=segments, comparison3600=host.segment(observations,2400,3600),
                       frozen_control=dict(checkpoint=2400,updates_after_copy=0,passing=sum(map(host.good,controls)),
                                           observations=len(controls),maximum_hq=max(p["hq"] for p in controls)),
                       completed_updates=trainer.completed_steps,observations=len(observations),
                       final=observations[-1],rate_rows=4600,source_and_runtime_assertions_passed=True)
        dump(args.output / "summary.json", metrics)
        status = "COMPLETE"
    except Exception as error:
        status = "ERROR"
        metrics = dict(error=repr(error),traceback=traceback.format_exc(),observations=len(observations))
        if torch is not None and trainer is not None and stream is not None and means is not None:
            try:
                torch.save(envelope(), args.output / "error-state.pt")
            except Exception as checkpoint_error:
                metrics["checkpoint_error"] = repr(checkpoint_error)
    result = dict(candidate=declaration["candidate"],gate=declaration["gate"],status=status,
                  status_scope="measurement completeness; supervisor assesses arrival and retention, no automatic winner claim",
                  seconds=time.monotonic()-started,metrics=metrics,artifact=str(args.output))
    dump(args.output / "result.json", result)
    dump(args.output / "artifact-sha256.json", {str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in args.output.rglob("*") if p.is_file() and p.name != "artifact-sha256.json"})
    print(json.dumps(result,allow_nan=False), flush=True)
    if status == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
