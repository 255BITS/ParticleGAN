"""Frozen actual-positive-lease sample/reload contract; CUDA execution is root-only."""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

AREA = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
LANE = ROOT / 'validation-cb64-ra8/learned'
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
CHECKPOINT = LANE / 'training/toy/CB64-RA8/checkpoint-2000.pt'
RUN_RECEIPT = CHECKPOINT.parent / 'config.json'
GPU_OUTPUT = ROOT / 'integration/review/ra8-positive-lease-gpu/result.json'
VARIANT = 'CB64-RA8'
CHUNK = 256
CHUNKS = 2
COHERENT_ROWS = 977


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_frozen(package_root):
    freeze_path = AREA / 'SOURCE-FROZEN.json'
    freeze = json.loads(freeze_path.read_text())
    for name, expected in freeze['file_sha256'].items():
        require(sha(name) == expected, f'frozen source/input changed: {name}')
    inputs = json.loads((AREA / 'INPUTS.json').read_text())
    require(str(Path(package_root).resolve()) == inputs['package_root'], 'package path differs')
    package = Path(package_root) / 'particlegan'
    sources = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*.py'))}
    require(sources == inputs['package_source_sha256'], 'whole package source map differs')
    h = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        h.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes() + b'\0')
    require(h.hexdigest() == inputs['package_sha256'], 'whole package digest differs')
    return inputs


def validate_saved(torch, checkpoint, inputs):
    state = checkpoint['trainer']
    require(state['schema'] == 5 and state['device'] == 'cuda:0', 'wrong trainer schema/device')
    require(state['completed_steps'] == 2000 and state['serial_backward'] is True, 'wrong saved cursor/mode')
    require(checkpoint['data_position'] == 2 * 2000 * 128, 'wrong real-stream position')
    require(checkpoint['receipt_sha256'] == sha(RUN_RECEIPT), 'run receipt hash differs')
    receipt = json.loads(RUN_RECEIPT.read_text())
    require(receipt['package']['package_root'] == inputs['package_root'], 'saved package path differs')
    require(receipt['package']['package_sha256'] == inputs['package_sha256'], 'saved package digest differs')
    require(receipt['package']['source_sha256'] == inputs['package_source_sha256'], 'saved source map differs')
    require(receipt['source_freeze_sha256'] == sha(LANE / 'SOURCE-FREEZE.json'), 'saved lane freeze differs')
    require(receipt['seed'] == 314159, 'original seed differs')
    require(state['recipe']['serve_average'] == 4.0, 'saved average law differs')
    require(state['recipe']['output_noise_mode'] == 'learnable'
            and state['recipe']['output_noise_std'] == .029, 'saved original noise law differs')
    require(state['recipe']['num_particles'] == 1024 and state['recipe']['z_dim'] == 128,
            'saved table dimensions differ')
    bd = state['birth_death']
    require(bd['backend'] == 'feature_cells' and bd['backend_schema'] == 7, 'wrong backend schema')
    stamp = bd['paired_average']
    require(stamp['schema'] == 1 and stamp['eligible'] is True, 'actual saved positive stamp is required')
    require(stamp['coherent_rows'] == COHERENT_ROWS and stamp['required'] == 973
            and stamp['rows'] == 1024, 'actual saved977/973 stamp differs')
    require(stamp == bd['last']['paired_average'], 'last/stamp consistency differs')
    require(stamp['snapshot'] == bd['snapshot_serial'] and stamp['step'] <= 2000,
            'actual stamp snapshot/step differs')
    require(bd['fill'] == 1024 and 0 <= bd['rows_since_eval'] < 1024, 'actual saved lease expired')
    rng = {'cpu_rng': state['cpu_rng'], 'cuda_rng': state['cuda_rng'],
           **{f'streams.{name}': value for name, value in state['streams'].items()},
           'birth_death.stream': bd['stream']}
    require(set(state['streams']) == {'latent_generator', 'penalty_generator', 'eval_generator', 'noise_generator'},
            'saved private stream names differ')
    for name, value in rng.items():
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == torch.uint8, f'RNG buffer placement differs: {name}')
    return state, stamp, {name: dict(device=str(value.device), dtype=str(value.dtype), numel=value.numel())
                          for name, value in rng.items()}


def original_digest(torch):
    # Execute only the exact typed fingerprint function from the unchanged
    # original replay. Its imports/main and CUDA setup are never executed here.
    source = ast.parse((LANE / 'replay.py').read_text())
    node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'digest')
    namespace = dict(torch=torch, hashlib=hashlib, struct=struct)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LANE / 'replay.py'), 'exec'), namespace)
    return namespace['digest']


def cpu_preflight(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    inputs = verify_frozen(args.package_root)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU metadata preflight must not initialize CUDA')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state, stamp, rng = validate_saved(torch, checkpoint, inputs)
    for path in (AREA / 'sample_reload.py', AREA / 'prepare_contract.py', LANE / 'common.py', LANE / 'replay.py'):
        compile(path.read_text(), str(path), 'exec')
    # Check the original draw API and original scorer stream constants without
    # importing the CUDA harness or calling its evaluator.
    scorer = ast.parse((PREV / 'models_metrics.py').read_text())
    draw = next(node for node in scorer.body if isinstance(node, ast.FunctionDef) and node.name == 'draw_samples')
    require('min(256, n - i)' in ast.unparse(draw) and 'range(0, n, 256)' in ast.unparse(draw),
            'original scorer chunk law differs')
    training = ast.parse((LANE / 'run_training.py').read_text())
    require('toy_metrics(trainer, SEED + 100)' in ast.unparse(training), 'original toy scorer seed expression differs')
    verify_frozen(args.package_root)
    require(not torch.cuda.is_initialized(), 'CPU metadata preflight created a CUDA context')
    result = dict(status='PASS_CPU_METADATA_PREFLIGHT', source_freeze_sha256=sha(AREA / 'SOURCE-FROZEN.json'),
                  checkpoint_sha256=sha(CHECKPOINT), actual_paired_average=stamp,
                  rng_buffer_placement=rng, completed_steps=state['completed_steps'],
                  original_scorer_seed=314259, original_chunk_rows=CHUNK, chunks_per_branch=CHUNKS,
                  independent_initial_restores=2, intermediate_reloads=1, training_updates=0,
                  model_constructions=0, model_forwards=0, generated_samples=0,
                  cuda_initialized=False, numerical_cuda_contract='NOT_RUN',
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv])
    write_exclusive(args.output, result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


def served_view(trainer, saved, digest):
    require(trainer._fast is not None and trainer._serve_settled(), 'actual positive lease did not apply serving')
    live = trainer._served_parameters()
    averages = trainer._served_averages()
    fast = [saved['models'][root][name] for root in ('G', 'prior')
            for name, _ in getattr(trainer, root).named_parameters()]
    require(len(live) == len(averages) == len(fast) == len(trainer._fast), 'serving parameter shape differs')
    require(digest(live) == digest(averages), 'live serving parameters are not paired averages')
    require(digest(trainer._fast) == digest(fast), 'retained FAST parameters differ from the saved state')
    require(digest(fast) != digest(averages), 'contract would not exercise a distinct served average')
    return dict(positive_lease=True, retained_fast_matches_saved=True,
                live_matches_paired_average=True, fast_differs_from_average=True)


def nonsemantic_state(trainer, digest):
    roots = ('G', 'D', 'prior', 'ema_G', 'ema_prior')
    return dict(modes={root: {name: module.training for name, module in getattr(trainer, root).named_modules()}
                       for root in roots},
                grads={root: {name: None if parameter.grad is None else digest(parameter.grad)
                              for name, parameter in getattr(trainer, root).named_parameters()}
                       for root in roots},
                buffers={root: digest({name: value for name, value in getattr(trainer, root).named_buffers()})
                         for root in roots})


def cache_receipt(trainer, digest):
    geometry = trainer.birth_death.latent_geometry
    entry = geometry._entries.get(id(trainer.prior.z))
    require(entry is not None and entry[0]() is trainer.prior.z and entry[1] == trainer.prior.z._version,
            'sampling did not build a valid served-prior axis cache')
    builds = geometry.work['builds']
    first = geometry._orders(trainer.prior.z)
    second = geometry._orders(trainer.prior.z)
    require(first is second and first is entry[2] and geometry.work['builds'] == builds,
            'version-matched cache accesses did not stay warm')
    require(trainer.birth_death.snapshot is None and trainer.birth_death._heads is None,
            'sampling unexpectedly reconstructed the learned chart/head')
    require(geometry.work['max_query_rows'] <= CHUNK
            and geometry.work['max_candidates'] <= geometry.neighbors + trainer.birth_death.lineage.degree,
            'sampling query work exceeded its original bound')
    return dict(axis_orders_sha256=digest(first), orders=len(first), warm_reuse=True,
                current_parameter_version_matches=True, chart_absent=True, heads_absent=True,
                work=dict(geometry.work))


def traced_sample(torch, trainer, stream, digest):
    trace = {}
    generate = trainer._generate
    perturb = trainer.birth_death.perturb_latent
    modes_grads_buffers = nonsemantic_state(trainer, digest)
    stream_before = stream.get_state().clone()
    globals_before = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(0).clone())

    def capture_perturb(latent, local_stream, controller=None, record=False, *, prior=None, rows=None):
        require(local_stream is stream and prior is trainer.prior and record is False,
                'sample latent perturbation deviated from the original evaluation path')
        output = perturb(latent, local_stream, controller, record, prior=prior, rows=rows)
        trace['perturbed_latent'] = output.detach().cpu().clone()
        return output

    def capture_generate(model, latent, sigma, local_stream, indices=None, *, rows=None):
        require(model is trainer.G and local_stream is stream and indices is None and rows is not None,
                'sample used an override instead of the ordinary served path')
        require(math.isfinite(float(sigma)) and float(sigma) > 0, 'original output noise must remain positive')
        trace.update(rows=rows.detach().cpu().clone(), latent=latent.detach().cpu().clone(), sigma=float(sigma))
        output = generate(model, latent, sigma, local_stream, rows=rows)
        trace['noisy_output'] = output.detach().cpu().clone()
        return output

    trainer._generate = capture_generate
    trainer.birth_death.perturb_latent = capture_perturb
    try:
        output = trainer.sample(CHUNK, generator=stream)
        require(digest(output.detach().cpu()) == digest(trace['noisy_output']), '_generate result differs from sample')
    finally:
        del trainer._generate
        del trainer.birth_death.perturb_latent
    trace['scorer_stream_before'] = stream_before
    trace['scorer_stream_after'] = stream.get_state().clone()
    require(digest(stream_before) != digest(trace['scorer_stream_after']), 'scorer stream failed to advance')
    globals_after = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(0))
    require(digest(globals_before) == digest(globals_after), 'sampling changed global RNG continuation state')
    require(nonsemantic_state(trainer, digest) == modes_grads_buffers, 'sampling changed modes/gradients/buffers')
    require(trace['rows'].shape == (CHUNK,) and trace['rows'].dtype == torch.int64
            and bool(((trace['rows'] >= 0) & (trace['rows'] < 1024)).all()), 'sampled row IDs are invalid')
    ema_rows = trainer.ema_prior.z.detach().cpu()[trace['rows']]
    require(digest(ema_rows) == digest(trace['latent']), 'incoming sample codes differ from their saved EMA rows')
    require(all(bool(torch.isfinite(trace[key]).all()) for key in ('latent', 'perturbed_latent', 'noisy_output')),
            'sample trace contains a nonfinite value')
    return trace, cache_receipt(trainer, digest)


def cuda_contract(args):
    require(Path(args.output).resolve() == GPU_OUTPUT, 'root CUDA output path differs')
    require(not Path(args.output).exists(), 'contract result already exists; preserve that attempt')
    inputs = verify_frozen(args.package_root)
    # Original common configures the original CUDA visibility before torch or
    # selected-package imports. CPU preflight never enters this function.
    sys.path.insert(0, str(LANE))
    import common
    sys.path.insert(0, str(PREV))
    import torch
    from models_metrics import networks
    with common.exclusive_learned_gpu():
        lane_inputs = common.verify_inputs()
        selected = common.select_package(lane_inputs, VARIANT)
        require(selected['package_root'] == inputs['package_root'], 'original lane selected another package')
        from particlegan.recipes import Recipe
        from particlegan.particle_prior import ParticlePrior
        from particlegan.training import GANTrainer
        runtime = common.configure_cuda(torch)
        digest = original_digest(torch)
        # Preserve native CUDA model tensors and CPU RNG buffers. No blanket
        # map_location is allowed in either original or intermediate load.
        checkpoint = torch.load(CHECKPOINT, weights_only=False)
        saved, stamp, rng_placement = validate_saved(torch, checkpoint, inputs)
        saved_digest = digest(saved)
        saved_sections = {name: digest(value) for name, value in saved.items()}
        branches = []
        started = time.perf_counter()
        for branch in range(2):
            G, D = networks('toy')
            prior = ParticlePrior(1024, 128)
            trainer = GANTrainer(Recipe(**saved['recipe']), G.to(common.DEVICE), D.to(common.DEVICE),
                                 prior=prior.to(common.DEVICE), seed=common.SEED, serial_backward=True)
            trainer.load_state_dict(saved)
            require(trainer.birth_death.snapshot is None and trainer.birth_death._heads is None
                    and not trainer.birth_death.latent_geometry._entries, 'load did not discard derived caches')
            serving = served_view(trainer, saved, digest)
            restored = trainer.state_dict()
            require(digest(restored) == saved_digest, f'branch{branch} initial restore differs')
            common.rng_cpu_buffers(torch, restored)
            stream = torch.Generator(device=trainer.device).manual_seed(common.SEED + 100)
            traces, caches, endpoint_sections = [], [], []
            for chunk in range(CHUNKS):
                trace, cache = traced_sample(torch, trainer, stream, digest)
                state = trainer.state_dict()
                common.rng_cpu_buffers(torch, state)
                sections = {name: digest(value) for name, value in state.items()}
                require(sections == saved_sections and digest(state) == saved_digest,
                        f'branch{branch} chunk{chunk} changed serialized training or RNG state')
                served_view(trainer, saved, digest)
                traces.append(trace); caches.append(cache); endpoint_sections.append(sections)
                common.log('positive_lease_chunk', branch=branch, chunk=chunk, rows=CHUNK,
                           trace_sha256=digest(trace), state_sha256=digest(state), sigma=trace['sigma'])
                if branch == 1 and chunk == 0:
                    resume_path = Path(args.output).parent / 'intermediate-reload.pt'
                    require(not resume_path.exists(), 'intermediate evidence already exists')
                    torch.save(dict(trainer=state, scorer_stream=stream.get_state().clone(),
                                    source_checkpoint_sha256=sha(CHECKPOINT)), resume_path)
                    resumed = torch.load(resume_path, weights_only=False)
                    require(digest(resumed['trainer']) == digest(state)
                            and digest(resumed['scorer_stream']) == digest(stream.get_state()),
                            'saved intermediate roundtrip differs')
                    trainer.load_state_dict(resumed['trainer'])
                    stream.set_state(resumed['scorer_stream'])
                    require(trainer.birth_death.snapshot is None and trainer.birth_death._heads is None
                            and not trainer.birth_death.latent_geometry._entries, 'reload did not discard derived caches')
                    require(digest(trainer.state_dict()) == saved_digest, 'intermediate trainer reload differs')
                    served_view(trainer, saved, digest)
            endpoint = Path(args.output).parent / f'branch-{branch}.pt'
            require(not endpoint.exists(), 'branch evidence already exists')
            torch.save(dict(trainer=state, sample_traces=traces, scorer_stream=stream.get_state().clone(),
                            source_checkpoint_sha256=sha(CHECKPOINT)), endpoint)
            branches.append(dict(branch=branch, initial_restore_exact=True, serving=serving,
                                 intermediate_reload=branch == 1, traces=[digest(trace) for trace in traces],
                                 trace_components=[{name: digest(value) for name, value in trace.items()} for trace in traces],
                                 caches=caches, state_sections=endpoint_sections,
                                 endpoint=str(endpoint), endpoint_sha256=sha(endpoint),
                                 final_scorer_stream_sha256=digest(stream.get_state())))
            del trainer, G, D, prior, restored, state, trace, traces, stream
            if branch == 1:
                del resumed
            gc.collect(); torch.cuda.empty_cache()
        require(branches[0]['traces'] == branches[1]['traces'], 'sample traces differ after reload')
        require(branches[0]['state_sections'] == branches[1]['state_sections'], 'serialized state differs across branches')
        require(branches[0]['final_scorer_stream_sha256'] == branches[1]['final_scorer_stream_sha256'],
                'scorer continuation state differs')
        for chunk in range(CHUNKS):
            require(branches[0]['caches'][chunk]['axis_orders_sha256'] == branches[1]['caches'][chunk]['axis_orders_sha256'],
                    'rebuilt axis cache differs after reload')
        torch.cuda.synchronize(0)
        verify_frozen(args.package_root)
        result = dict(status='PASS', source_freeze_sha256=sha(AREA / 'SOURCE-FROZEN.json'),
                      checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
                      package_root=inputs['package_root'], package_sha256=inputs['package_sha256'],
                      actual_paired_average=stamp, actual_saved_coherent_rows=COHERENT_ROWS,
                      original_scorer_seed=common.SEED + 100, chunk_rows=CHUNK, chunks_per_branch=CHUNKS,
                      generated_rows_per_branch=CHUNK * CHUNKS, independent_initial_restores=2,
                      intermediate_reloads=1, training_updates=0, generator_forwards=2 * CHUNKS,
                      critic_forwards=0, learned_feature_passes=0, quality_metric_evaluations=0,
                      full_state_bit_identical=True, sampled_rows_bit_identical=True,
                      latent_codes_bit_identical=True, perturbed_codes_bit_identical=True,
                      noisy_outputs_bit_identical=True, scorer_rng_continuation_bit_identical=True,
                      saved_private_and_global_rng_unchanged=True, original_noise_law_unchanged=True,
                      positive_lease_cache_rebuild_exact=True, state_excluded_fields=[],
                      rng_buffer_placement=rng_placement, branches=branches,
                      runtime=runtime, elapsed_seconds=time.perf_counter() - started,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0),
                      command=[sys.executable, *sys.argv], finished_utc=common.utc_now(),
                      scope='actual saved positive-lease sample/reload mechanics; no training or quality verdict')
        write_exclusive(args.output, result)
        common.log('positive_lease_contract', status='PASS', rows_per_branch=CHUNK * CHUNKS,
                   generated_total=2 * CHUNK * CHUNKS, actual_coherent_rows=COHERENT_ROWS,
                   elapsed_seconds=result['elapsed_seconds'])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        return cpu_preflight(args) if args.cpu_preflight else cuda_contract(args)
    except Exception as error:
        failure = dict(status='ERROR', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), command=[sys.executable, *sys.argv],
                       cpu_preflight=args.cpu_preflight, finished_utc=datetime.now(timezone.utc).isoformat())
        if not Path(args.output).exists():
            write_exclusive(args.output, failure)
        print(json.dumps(failure, allow_nan=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
