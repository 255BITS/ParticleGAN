"""Exact CPU cache/API parity with frozen RA3 and canonical native harness."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
import unittest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT/'pkg-CB64-RA3'
PACKAGE = HERE/'pkg-AXIS-ID'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
HARNESS_SHA = 'ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c'
SEED = 90229
DETAILS = {}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def sources(package):
    return {str(p.relative_to(package)):sha(p) for p in sorted((package/'particlegan').rglob('*.py'))}


def frozen_harness_functions():
    assert sha(HARNESS) == HARNESS_SHA
    parsed = ast.parse(HARNESS.read_text())
    defaults = next(n for n in parsed.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id=='DEFAULT_OPTIONS' for t in n.targets))
    resolve = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name=='resolve_options')
    namespace = dict(inspect=inspect)
    exec(compile(ast.Module(body=[defaults, resolve], type_ignores=[]), str(HARNESS), 'exec'), namespace)
    # Execute the exact argument construction and call from native draw, with
    # supplied fixed tensors/streams. No harness seeding, scoring or training.
    draw = next(n for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef) and n.name=='draw'
                and [a.arg for a in n.args.args]==['n','ema','latent_seed','noise_seed'])
    context = next(n for n in ast.walk(draw) if isinstance(n, ast.With)
                   and any(isinstance(item.context_expr, ast.Call)
                           and isinstance(item.context_expr.func, ast.Attribute)
                           and item.context_expr.func.attr=='fork_rng' for item in n.items))
    nodes = context.body
    start = next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='arguments' for t in n.targets))
    call_nodes = deepcopy(nodes[start:start+3])
    assert isinstance(call_nodes[1], ast.If) and ast.unparse(call_nodes[1].body[0])=='arguments.append(indices)'
    wrapper = ast.parse('def native_call(trainer, model, latent, latent_stream, indices, options):\n    pass\n').body[0]
    wrapper.body = call_nodes+[ast.Return(value=ast.Name(id='clean',ctx=ast.Load()))]
    ast.fix_missing_locations(wrapper)
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(HARNESS),'exec'),namespace)
    return namespace['resolve_options'], namespace['native_call']


def stream(): return torch.Generator(device='cpu').manual_seed(SEED)


def operation_counts(call):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        output = call()
    counts = {e.key:e.count for e in profiler.key_averages()}
    return output, {k:counts.get(k,0) for k in ('aten::item','aten::_local_scalar_dense','aten::nonzero')}


class AxisIdTests(unittest.TestCase):
    def test_01_saved_kernels_match_bits_and_cached_axes_are_ints(self):
        records = []
        for case in helper.INPUTS['cases']:
            points, query, width, noise = (case[k].clone() for k in ('prior','query','bandwidth','noise'))
            # Keep the saved query/noise exactly; no new random input.
            prior = SimpleNamespace(z=points)
            for rank in ((1,8,64,128) if points.shape[1]==128 else (1,8)):
                old = old_fc.BoundedLatentGeometry(rank=rank,chunk=37)
                new = new_fc.BoundedLatentGeometry(rank=rank,chunk=37)
                for _ in range(2):
                    a = old._local_geometry(query,prior); b = new._local_geometry(query,prior)
                    self.assertTrue(torch.equal(a[0],b[0]), (case['name'],rank,'radius'))
                    self.assertTrue(torch.equal(a[1],b[1]), (case['name'],rank,'width'))
                    self.assertTrue(torch.equal(old.displacement(query,prior,width,noise),
                                               new.displacement(query,prior,width,noise)))
                self.assertTrue(all(type(axis) is int for axis,_,_ in new._orders(points)))
                self.assertEqual(old.work,new.work)
                records.append(dict(name=case['name'],rank=rank,cold_and_warm_bit_identical=True))
        DETAILS['saved_kernels'] = records

    def test_02_cache_versions_ties_duplicates_and_lineage_match(self):
        points = torch.nn.Parameter(torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[0.,0.],[.001,.001]]))
        prior = SimpleNamespace(z=points)
        old_graph, new_graph = old_fc.LatentLineage(6,5,'cpu'), new_fc.LatentLineage(6,5,'cpu')
        old_graph.register_copies(torch.tensor([5]),torch.tensor([0]))
        new_graph.register_copies(torch.tensor([5]),torch.tensor([0]))
        old = old_fc.BoundedLatentGeometry(lineage=old_graph,chunk=2)
        new = new_fc.BoundedLatentGeometry(lineage=new_graph,chunk=2)
        rows = torch.arange(6)
        for mutate in ('initial','scaled','one_row','constant'):
            with torch.no_grad():
                if mutate=='scaled': points.mul_(7.)
                elif mutate=='one_row': points[0,0] += .03
                elif mutate=='constant': points.fill_(1.)
            a=old._local_geometry(points,prior,rows=rows);b=new._local_geometry(points,prior,rows=rows)
            self.assertTrue(torch.equal(a[0],b[0]),mutate)
            self.assertTrue(torch.equal(a[1],b[1]),mutate)
            self.assertEqual(old.work,new.work)
        self.assertEqual(new.work['builds'],4)

    def test_03_warm_axis_scalar_reads_removed(self):
        results=[]
        for name,count in (('native_cpu_saved_density',2048),('mnist_E22_0',1024)):
            case=next(c for c in helper.INPUTS['cases'] if c['name']==name)
            points=case['prior']; prior=SimpleNamespace(z=points)
            selection=torch.arange(count)%len(case['query'])
            query,noise=case['query'][selection],case['noise'][selection]
            old,new=old_fc.BoundedLatentGeometry(),new_fc.BoundedLatentGeometry()
            old.displacement(query,prior,case['bandwidth'],noise)
            new.displacement(query,prior,case['bandwidth'],noise)
            a,old_count=operation_counts(lambda:old.displacement(query,prior,case['bandwidth'],noise))
            b,new_count=operation_counts(lambda:new.displacement(query,prior,case['bandwidth'],noise))
            self.assertTrue(torch.equal(a,b))
            self.assertGreater(old_count['aten::_local_scalar_dense'],0)
            self.assertEqual(new_count['aten::_local_scalar_dense'],0)
            results.append(dict(name=name,query_rows=count,reference=old_count,optimized=new_count))
        DETAILS['warm_cpu_axis_counts']=results

    def test_04_frozen_harness_detection_and_positional_call(self):
        resolve, native = frozen_harness_functions()
        old_options,_=resolve(SimpleNamespace(GANTrainer=old_training.GANTrainer),{}, {})
        new_options,detected=resolve(SimpleNamespace(GANTrainer=new_training.GANTrainer),{}, {})
        self.assertEqual(old_options['evaluation_generate'],'plain')
        self.assertEqual(new_options['evaluation_generate'],'indexed')
        trainer=helper.make_trainer(); bd=trainer.birth_death
        bd._move(trainer,torch.tensor([0,1]),torch.tensor([300,301]))
        rows=torch.tensor([0,300,1,301,500]);latent=trainer.prior.z[rows]
        forwarded=[];original=bd.perturb_latent
        def observed(latent,*args,rows=None,**kw):
            forwarded.append(rows)
            return original(latent,*args,rows=rows,**kw)
        bd.perturb_latent=observed
        generated=native(trainer,trainer.G,latent,stream(),rows,new_options)
        self.assertIs(forwarded[-1],rows)
        expected=trainer._generate(trainer.G,latent,0.,stream(),rows=rows)
        self.assertTrue(torch.equal(generated,expected))
        parameter=inspect.signature(new_training.GANTrainer._generate).parameters['indices']
        self.assertEqual(parameter.kind,inspect.Parameter.POSITIONAL_OR_KEYWORD)
        DETAILS['canonical_harness']=dict(source=str(HARNESS),sha256=sha(HARNESS),
            old_auto_mode=old_options['evaluation_generate'],new_auto_mode=new_options['evaluation_generate'],
            fifth_positional_ids_forwarded=True,exact_frozen_argument_fragment_executed=True)

    def test_05_four_arguments_and_rows_alias_preserve_outputs_rng(self):
        trainer=helper.make_trainer();bd=trainer.birth_death
        bd._move(trainer,torch.tensor([0,1]),torch.tensor([300,301]))
        rows=torch.tensor([0,1,300,301,500]);latent=trainer.prior.z[rows]
        for ema in (False,True):
            model,prior=(trainer.ema_G,trainer.ema_prior) if ema else (trainer.G,trainer.prior)
            latent=prior.z[rows]
            for sigma in (0.,.029):
                a,b=stream(),stream()
                old=old_training.GANTrainer._generate(trainer,model,latent,sigma,a)
                new=trainer._generate(model,latent,sigma,b)
                self.assertTrue(torch.equal(old,new));self.assertTrue(torch.equal(a.get_state(),b.get_state()))
                a,b,c=stream(),stream(),stream()
                baseline=old_training.GANTrainer._generate(trainer,model,latent,sigma,a,rows=rows)
                positional=trainer._generate(model,latent,sigma,b,rows)
                alias=trainer._generate(model,latent,sigma,c,rows=rows)
                self.assertTrue(torch.equal(baseline,positional));self.assertTrue(torch.equal(positional,alias))
                self.assertTrue(torch.equal(a.get_state(),b.get_state()));self.assertTrue(torch.equal(b.get_state(),c.get_state()))

    def test_06_trainer_sampling_and_two_update_ra3_state_parity(self):
        trainer=helper.make_trainer();real=helper.real_batch();bd=trainer.birth_death
        bd._move(trainer,torch.tensor([0,1]),torch.tensor([300,301]))
        saved=trainer.state_dict()
        for ema in (False,True):
            sampled=trainer.sample(47,ema=ema,generator=stream())
            model,prior=(trainer.ema_G,trainer.ema_prior) if ema else (trainer.G,trainer.prior)
            expected_stream=stream();latent,rows=prior.sample(47,generator=expected_stream)
            expected=old_training.GANTrainer._generate(trainer,model,latent,0.,expected_stream,rows=rows)
            self.assertTrue(torch.equal(sampled,expected))
        losses,endpoints=[],[]
        for _ in range(2):
            result=trainer.step(real)
            losses.append(helper.digest({k:result[k] for k in ('loss_d','loss_g','loss_gan','prior_regularization','penalty')}))
            endpoints.append(helper.digest(trainer.state_dict()))
        reference=helper.make_trainer();reference.load_state_dict(saved)
        # These are the only changed runtime functions; execute frozen RA3
        # implementations against the identical restored trainer state.
        reference._generate=MethodType(old_training.GANTrainer._generate,reference)
        reference.birth_death.latent_geometry=old_fc.BoundedLatentGeometry(rank=8,neighbors=64,chunk=256,
                                                                          lineage=reference.birth_death.lineage)
        for loss,endpoint in zip(losses,endpoints):
            result=reference.step(real)
            self.assertEqual(loss,helper.digest({k:result[k] for k in ('loss_d','loss_g','loss_gan','prior_regularization','penalty')}))
            self.assertEqual(endpoint,helper.digest(reference.state_dict()))
        DETAILS['state_parity']=dict(reference='frozen RA3 changed-function implementations',
            live_and_ema_sample_bit_identical=True,updates_compared=2,
            loss_and_semantic_state_bit_identical=True,endpoint_sha256=endpoints[-1])

    def test_07_ambiguous_ids_rejected_before_rng_or_state_change(self):
        trainer=helper.make_trainer();rows=torch.arange(3);latent=trainer.prior.z[rows]
        draw=stream();before=draw.get_state().clone();graph=trainer.birth_death.lineage.neighbors.clone()
        with self.assertRaises(ValueError):trainer._generate(trainer.G,latent,0.,draw,rows,rows=rows)
        self.assertTrue(torch.equal(before,draw.get_state()))
        self.assertTrue(torch.equal(graph,trainer.birth_death.lineage.neighbors))


def main():
    global PACKAGE,helper,old_fc,new_fc,old_training,new_training
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=PACKAGE)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();PACKAGE=args.package_root.resolve()
    if args.output.exists():raise RuntimeError('Output evidence already exists')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    base_map,new_map=sources(BASE),sources(PACKAGE)
    assert set(base_map)==set(new_map)
    assert {k for k in base_map if base_map[k]!=new_map[k]}=={'particlegan/feature_cells.py','particlegan/training.py'}
    helper=load('axis_lineage_fixtures',ROOT/'geometry/training-regression/lineage_checks.py')
    helper.setup(PACKAGE,'cpu')
    import particlegan.feature_cells as new_fc
    import particlegan.training as new_training
    old_fc=load('particlegan._axis_ra3_fc',BASE/'particlegan/feature_cells.py')
    old_training=load('particlegan._axis_ra3_training',BASE/'particlegan/training.py')
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AxisIdTests))
    assert sources(BASE)==base_map and sources(PACKAGE)==new_map
    assert not torch.cuda.is_initialized()
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),seconds=time.perf_counter()-started,
        seed=SEED,new_seeds=0,cpu_threads=1,cuda_initialized=False,
        scope='exact cache/API/RNG/state parity; no quality verdict',
        source_sha256=new_map,base_source_sha256=base_map,script_sha256=sha(__file__),
        harness_sha256=sha(HARNESS),details=DETAILS,
        failure_details=[dict(test=str(t),traceback=tb) for t,tb in result.failures+result.errors])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],tests=result.testsRun,output=str(args.output))),flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
