"""Focused backend contracts, independent of acceptance/quality harnesses."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'pkg-CB64-RA'))
from particlegan import Recipe,ParticlePrior,GANTrainer,FeatureCellSnapshot,FeatureCellBirthDeath
from particlegan.feature_cells import conditional_count_pvalues,bounded_jitter
from scipy.stats import fisher_exact

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
SEED=90229


def stream():
    return torch.Generator().manual_seed(SEED)


def recipe(n=256,backend='feature_cells'):
    options=json.loads((ROOT/'configs/overrides-CB64-RA.json').read_text())
    options.update(num_particles=n,z_dim=2,initialization=None,output_noise_mode='fixed',
                   output_noise_std=0.,serve_average=0.)
    if backend=='knn':
        for key in list(options):
            if key in ('birth_death_backend','birth_death_cells','birth_death_metric_rank',
                       'birth_death_chunk','birth_death_parent_policy'):
                options.pop(key)
    return Recipe(**options)


def make_trainer(n=256,backend='feature_cells'):
    torch.manual_seed(SEED)
    G=torch.nn.Linear(2,2)
    with torch.no_grad():
        G.weight.copy_(torch.eye(2));G.bias.zero_()
    D=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    # Saturated learned features keep this discrete state/replay fixture's
    # two support atoms stable through gradient updates. This tests row/state
    # mechanics, not efficacy or a useful trained critic.
    with torch.no_grad():
        D[0].weight[:,0]=torch.where(D[0].weight[:,0]>=0,100.,-100.)
        D[0].weight[:,1]=0.;D[0].bias.zero_()
    prior=ParticlePrior(n,2,init_std=0.,generator=stream())
    with torch.no_grad():
        prior.z[:,0]=3.;prior.z[:n//10,0]=-3.
    return GANTrainer(recipe(n,backend),G,D,prior=prior,seed=SEED,serial_backward=True)


def real_batch(n=256):
    real=torch.zeros(n,2)
    real[:,0]=-3.;real[:n//10,0]=3.
    return real[torch.randperm(n,generator=stream())]


def same_state(a,b):
    if isinstance(a,torch.Tensor):
        # The legacy stationarity tester records NaN diagnostic blocks for
        # exactly zero gradients. Compare their bits too, rather than treating
        # identical checkpoint payloads as unequal via NaN!=NaN.
        return (isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype
                and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                b.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(same_state(a[k],b[k]) for k in a if k!='eval_seconds')
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(same_state(x,y) for x,y in zip(a,b))
    return a==b


class Integration(unittest.TestCase):
    def test_exact_count_tests(self):
        for r,f,m,n in ((3,12,20,30),(0,17,20,30),(10,15,20,30),(20,30,20,30)):
            rc=torch.tensor([r,m-r]);fc=torch.tensor([f,n-f])
            actual=conditional_count_pvalues(rc,fc,m,n)
            expected=float(fisher_exact([[r,m-r],[f,n-f]]).pvalue)
            self.assertAlmostEqual(float(actual[0]),expected,places=10)
            self.assertAlmostEqual(float(actual[1]),expected,places=10)

    def test_rank_zero_tiny_and_invalid_inputs(self):
        real=torch.zeros(6,8,dtype=torch.float64)
        snap=FeatureCellSnapshot.fit(real,generator=stream())
        flags,p,_=snap.support(real)
        self.assertEqual(snap.rank,0);self.assertFalse(bool(flags.any()));self.assertTrue(bool((p==1).all()))
        comparison=snap.cell_comparison(real)
        child,parent,detail=snap.ordinary_transport(real,flags,comparison,generator=stream(),pvalues=p)
        self.assertEqual(len(child),0);self.assertEqual(detail['budget'],0)
        varied=torch.zeros(64,8,dtype=torch.float64)
        varied[:,:2]=torch.randn((64,2),generator=stream(),dtype=torch.float64)
        snap=FeatureCellSnapshot.fit(varied,generator=stream())
        self.assertEqual(snap.rank,2)
        for options in ({'birth_death_cells':0},{'birth_death_metric_rank':True},
                        {'birth_death_backend':'bad'},{'birth_death_space':'data'},
                        {'birth_death_feature_scale':'none'},{'num_particles':5}):
            with self.assertRaises(ValueError):
                recipe().replace(**options)
        with self.assertRaises(ValueError):
            FeatureCellSnapshot.fit(torch.zeros(5,8),generator=stream())
        with self.assertRaises(ValueError):
            snap.support(torch.full((64,8),float('nan')))

    def test_actual_ordinary_transport_and_bounded_capture(self):
        trainer=make_trainer(512)
        counts=[]
        handle=trainer.G.register_forward_pre_hook(lambda module,args:counts.append(len(args[0])))
        bd=trainer.birth_death
        bd.observe_real(real_batch(512))
        result=bd.maybe_apply(trainer,0.)
        handle.remove()
        self.assertGreater(result['ordinary_discoveries'],0)
        self.assertGreater(result['ordinary_moves'],0)
        self.assertEqual(result['iso_moves'],0)
        self.assertLessEqual(result['ordinary_moves'],int(.05*512))
        self.assertLessEqual(max(counts),256)
        self.assertEqual(result['moves'],len(bd.moved_rows))
        self.assertGreater(result['work']['count_test_terms'],0)
        self.assertLessEqual(result['work']['max_distance_columns'],256)
        self.assertEqual(bd.snapshot.cache_version,1)
        self.assertGreater(result['invalidated_cells'],0)
        self.assertTrue(bool((bd.n==0).all()))

    def test_real_anchor_ball_and_guard(self):
        g=stream()
        n=1024
        real=torch.cat((torch.randn((n//2,4),generator=g,dtype=torch.float64)*.04-2,
                        torch.randn((n//2,4),generator=g,dtype=torch.float64)*.04+2))
        query=real.clone()
        snap=FeatureCellSnapshot.fit(real,generator=stream())
        _,p,_=snap.support(query)
        flags=torch.zeros(n,dtype=torch.bool);flags[:32]=True
        excluded=torch.tensor([32,33,34])
        child,parent,detail=snap.select_parents(query,flags,ordinary_children=excluded,generator=stream(),pvalues=p)
        self.assertEqual(len(child),32)
        self.assertLessEqual(detail['candidate_ids'].shape[1],256)
        self.assertFalse(bool(flags[parent].any()))
        self.assertFalse(bool(torch.isin(parent,excluded).any()))
        self.assertTrue(bool(((detail['candidate_ids']==parent[:,None])&detail['candidate_mask']).any(1).all()))
        self.assertTrue(bool((detail['anchor_reference_rows']%2==0).all()))
        too_many=flags.clone();too_many[:100]=True
        child,parent,detail=snap.select_parents(query,too_many,generator=stream(),pvalues=p)
        self.assertEqual(len(child),0);self.assertFalse(detail['guard_passed'])

    def test_row_copy_and_aligned_jitter(self):
        trainer=make_trainer()
        bd=trainer.birth_death
        child=torch.tensor([0,1]);parent=torch.tensor([30,31])
        with torch.no_grad():
            trainer.ema_prior.z.copy_(trainer.prior.z+10.)
        state=trainer.opt_g.state[trainer.prior.z]
        for key,offset in (('exp_avg',1.),('exp_avg_sq',2.),('max_exp_avg_sq',3.)):
            state[key]=torch.arange(512,dtype=torch.float32).reshape(256,2)+offset
        state['step']=torch.tensor(7.)
        trainer.opt_g.latent_history.copy_(torch.arange(512,dtype=torch.float32).reshape(256,2)+4.)
        z,ema=trainer.prior.z.detach().clone(),trainer.ema_prior.z.detach().clone()
        buffers={key:value.clone() for key,value in state.items()}
        history=trainer.opt_g.latent_history.clone()
        expected_stream=torch.Generator().set_state(bd.stream.get_state())
        delta=bounded_jitter(z[parent],torch.randn((2,2),generator=expected_stream))
        training_stream=trainer.noise_generator.get_state().clone()
        bd._move(trainer,child,parent)
        self.assertTrue(torch.equal(trainer.prior.z[child],z[parent]+delta))
        self.assertTrue(torch.equal(trainer.ema_prior.z[child],ema[parent]+delta))
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            self.assertTrue(torch.equal(state[key][child],buffers[key][parent]))
        self.assertTrue(torch.equal(state['step'],buffers['step']))
        self.assertTrue(torch.equal(trainer.opt_g.latent_history[child],history[parent]))
        self.assertTrue(torch.equal(training_stream,trainer.noise_generator.get_state()))
        latent=z[:16]
        g1,g2=stream(),stream()
        expected=trainer.G(latent+bounded_jitter(latent,torch.randn(latent.shape,generator=g1)))
        trainer.controller.perturb_latent=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('legacy nearest pass'))
        actual=trainer._generate(trainer.G,latent,0.,g2)
        self.assertTrue(torch.equal(actual,expected))

    def test_stale_refresh(self):
        real=torch.randn((128,8),generator=stream(),dtype=torch.float64)
        snap=FeatureCellSnapshot.fit(real,generator=stream())
        ids=snap.cache_queries(real).clone()
        rows=torch.tensor([0,1])
        repaired=real[90:92].clone()
        new_ids,_=snap.assign(repaired)
        invalidated,affected=snap.refresh_rows(rows,repaired)
        expected_cells=torch.unique(torch.cat((ids[rows],new_ids)))
        self.assertTrue(bool(affected[expected_cells].all()))
        self.assertTrue(bool(invalidated[torch.isin(ids,expected_cells)].all()))
        self.assertTrue(torch.equal(snap.query_cell_ids[rows],new_ids))
        self.assertEqual(int(snap.query_counts.sum()),128)
        self.assertEqual(snap.cache_version,1)

    def test_checkpoint_guards_empty_and_nonzero_move_replay(self):
        trainer=make_trainer()
        empty=deepcopy(trainer.birth_death.state_dict())
        trainer.birth_death.load_state_dict(empty)
        for change in ({'fill':1},{'sample_shape':(2,)},{'backend_schema':99}):
            bad=deepcopy(empty);bad.update(change)
            with self.assertRaises(ValueError):trainer.birth_death.check_state(bad)
        real=real_batch()
        trainer.step(real)
        self.assertGreater(trainer.birth_death.last['ordinary_moves'],0)
        saved=trainer.state_dict()
        for change in ({'reservoir':None},{'reservoir':saved['birth_death']['reservoir'].double()},
                       {'reservoir':saved['birth_death']['reservoir'][:,:1]}, {'sample_shape':None}):
            bad=deepcopy(saved['birth_death']);bad.update(change)
            with self.assertRaises(ValueError):trainer.birth_death.check_state(bad)
        continued=trainer.step(real)
        expected=trainer.state_dict()
        self.assertGreater(trainer.birth_death.last['ordinary_moves'],0)
        restored=make_trainer()
        restored.load_state_dict(saved)
        self.assertIsNone(restored.birth_death.snapshot)
        replay=restored.step(real)
        self.assertTrue(torch.equal(continued['loss_g'],replay['loss_g']))
        self.assertTrue(same_state(expected,restored.state_dict()))

    def test_default_source_recipe_checkpoint_and_hooks(self):
        reference=Path('/ml2/hypergan/gan-attempts/noout-20260928/pkg-E22/particlegan')
        self.assertEqual((reference/'birth_death.py').read_bytes(),(ROOT/'pkg-CB64-RA/particlegan/birth_death.py').read_bytes())
        legacy=make_trainer(backend='knn')
        self.assertEqual(type(legacy.birth_death).__name__,'ParticleBirthDeath')
        self.assertFalse(any(key.startswith('birth_death_') and key in ('birth_death_backend','birth_death_cells',
            'birth_death_metric_rank','birth_death_chunk','birth_death_parent_policy') for key in legacy.recipe.to_dict()))
        saved=legacy.state_dict();legacy.load_state_dict(saved)
        with self.assertRaises(ValueError):make_trainer().load_state_dict(saved)
        trainer=make_trainer()
        called=[]
        tester=trainer._table_tester()
        old_rebase=tester.rebase
        tester.rebase=lambda parameters,rows:(called.append(('tester',rows.clone())),old_rebase(parameters,rows))[-1]
        old_reset=trainer.row_evidence.reset
        trainer.row_evidence.reset=lambda rows:(called.append(('row_evidence',rows.clone())),old_reset(rows))[-1]
        trainer.step(real_batch())
        self.assertGreater(trainer.birth_death.last['moves'],0)
        self.assertEqual([name for name,_ in called],['tester','row_evidence'])
        for name,rows in called:self.assertTrue(torch.equal(rows,trainer.birth_death.moved_rows))


if __name__=='__main__':
    begin=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Integration))
    receipt=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                 seconds=time.perf_counter()-begin,seed=SEED,cpu_threads=torch.get_num_threads())
    (ROOT/'implementation/test_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
