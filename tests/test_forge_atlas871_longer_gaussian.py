"""ROOT-only software controls. Authored NOT_RUN; no model, sampler or optimizer execution."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.forge import atlas871_longer as duration
from experiments.forge import atlas871_longer_owner as owner
from experiments.forge import atlas871_contract as contract

OPERATIONS = {name:0 for name in contract.EXTRA_OPERATIONS}


def no_operation(name):
    def denied(*args, **kwargs):
        OPERATIONS[name] += 1
        raise AssertionError('forbidden software operation: ' + name)
    return denied


def candidate():
    return dict(id=duration.CANDIDATE_ID, recipe_preset='atlas', recipe_overrides={},
        extensions={}, claim_contract={'experimental_track':duration.TRACK_ID})


def curve():
    return [dict(step=s, sample_count=4096, finite_fraction=1., mean_error_sigma=0.,
                 std_ratio=1., cdf_ks=.03) for s in duration.checkpoints()]


class LongerControls(unittest.TestCase):
    def test_task_duration_delta(self):
        task = deepcopy(duration.EXPECTED_TASK)
        self.assertEqual(duration.validate(task)['task_payload_sha256'], duration.TASK_DIGEST)
        self.assertEqual(task['execution']['steps'], 2000)
        self.assertEqual(task['execution']['host_definition']['steps'], 2000)
        self.assertEqual(task['evaluation']['observations'], 48)
        self.assertEqual(task['resources']['timeout_seconds'], 240)
        self.assertEqual(task['execution']['prior'], duration.EXPECTED_PARENT['execution']['prior'])
        self.assertEqual(task['evaluation']['thresholds'], duration.EXPECTED_PARENT['evaluation']['thresholds'])

    def test_original_task_pin(self):
        root = Path(__file__).resolve().parents[1]
        raw = (root / duration.PARENT_PATH).read_bytes()
        self.assertEqual({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}, duration.PARENT_PIN)
        self.assertEqual(json.loads(raw), duration.EXPECTED_PARENT)
        self.assertEqual(duration.digest(json.loads(raw)), duration.PARENT_DIGEST)
        duration.validate(duration.EXPECTED_TASK, root=root)

    def test_original_prefix_clocks(self):
        original = [math.ceil(k * 1000 / 24) for k in range(1,25)]
        self.assertEqual(duration.checkpoints()[:24], original)
        self.assertEqual(len(set(duration.checkpoints())), 48)
        self.assertEqual(original[-1], 1000)

    def test_appended_clocks(self):
        expected = [math.ceil(k*1000/24) for k in range(1,49)]
        self.assertEqual(duration.checkpoints(), expected)
        self.assertEqual(expected[24:30], [1042,1084,1125,1167,1209,1250])
        self.assertEqual(expected[-1], 2000)
        self.assertNotEqual(expected[:24], [math.ceil(k*2000/24) for k in range(1,25)])
        from experiments.forge.adapters import _checkpoints
        self.assertEqual(_checkpoints(duration.EXPECTED_TASK), expected)
        self.assertEqual(_checkpoints(duration.EXPECTED_PARENT), expected[:24])

    def test_unchanged_five_suffix_gates(self):
        from benchmarks.transfer_suite.protocol import test_verdict as original_verdict
        spec={'steps':2000,'thresholds':duration.EXPECTED_PARENT['evaluation']['thresholds']}
        rows=curve()
        rows[-5]['cdf_ks']=.0500001
        result=duration.test_verdict(spec,{'observations':rows,'live':rows[-1]})
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['convergence']['passing_suffix'],4)
        self.assertIsNone(result['convergence']['confirmed_step'])
        rows[-5]['cdf_ks']=.05
        self.assertEqual(duration.test_verdict(spec,{'observations':rows,'live':rows[-1]})['status'],'PASS')
        old=original_verdict({'steps':1000,'thresholds':spec['thresholds']},
                            {'observations':rows[:24],'live':rows[23]})
        prefix=duration.test_verdict(spec,{'observations':rows,'live':rows[-1]})
        self.assertEqual(old['metrics'],prefix['metrics'])
        self.assertEqual(old['convergence']['minimum_stable_checks'],5)
        self.assertEqual(prefix['convergence']['minimum_stable_checks'],5)

    def test_incomplete_curve_rejected(self):
        spec={'steps':2000,'thresholds':duration.EXPECTED_PARENT['evaluation']['thresholds']}
        rows=curve()
        self.assertEqual(duration.test_verdict(spec,{'observations':rows[:-1],'live':rows[-1]})['status'],'INCOMPLETE')
        self.assertEqual(duration.grade_transfer(duration.EXPECTED_TASK,{'observations':rows[:24],
            'live':rows[23]})['status'],'INCOMPLETE')
        rows[1]['step']=rows[0]['step']
        self.assertEqual(duration.test_verdict(spec,{'observations':rows,'live':rows[-1]})['status'],'INVALID')

    def test_strict_task_mutations(self):
        mutations=[lambda t:t['execution']['prior'].__setitem__('sigma',.03),
          lambda t:t['evaluation']['thresholds'][0].__setitem__(2,4095),
          lambda t:t['execution']['host_definition'].__setitem__('hidden',64),
          lambda t:t['execution'].__setitem__('steps',1999),
          lambda t:t['evaluation'].__setitem__('observations',24),
          lambda t:t['evaluation'].__setitem__('scoring_weights','state_selected'),
          lambda t:t['resources'].__setitem__('cpu_threads',2)]
        for mutate in mutations:
            task=deepcopy(duration.EXPECTED_TASK);mutate(task)
            with self.subTest(task=task):
                with self.assertRaises(ValueError): duration.validate(task)
        task=deepcopy(duration.EXPECTED_TASK);task['preflight_blockers']=[];task['field_ownership']={}
        self.assertEqual(duration.validate(task)['task_payload_sha256'],duration.TASK_DIGEST)

    def test_full79_recipe_binding(self):
        from dataclasses import asdict
        from experiments.forge.api import task_formulation_context,FormulationContext
        fences={name:no_operation('extra_model_forwards') for name in ('construct','build_trainer','build_prior')}
        with patch.multiple(FormulationContext, **fences):
            context=task_formulation_context(candidate(),duration.EXPECTED_TASK,{'seed':0})
        expected=owner.EXPECTED_RECIPES[duration.TASK_ID]
        self.assertEqual(len(expected),79)
        self.assertEqual(duration.canonical(asdict(context.recipe)),duration.canonical(expected))
        self.assertIsNone(context.recipe.total_steps)
        self.assertEqual(expected['network_lr_horizon_cap'],1600)
        self.assertFalse(hasattr(context,'_radius_observer844'))

    def test_fixed_rng_and_initializer(self):
        from experiments.forge.rng import NamedStreams
        from experiments.forge.initialization import task_initializer
        first,second=NamedStreams(0),NamedStreams(0)
        for family,component,purpose in [('init','generator','construction'),('init','prior','z'),
            ('data','target','training'),('prior','latent','indices'),('noise','prior','gaussian'),
            ('eval','live','samples')]:
            self.assertEqual(first.seed_for(family,component=component,purpose=purpose),
                             second.seed_for(family,component=component,purpose=purpose))
        self.assertEqual(task_initializer(duration.EXPECTED_TASK,candidate()),
                         task_initializer(duration.EXPECTED_PARENT,candidate()))

    def test_ordinary_admission_resources(self):
        task=deepcopy(duration.EXPECTED_TASK)
        resources=task['resources']
        protocol={'seed':0}
        compute={'backend':'cuda','threads':1,'model':'NVIDIA RTX A6000'}
        science={'candidate_revision':'new-revision','protocol':protocol,'compute':compute}
        key=duration.digest(science)
        job={'compatibility_key':key,'science':science,'task_id':duration.TASK_ID,
            'task_ids':[duration.TASK_ID],'budget_seconds':240,'resources':{
                'memory_mb':resources['gpu_memory_mb'],'gpus':1,'cpu_threads':1,'allow_cpu':False,
                'backend':'cuda','gpu_model':compute['model']}}
        request={'candidate':candidate(),'candidate_revision':'new-revision','protocol':protocol,
                 'tasks':{duration.TASK_ID:task},'jobs':[job],'source':{'digest':'new-source'}}
        output=Path('/tmp/atlas871-software-fixture-attempt')
        envelope={'schema_version':1,'request':request,'job':job,'worker':{
            'device':'1','directory':str(output),'token':'owned','attempt':'fixture'}}
        got=owner.vector_admission_contract(request,task,output,'cuda:0',envelope,cuda_visible_devices='1')
        self.assertEqual(got['budget_seconds'],240)
        self.assertEqual(got['device'],'cuda:0')
        for bad in ['0', 'cpu', None]:
            with self.assertRaises(ValueError):
                owner.vector_admission_contract(request,task,output,'cuda:0',envelope,cuda_visible_devices=bad)
        mutated=deepcopy(envelope);mutated['job']['resources']=deepcopy(resources)
        with self.assertRaises(ValueError):
            owner.vector_admission_contract(request,task,output,'cuda:0',mutated,cuda_visible_devices='1')

    def test_scoped_registry(self):
        from experiments.forge import atlas_existing_mog
        self.assertTrue(atlas_existing_mog.supports_candidate(candidate()))
        self.assertEqual(atlas_existing_mog.task_resources(duration.EXPECTED_TASK),
                         {'num_particles':256,'z_dim':2,'batch_size':128})
        self.assertEqual(atlas_existing_mog.blockers(duration.EXPECTED_TASK,candidate()),[])
        self.assertEqual(atlas_existing_mog.supporting_source_paths(duration.EXPECTED_PARENT,candidate()),())
        bad=candidate();bad['recipe_overrides']={'lr':.001}
        self.assertFalse(atlas_existing_mog.supports_candidate(bad))
        self.assertTrue(atlas_existing_mog.blockers(duration.EXPECTED_TASK,bad))
        with self.assertRaises(ValueError):duration.validate(duration.EXPECTED_PARENT)
        old=candidate();old['id']='atlas-existing-mog-nearest-positive791-v1'
        old['claim_contract']['experimental_track']='atlas791_existing_mog_nearest_positive'
        self.assertTrue(atlas_existing_mog.blockers(duration.EXPECTED_TASK,old))
        from experiments.forge.api import FormulationContext,CapabilityError
        with self.assertRaises(CapabilityError):
            FormulationContext(recipe_preset='atlas',recipe_overrides=owner.task_resources(duration.EXPECTED_TASK),
                prior=duration.EXPECTED_TASK['execution']['prior'],policy_task=duration.EXPECTED_TASK,
                candidate_id=old['id'])
        with self.assertRaises(CapabilityError):
            FormulationContext(recipe_preset='atlas',policy_task=duration.EXPECTED_PARENT,
                candidate_id=duration.CANDIDATE_ID)

    def test_observer_omitted(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'experiments/forge/atlas871_longer_owner.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(name and ('observer844' in name or 'radius_owner' in name) for name in imports))
        self.assertNotIn('PassiveRadiusObserver',ast.dump(tree))
        self.assertEqual(OPERATIONS,{name:0 for name in contract.EXTRA_OPERATIONS})


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LongerControls))
    passed=result.wasSuccessful() and all(type(v) is int and v==0 for v in OPERATIONS.values())
    report=dict(schema='forge_atlas871_software_control_result_v1',status='PASS' if passed else 'FAIL',
        checks={name:'PASS' if passed else 'UNPROVEN' for name in contract.SOFTWARE_CHECKS},
        added_operations=OPERATIONS, test_count=result.testsRun)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    raise SystemExit(0 if passed else 1)
