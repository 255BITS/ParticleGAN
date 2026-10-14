"""Protocol and identity fences only; no models, GPU training or paid retries."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.research_protocols import registered_protocol, protocol_semantics, validate_unit, comparison_identity, validate_submission
from experiments.forge.codex_runner import semantic_key, observe_training, process_identity
import os
from unittest.mock import patch, MagicMock

class ResearchProtocols(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.original=dict(schema_version=1,id='screening',revision=1,seed=0,rng={'candidate_identity_in_seed':False},scoring={'weights':'live'})
        self.protocol={**self.original,'id':'paired-seed-1','seed':1,'research':{'cohort_id':'paired','purpose':'paired_replication'}}
        atomic_json(self.root/'configs/forge/protocols/screening.json',self.original)
        self.protocol_path=self.root/'configs/forge/protocols/paired-seed-1.json'; atomic_json(self.protocol_path,self.protocol)
        code=self.root/'particlegan/protected.py'; code.parent.mkdir(); code.write_text('fixed-code')
        self.roles={role:{'candidate_id':role,'recipe_overrides_sha256':stable_hash({'active':role=='candidate'})} for role in ('control','candidate')}
        self.registry=dict(schema_version=1,status='preregistered',qualification_input=False,seed_set=[1,2,3,4,5,6],roles=self.roles,
                           tasks={'original':{'evaluation':{'fixed':True}}},screening_sha256=file_hash(self.root/'configs/forge/protocols/screening.json'),
                           protocols={'paired-seed-1':file_hash(self.protocol_path)},source_semantics={'particlegan/protected.py':file_hash(code)})
        self.registry['view']={'evidence_scope':'research_diagnostic'}
        self.registry_path=self.root/'configs/forge/research-cohorts/paired.json'; atomic_json(self.registry_path,self.registry)
        self.req=dict(protocol=self.protocol,source={'snapshot_path':str(self.root),'digest':'frozen'},candidate_revision='candidate-frozen',
                      candidate={'id':'candidate','recipe_overrides':{'active':True}},view={'evidence_scope':'research_diagnostic'},tasks=self.registry['tasks'])
        self.unit=dict(role='candidate',replication_role='paired_seed',seed=1,protocol_semantics_sha256=stable_hash(protocol_semantics(self.protocol)),
                       comparison_sha256=stable_hash(comparison_identity(self.registry)),source_semantics_sha256=stable_hash(self.registry['source_semantics']),
                       source_digest='frozen',candidate_revision='candidate-frozen',registration_sha256=file_hash(self.registry_path),protocol_id='paired-seed-1')

    def test_frozen_registration_and_role_bindings(self):
        protocol,registry=registered_protocol(self.root,'paired-seed-1'); self.assertEqual(protocol,self.protocol); self.assertEqual(registry,self.registry)
        validate_unit(self.req,self.unit)
        for key,value in [('seed',2),('role','control'),('source_digest','other-source')]:
            with self.assertRaisesRegex(ValueError,'unit'): validate_unit(self.req,{**self.unit,key:value})

    def test_no_unregistered_seed_or_gate_change(self):
        for delta in ({'seed':7},{'scoring':{'weights':'ema'}}):
            atomic_json(self.protocol_path,{**self.protocol,**delta})
            with self.assertRaises(ValueError): registered_protocol(self.root,'paired-seed-1')
        with self.assertRaises(FileNotFoundError): registered_protocol(self.root,'unregistered')
        atomic_json(self.root/'configs/forge/protocols/screening.json',{**self.original,'seed':1})
        with self.assertRaisesRegex(ValueError,'seed 0'): registered_protocol(self.root,'screening')

    def test_scientific_source_and_task_drift_are_fenced(self):
        altered=deepcopy(self.req); altered['tasks']['original']['evaluation']['fixed']=False
        with self.assertRaisesRegex(ValueError,'task'): validate_unit(altered,self.unit)
        (self.root/'particlegan/protected.py').write_text('changed-code')
        with self.assertRaisesRegex(ValueError,'implementation'): registered_protocol(self.root,'paired-seed-1')

    def test_twelve_registered_units_aliases_and_exact_repeats(self):
        study={'semantic_factors':{'mechanism':'fixed'},'task_ids':['original'],'experimental_unit':self.unit}; keys=[]
        for seed in range(1,7):
            for role in ('control','candidate'):
                unit={**self.unit,'seed':seed,'role':role,'protocol_semantics_sha256':stable_hash({**protocol_semantics(self.protocol),'seed':seed})}
                keys.append(semantic_key({**study,'experimental_unit':unit}))
        self.assertEqual(len(set(keys)),12)
        self.assertEqual(semantic_key(study),semantic_key({**study,'id':'new-name','cohort':'new-label','seed':99}))
        self.assertEqual(semantic_key(study),semantic_key({**study,'semantic_factors':{'cosmetic':'new-label'}}))
        self.assertEqual(semantic_key(study),semantic_key({**study,'experimental_unit':{**self.unit,'protocol_id':'cosmetic-id','registration_sha256':'renamed'}}))
        with self.assertRaises(ValueError): semantic_key({**study,'experimental_unit':{'seed':1}})

    def test_positive_new_updates_required_before_counting_training(self):
        directory=self.root/'attempt'; directory.mkdir(); pid=os.getpid()
        atomic_json(directory/'child.json',{'pid':pid,'process_identity':process_identity(pid),'phase':'training'})
        class Proc: pass
        proc=Proc(); proc.pid=pid
        active={'directory':directory,'attempt':'fixture','task_id':'original','log_offset':0,'prefix_steps':1000,'process':proc,
                'owner':{'request':self.req,'study':{'id':'fixture','experimental_unit':self.unit}},'lease':{'inode':1}}
        log=directory/'run.log'; log.write_text('{"event":"observation","step":1000}\n'); self.assertIsNone(observe_training(active,[]))
        with log.open('a') as stream: stream.write('{"event":"observation","step":1042}\n')
        row=observe_training(active,[]); self.assertEqual(row['new_updates_confirmed'],42); self.assertIsNone(observe_training(active,[row]))

    def test_research_submission_keeps_rng_and_nonqualification_bound(self):
        req=deepcopy(self.req); req['research_unit']=self.unit; req['rng']={'bound':'seed1'}
        req['jobs']=[{'science':{'seed':1,'protocol':self.protocol,'rng':req['rng'],'evidence_use':'research_diagnostic'}}]
        context=MagicMock(); context.streams.manifest.return_value=req['rng']
        with patch('experiments.forge.api.FormulationContext',return_value=context):
            validate_submission(req)
            for field,value in [('seed',2),('evidence_use','ordinary_qualification'),('rng',{'bound':'other'})]:
                altered=deepcopy(req); altered['jobs'][0]['science'][field]=value
                with self.assertRaisesRegex(ValueError,'protocol/RNG'): validate_submission(altered)
