"""Audit completed original two-pole diagnostics without constructing a model."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import json

import torch

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.rng import NamedStreams
from experiments.forge.sources import verify_snapshot
from experiments.forge.state import state_digest

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def main():
    spec=importlib.util.spec_from_file_location('publication',ROOT/'reports/forge/bcap-develop-integration/publish.py')
    publication=importlib.util.module_from_spec(spec);spec.loader.exec_module(publication)
    entries={};sources={};reference_task=None;reference_streams=None;panels=[]
    for cohort in ('ablations','repairs'):
        report=read_json(OUT/('results.json' if cohort=='ablations' else 'repairs/results.json'))
        for item in report['rows']:
            attempt=publication.certified_attempt(ROOT,item['attempt_id'])
            row=attempt['result']['task_results'][0]
            saved,proof=publication.checkpoint(row)
            request=attempt['request'];source=request['source']
            assert source['origin_commit']==item['source']['origin_commit']
            assert stable_hash(source['files'])==source['digest']==item['source']['digest']
            if source['digest'] not in sources:
                verify_snapshot(Path(source['snapshot_path']),source)
                sources[source['digest']]=dict(origin_commits=[],verified_files=len(source['files']),
                    snapshot=source['snapshot_path'])
            if source['origin_commit'] not in sources[source['digest']]['origin_commits']:
                sources[source['digest']]['origin_commits'].append(source['origin_commit'])
            task=deepcopy(request['tasks']['two_pole']);task.pop('field_ownership')
            assert task['execution']['fixed_initialization']==dict(critic='stored_host_weights',particles='zeros')
            assert task['execution']['steps']==proof['completed_steps']==80
            if reference_task is None:reference_task=task
            assert task==reference_task
            streams=saved['streams'];bindings=streams['manifest']['bindings']
            named=NamedStreams(streams['manifest']['seed'],version=streams['manifest']['version'])
            named.validate_state_dict(streams)
            assert streams['manifest']['seed']==0
            non_eval={key:dict(binding=bindings[key],state=state_digest(value))
                for key,value in streams['states'].items() if bindings[key]['family']!='eval'}
            if reference_streams is None:reference_streams=non_eval
            assert non_eval==reference_streams
            assert row['evidence']['guards']['unintended_rng_deviations']==0
            assert len(row['evidence']['observations'])==24
            assert row['gate_status']==item['gate_status'] and row['metrics']==item['metrics']
            points=torch.cat([value['value'].flatten() for value in saved['role_parameters']['prior']])
            panels.append(dict(cohort=cohort,role=item['role'],count=points.numel(),
                negative=int((points<0).sum()),positive=int((points>0).sum()),
                minimum=float(points.min()),maximum=float(points.max()),
                population_std=float(points.std(unbiased=False)),
                state_sha256=state_digest(saved['role_parameters']['prior'])))
            entries[(cohort,item['role'])]=(row,saved)
    pairs=[]
    for first,second in (('incumbent','projection'),('transport','combined')):
        a,sa=entries[('ablations',first)];b,sb=entries[('ablations',second)]
        for key in ('models','role_parameters'):
            assert state_digest(sa[key])==state_digest(sb[key])
        assert a['evidence']['observations']==b['evidence']['observations']
        pairs.append(dict(roles=[first,second],actual_final_models_equal=True,
            actual_final_role_parameters_equal=True,all_24_observations_equal=True,
            model_state_sha256=state_digest(sa['models']),role_state_sha256=state_digest(sa['role_parameters'])))
    result=dict(schema_version=1,status='PASS',qualification_input=False,completed_attempts=len(entries),
        sources=sources,task_contract_sha256=stable_hash(reference_task),
        initialization_evidence='Exact declared stored-host-critic/zero-particle fixture bound to frozen construction source; no separate initial tensor dump is retained.',
        non_eval_named_streams=len(reference_streams),all_consumed_non_eval_states_and_bindings_equal=True,
        non_eval_stream_proof_sha256=stable_hash(reference_streams),projection_inactivity_pairs=pairs,
        final_actual_particle_panels=panels)
    atomic_json(OUT/'audit.json',result)
    print(json.dumps(dict(status=result['status'],completed_attempts=len(entries),
        sources=list(sources),non_eval_named_streams=len(reference_streams),panels=panels),sort_keys=True),flush=True)


if __name__=='__main__':main()
