"""Saved-array media contracts; no networks, training or model sampling."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.contracts import file_hash
from experiments.forge.gaussian_tasks import schedule
from experiments.forge.tier1_media import _scored_outputs, render

ROOT=Path(__file__).resolve().parents[1]
METRICS=dict(sample_count=4096,finite_fraction=1.,mean_error_sigma=.01,std_ratio=1.,cdf_ks=.01)


def fixture(tmp_path,phase='smoke'):
    task=json.loads((ROOT/'configs/forge/tasks'/('gaussian1d_'+phase+'.json')).read_text())
    root=tmp_path/'evaluator';root.mkdir()
    steps=schedule(0,1000) if phase=='smoke' else schedule(1000,4000)+schedule(4000,6000)
    observations=[dict(step=step,**METRICS) for step in steps]
    # Explicit saved-array software fixture, never scientific evidence.
    records=[dict(step=0 if phase=='smoke' else 1000,samples=torch.zeros(4096,1),metrics=deepcopy(METRICS))]
    records += [dict(step=step,samples=torch.full((4096,1),3. if step>4000 else 2.),metrics=deepcopy(METRICS)) for step in steps]
    path=root/'observed-samples.pt';torch.save(records,path)
    evidence=dict(artifact_root=str(root),artifact_manifest=manifest_artifacts(root),observations=observations,
                  saved_observer_outputs=dict(path=path.name,sha256=file_hash(path)),
                  sampling_law='public_prior_without_output_noise',scoring_weights='live')
    return task,evidence,path,records


@pytest.mark.parametrize('phase',['smoke','stability'])
def test_gaussian_saved_arrays_use_certified_root_and_drop_only_initial(tmp_path,phase):
    task,evidence,path,records=fixture(tmp_path,phase)
    saved,inputs=_scored_outputs(task,evidence,tmp_path/'different-local-directory')
    assert [row['step'] for row in saved]==[row['step'] for row in records[1:]]
    assert inputs=={str(path):file_hash(path)}


@pytest.mark.parametrize('delta',['descriptor','bytes','extra_file','missing_initial','off_cadence','metrics'])
def test_gaussian_media_rejects_descriptor_manifest_and_schedule_corruption(tmp_path,delta):
    task,evidence,path,records=fixture(tmp_path)
    if delta=='descriptor':evidence['saved_observer_outputs']['sha256']='0'*64
    elif delta=='bytes':evidence['saved_observer_outputs']['bytes']=0
    elif delta=='extra_file':(path.parent/'extra.txt').write_text('uncertified')
    else:
        if delta=='missing_initial':records[0]['step']=1
        elif delta=='off_cadence':records[1]['step']=43
        else:records[1]['metrics']['cdf_ks']=.2
        torch.save(records,path)
        evidence['saved_observer_outputs']['sha256']=file_hash(path)
        evidence['artifact_manifest']=manifest_artifacts(path.parent)
    with pytest.raises(ValueError):_scored_outputs(task,evidence,tmp_path)


def test_legacy_descriptor_stays_strict(tmp_path):
    task,evidence,path,records=fixture(tmp_path)
    task['evaluation']['kind']='transfer_sustained'
    evidence['saved_observer_outputs']['path']='evaluator/observed-samples.pt'
    # Legacy descriptors still require bytes and exactly matching scored records.
    with pytest.raises(KeyError):_scored_outputs(task,evidence,tmp_path)
    evidence['saved_observer_outputs']['bytes']=path.stat().st_size
    with pytest.raises(ValueError,match='schedules differ'):_scored_outputs(task,evidence,tmp_path)


def test_stability_media_uses_declared_shifted_mean_without_model_calls(tmp_path,monkeypatch):
    from matplotlib.axes import Axes
    means=[];original=Axes.plot
    def plot(axis,*args,**kwargs):
        if kwargs.get('label')=='Declared target density':
            x,y=args[:2];means.append(float(x[y.argmax()]))
        return original(axis,*args,**kwargs)
    monkeypatch.setattr(Axes,'plot',plot)
    task,evidence,_,_=fixture(tmp_path,'stability')
    result=render(task,dict(evidence=evidence,gate_status='PASS'),tmp_path,tmp_path/'goal.gif')
    assert any(abs(mean-2.)<.02 for mean in means)
    assert any(abs(mean-3.)<.02 for mean in means)
    assert result['optimizer_updates_added']==result['sampling_draws_added']==0
    assert result['observation_count']==120
