import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.evaluate_gym_gan_control import (make_controller, refresh_leaderboard,
    verify_gan_training, verify_legacy_gan)
from lib.gym_state_control import parameter_hashes


def valid_gan_bundle():
    critic=nn.Linear(2,1)
    initial=parameter_hashes(dict(G=critic,E=critic,prior=critic))['G']
    with torch.no_grad():critic.weight.add_(.1)
    return dict(config=dict(arm='joint',steps=2500,adversarial_weight=1.,marginal_weight=1.),
        step=250,gan_steps=250,gan_training=True,
        D=SimpleNamespace(critics=nn.ModuleDict({'joint':critic})),
        provenance=dict(gan_training=True,sources={'experiments/train_gym_gan_control.py':'x','lib/gym_gan_control.py':'y'},
            initial_parameters={'D.joint':initial}),
        training_summary=dict(gan_training=True,gan_steps=2500))


def test_eligibility_needs_gan_updates_known_source_and_trained_discriminator():
    bundle=valid_gan_bundle();verify_gan_training(bundle)
    variants=[]
    for key,value in [('gan_training',False),('gan_steps',0),('D',None)]:
        b=copy.deepcopy(bundle);b[key]=value;variants.append(b)
    b=copy.deepcopy(bundle);b['config']['adversarial_weight']=0;variants.append(b)
    b=copy.deepcopy(bundle);b['provenance']['sources']={};variants.append(b)
    b=copy.deepcopy(bundle);critic=b['D'].critics['joint'];b['provenance']['initial_parameters']['D.joint']=parameter_hashes(dict(G=critic,E=critic,prior=critic))['G'];variants.append(b)
    b=copy.deepcopy(bundle)
    with torch.no_grad():b['D'].critics['joint'].weight.fill_(float('nan'))
    variants.append(b)
    for bad in variants:
        with pytest.raises(ValueError):verify_gan_training(bad)


def test_non_gan_routes_and_imitation_legacy_rejected():
    for arm in ('expert','probes','auxiliary','imitation','full_label'):
        with pytest.raises(ValueError,match='GAN-only'):make_controller(arm,'unused.pt','cpu')
    with pytest.raises(ValueError,match='never imitation'):
        verify_legacy_gan(dict(config={'arm':'imitation'},D=nn.Linear(2,1)),'unused.pt')


def test_non_gan_selection_cannot_enter_gan_leaderboard(tmp_path,monkeypatch):
    import json
    path=tmp_path/'selections';path.mkdir()
    (path/'probes.json').write_text(json.dumps({'arm':'probes'}))
    with pytest.raises(RuntimeError,match='ineligible selection'):refresh_leaderboard(tmp_path)
    (path/'probes.json').unlink()
    checkpoint=tmp_path/'fake.pt';checkpoint.write_text('not a trained GAN')
    (path/'joint.json').write_text(json.dumps(dict(arm='joint',selected_checkpoint=str(checkpoint),
        validation=dict(gan_eligibility={'eligible':True}))))
    def reject_checkpoint(*args):
        raise ValueError('Actual checkpoint failed GAN eligibility')
    monkeypatch.setattr('experiments.evaluate_gym_gan_control.make_controller',reject_checkpoint)
    with pytest.raises(ValueError,match='Actual checkpoint'):
        refresh_leaderboard(tmp_path)
