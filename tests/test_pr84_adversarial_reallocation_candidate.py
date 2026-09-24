from contextlib import contextmanager
import gzip
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import patch

import torch

from particlegan.gan_loss import GANLoss
from reports.toy100 import pr84_prediction_state_filter as replay
from reports.toy100 import pr84_adversarial_reallocation_candidate as candidate
from reports.toy100 import pr84_adversarial_reallocation_assay_v2 as output_v2

ROOT=Path(__file__).resolve().parents[1]


def test_generic_score_matches_v2_once_smoothed_loss():
    class Critic(torch.nn.Module):
        def forward(self,x): return x[:,0]**3 + .2*x[:,1]
    critic=Critic()
    fake=torch.tensor([[-1.,0.],[0.,0.],[-1.,0.],[1.,0.]],dtype=torch.float64)
    batch=dict(indices=torch.tensor([0,1,0,2]),real=torch.tensor([[2.,0.],[1.,0.],[0.,0.],[-1.,0.]],dtype=torch.float64),
               sigma=.029,noise=torch.zeros(4,2,dtype=torch.float64))
    candidates=torch.tensor([[2.,.4],[-2.,.1]],dtype=torch.float64)
    gan=GANLoss('logistic','rp')
    a,_,_=output_v2.select(critic,fake,batch,candidates,3,gan)
    b=candidate.select_proposal(lambda x:output_v2.base.fit.smooth(critic,x),gan,fake,batch,candidates,3)
    assert (a['donor'],a['candidate'],a['target'])==(b['donor'],b['real_sample'],b['target'])
    assert a['native_loss_before']==b['pre_loss'] and a['native_loss_after']==b['proposal_loss']


def test_actual_host_disabled_exact_active_batch_and_state_ownership(tmp_path):
    torch.set_num_threads(1)
    path=ROOT/'reports/toy100/continuous-evidence/convex-profiled-value1530/v2/tensors.pt.gz'
    captured=torch.load(BytesIO(gzip.decompress(path.read_bytes())),weights_only=True)['captured']
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    @contextmanager
    def selected(*,task,prediction):
        with candidate.adversarial_reallocation(task=task,correction=prediction) as value: yield value
    with patch.object(replay.prediction_module,'pr84_opponent_prediction',selected):
        original,_=replay.run_local(config,captured['pre_step'],start=1530,end=1530,
                                    opponent='current',source_dir=tmp_path)
        active,_=replay.run_local(config,captured['pre_step'],start=1530,end=1530,
                                  opponent='predicted',source_dir=tmp_path)
    assert original['accepted_states'][0]['accepted_state_sha256']==replay.state_hash(captured['post_bounded_g'])
    assert active['rng_final_sha256']==original['rng_final_sha256']
    assert active['noise']==original['noise']
    assert active['moment_steps']==original['moment_steps']=={'d':[1530],'g':[1530]}
    receipt=active['dynamics']
    assert receipt['native_G_loss_exact_checks']==1
    assert receipt['native_batch_checks']==receipt['native_index_and_noise_checks']==3
    assert receipt['correction_rng_checks']==receipt['correction_owner_checks']==1
    row=receipt['corrections'][0]
    assert row['final_loss']<=min(row['pre_loss'],row['native_loss'])
    assert row['selected'] in ('rest','native_gan','joint_fit')
    if row['selected']=='joint_fit':
        assert row['fit']['status']=='CONVERGED'
        assert row['final_loss']<min(row['pre_loss'],row['native_loss'])
    assert receipt['moment_updates_per_outer_step']==1
