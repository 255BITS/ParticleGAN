"""Read-only check of the actual Rp pairing in archived common-noise inputs."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from particlegan.gan_loss import GANLoss
from reports.toy100 import common_instance_noise_cold_endpoint as cold
from reports.toy100 import common_instance_noise_falsifier as common
from reports.toy100 import pr84_critic_relaxation as prior_diagnostic

SNAPSHOT=ROOT/'reports/toy100/continuous-evidence/common-instance-noise-round10/states/pr84-cold1200.pt.gz'
OUTPUT=ROOT/'reports/toy100/continuous-evidence/common-instance-noise-round10/pairing-audit.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run():
    raw=gzip.decompress(SNAPSHOT.read_bytes())
    if sha(raw)!=cold.ARCHIVE_RAW_SHA:
        raise AssertionError('wrong archived PR84 cold state')
    before=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        saved=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
        generator,critic,prior=prior_diagnostic.modules(cold.saved_view(saved))
        train,heldout,g_banks=cold.banks(saved,generator,prior)
        native={'real':train['real'][:128],'fake':train['fake'][:128]}
        observation=torch.Generator().manual_seed(cold.OBSERVATION_SEED)
        observed=common.bank_with_instance_noise(native,cold.WIDTH,observation)
        real=critic(observed['real'])
        fake=critic(observed['fake'])
        gan=GANLoss('logistic','rp')
        actual_d=gan.d_loss(real,fake)
        matched_d=F.softplus(fake-real).mean()
        crossed_d=F.softplus(fake[:,None]-real[None,:]).mean()

        # Advance the independent observation stream exactly as in the frozen
        # cold assay before constructing its first G bank. No parameter step.
        observation=torch.Generator().manual_seed(cold.OBSERVATION_SEED)
        common.bank_with_instance_noise(train,cold.WIDTH,observation)
        common.bank_with_instance_noise(heldout,cold.WIDTH,observation)
        g_bank=common.g_bank_with_instance_noise(g_banks[0],cold.WIDTH,observation)
        native_fake=generator(prior.z[g_bank['indices']])+g_bank['sigma']*g_bank['noise']
        observed_fake=torch.cat((native_fake+g_bank['eta_fake'],native_fake-g_bank['eta_fake']))
        observed_real=torch.cat((g_bank['real']+g_bank['eta_real'],g_bank['real']-g_bank['eta_real']))
        fake_g=critic(observed_fake)
        real_g=critic(observed_real)
        actual_g=common.common_g_loss(generator,prior,critic,g_bank,gan)
        matched_g=F.softplus(real_g-fake_g).mean()
        crossed_g=F.softplus(real_g[None,:]-fake_g[:,None]).mean()
        if not torch.equal(actual_d,matched_d) or not torch.equal(actual_g,matched_g):
            raise AssertionError('actual Rp scalar is not the matched-pair scalar')
        if torch.equal(actual_d,crossed_d) or torch.equal(actual_g,crossed_g):
            raise AssertionError('archive failed to distinguish crossed pairs')
        receipt={
            'scope':'read-only archived original critic and first native128 bank; no fit or training',
            'snapshot_raw_sha256':sha(raw),
            'source_sha256':sha(Path(__file__).read_bytes()),
            'native_shape':list(native['real'].shape),
            'observed_shape':list(observed['real'].shape),
            'd_logit_shape':list(real.shape),
            'd_subtraction_shape':list((real-fake).shape),
            'd_actual':float(actual_d.detach()),
            'd_matched_2n':float(matched_d.detach()),
            'd_crossed_2n_by_2n_counterfactual':float(crossed_d.detach()),
            'g_logit_shape':list(fake_g.shape),
            'g_subtraction_shape':list((fake_g-real_g).shape),
            'g_actual':float(actual_g.detach()),
            'g_matched_2n':float(matched_g.detach()),
            'g_crossed_2n_by_2n_counterfactual':float(crossed_g.detach()),
            'actual_equals_matched_bitwise':True,
            'actual_equals_crossed':False,
        }
    if not torch.equal(torch.get_rng_state(),before):
        raise AssertionError('global RNG changed')
    OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    torch.set_num_threads(1)
    print(json.dumps(run(),indent=2))
