"""Two original-auxiliary host controls; explicitly NOT a pure-GAN result.

Run only ae_gan_hold or unused_token_hold. Imported archived H signal math is
unchanged. Observe host settings without modifying them: reconstruction/hold
weights retain the original value 1. H's other disabled auxiliary weights stay0.
"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
import critic_signal
import critic_signal_screen as screen
from benchmarks.locked_shared.hosts import ae_gan_hold,unused_token_hold
ARCHIVED_POLICY=critic_signal.signal_policy

@contextmanager
def compatibility_policy(options):
    with ARCHIVED_POLICY(options) as receipt, ExitStack() as stack:
        control={'track':'original-required-auxiliary compatibility control',
                 'exclusively_adversarial':False,'host_overrides':[],
                 'original_host_auxiliary_settings':[]}
        receipt['compatibility_control']=control
        receipt['generator_objective']='GANLoss.g_loss plus original required host reconstruction or unused-token hold; NOT exclusively adversarial'
        original_ae=ae_gan_hold.train
        original_unused=unused_token_hold.train
        def ae(cfg,*args,**kwargs):
            assert cfg.reconstruction_weight==1.
            assert cfg.fm_weight==cfg.cover_weight==cfg.particle_l2==0.
            control['original_host_auxiliary_settings'].append({'host':'ae_gan_hold','reconstruction_weight':cfg.reconstruction_weight,'cover_weight':cfg.cover_weight,'fm_weight':cfg.fm_weight,'particle_l2':cfg.particle_l2})
            return original_ae(cfg,*args,**kwargs)
        def unused(cfg,*args,**kwargs):
            assert cfg.hold_weight==1.
            assert cfg.fm_weight==cfg.cover_weight==cfg.particle_l2==0.
            control['original_host_auxiliary_settings'].append({'host':'unused_token_hold','hold_weight':cfg.hold_weight,'cover_weight':cfg.cover_weight,'fm_weight':cfg.fm_weight,'particle_l2':cfg.particle_l2})
            return original_unused(cfg,*args,**kwargs)
        stack.enter_context(patch.object(ae_gan_hold,'train',ae))
        stack.enter_context(patch.object(unused_token_hold,'train',unused))
        yield receipt

critic_signal.signal_policy=compatibility_policy
original_sources=screen.source_hashes

def sources():
    result=original_sources()
    rel='reports/toy100/selected_h_compatibility.py'
    result[rel]=screen.sha(ROOT/rel)
    return result

screen.source_hashes=sources
if __name__=='__main__':
    screen.main()
