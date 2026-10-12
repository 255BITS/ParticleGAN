"""Read-only fixed-row word diagnostic; no construction, draws or updates."""
from pathlib import Path
import copy
import hashlib
import json
import random

import torch
import torch.nn.functional as F

from experiments.forge.api import task_formulation_context
from particlegan.kinetic_transport import kinetic_transport_loss

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/queue')
CASES = {
    'constant_parent': 'r1r2-tier1-constant-search-v1/fd19e04990e34421a25166d36f7ee319',
    'm110': 'r1r2-role-m110-v1/3a45cf3eabb64155a1263efabae653e5',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone(), random.getstate()
    rows = []
    for role, relative in CASES.items():
        attempt = ARCHIVE / relative
        checkpoint = attempt / 'provenance/provenance-state.pt'
        records_path = attempt / 'observed-records.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['fixture']['api_state']
        weights, latent = state['models']['generator'], state['table'].detach()
        h = F.leaky_relu(F.linear(latent, weights['net.0.weight'], weights['net.0.bias']), .2)
        h = F.leaky_relu(F.linear(h, weights['net.2.weight'], weights['net.2.bias']), .2)
        logits = F.linear(h, weights['net.4.weight'], weights['net.4.bias']).detach().reshape(5, 28, 6).requires_grad_(True)
        fake = logits.softmax(1)
        records = torch.load(records_path, map_location='cpu', weights_only=False)
        target = records[-1]['views'][2]['target'].detach()
        loss = kinetic_transport_loss(fake, target, projections=32)
        output_gradient = torch.autograd.grad(loss, fake, retain_graph=True)[0]
        logits_gradient = torch.autograd.grad(loss, logits)[0]
        rows.append(dict(role=role, attempt_archive_path=relative,
                         checkpoint_sha256=sha(checkpoint), retained_records_sha256=sha(records_path),
                         unit_global_loss=float(loss.detach()),
                         output_gradient_l2=float(output_gradient.norm()),
                         logits_gradient_l2=float(logits_gradient.norm()),
                         minimum_logit=float(logits.detach().min()), maximum_logit=float(logits.detach().max())))
    # Static construction/effective-binding check, never an admitted candidate.
    parent_id = 'r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9'
    candidate = json.loads((ROOT / 'configs/forge/configurations' / (parent_id + '.json')).read_text())
    candidate = copy.deepcopy(candidate)
    candidate['recipe_overrides']['kinetic_transport_weight'] = 1.
    protocol = json.loads((ROOT / 'configs/forge/protocols/screening.json').read_text())
    tasks = ['gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold', 'ring16_acquisition', 'five_word_joint_smoke', 'clockfree_audit_measurement_v1']
    bindings = []
    for task_id in tasks:
        task = json.loads((ROOT / 'configs/forge/tasks' / (task_id + '.json')).read_text())
        context = task_formulation_context(candidate, task, protocol, root=ROOT)
        bindings.append(dict(task=task_id, execution_path=context.execution_path,
                             consumer=context.component_transport, global_weight=context.recipe.kinetic_transport_weight,
                             local_weight=context.recipe.kinetic_transport_local_weight,
                             reg_arm=context.recipe.reg_arm, reg_every=context.recipe.reg_every))
    assert torch.equal(rng[0], torch.get_rng_state()) and rng[1] == random.getstate()
    print(json.dumps(dict(schema_version=1, measurement='CPU float32 functional saved decoder at all five uniform zero-sigma prior rows versus retained five-word target bank',
                          scope='endpoint population-panel output/logit sensitivity; not actual G-phase minibatch force or a prospective from-initialization experiment',
                          rows=rows, hypothetical_in_memory_bindings=bindings,
                          training_updates=0, sampling_draws=0, rng_states_unchanged=True,
                          candidate_declared=False, queue_work=False), indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
