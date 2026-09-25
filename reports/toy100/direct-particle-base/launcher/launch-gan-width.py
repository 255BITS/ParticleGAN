#!/usr/bin/env python3
"""Start three bounded attempts from the selected direct-particle-response GAN."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).with_name('launch-gan-formulations.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.REFERENCES = '''Use the committed reports/toy100/direct-particle-base bundle as the exact parent.
Read its README, original-declaration.json, audit.json and unequal-width raw receipt.
The config, mechanism, response and probe together define the formulation.
All sixteen fixtures are bundled. replay.py reconstructs hash-verified sources.
Earlier recipes and attempt directories are historical, not your starting base.
'''
launcher.LANES = {
    'critic_local_shape': '''Own critic regularization for local distribution shape.
The unequal-width gate has acceptable mass and global distance but excessive
component covariance error, dominated by its narrowest component. Inspect the
critic-gradient evidence, then test small generally applicable changes to the
selected real-gradient penalty or its interaction with the fake RMS cap. Keep
direct-particle response, optimizers and noise schedules unchanged. No known
component widths, labels, centers, target-statistic normalization or task-name
branches in training. Do not repeat the prior dead zones, real warmups or symmetric
cap variants unchanged. Gate unequal width, then protect all fifteen passes.''',
    'latent_shape_dynamics': '''Own learned-latent particle update dynamics.
The successful direct-particle response excludes registered ParticlePrior latent
parameters, so it cannot alter unequal width. Inspect actual latent gradients and
displacements to propose a small role-based update mechanism improving settled
shape without losing coverage. Preserve the exact direct-particle rule, critic,
network optimizer and noise schedules. Applying short-memory coherent gain to
all latent priors already failed ring; do not repeat it unchanged. No mode-based
preconditioning, target forces, metric feedback, extra updates or LR grids.
Gate unequal width first, then all fifteen parent passes before unmeasured toys.''',
    'noise_fidelity': '''Own training-noise fidelity near convergence.
Inspect where input/output noise enters real and generated critic inputs and
generator gradients. Test whether a generally applicable, explicitly declared
noise mechanism or schedule can retain narrow component shape under the original
finite training budget. This lane may change training noise as its formulation;
preserve evaluation/data noise, seeds, all fixed random draws where possible,
model architecture, LR schedules, critic penalty and exact direct response.
No knowledge of target widths or task-specific schedules. Declare changed noise
and RNG consumption before execution. Prefer one structural hypothesis and
evidence-driven refinements to a coefficient sweep. Gate unequal width first;
continue every passing proposal through all old passes and remaining GPU toys.''',
}

if __name__ == '__main__':
    launcher.main()
