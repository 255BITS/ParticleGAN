"""No-init saved model fixture, extracted from the frozen RA6 CPU contract."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
ORIGINAL = ROOT/'quality/ra6/integration-contract/check_reaction.py'


def fixture(state, module):
    names = {'linear','network','fixture'}
    definitions = [n for n in ast.parse(ORIGINAL.read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions}==names
    namespace = dict(torch=torch,deepcopy=deepcopy,SimpleNamespace=SimpleNamespace,module=module)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(ORIGINAL),'exec'),namespace)
    trainer,backend = namespace['fixture'](state)
    # Every chosen saved state still has an open prior/sigma scale, so the
    # unchanged learnable-noise law uses its configured floor. The reaction
    # probe uses that actual sigma, rather than raw exp(log_sigma).
    assert any(t['s']>1/64 for t in state['lr_settle'][0] if t is not None)
    trainer.last_output_sigma = max(trainer.last_output_sigma,state['recipe']['output_noise_std'])
    return trainer,backend
