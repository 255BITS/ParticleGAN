"""Equality/empty-lineage control for the independently bounded copy law."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from check_paired_copy import (args, ROOT, HERE, FeatureCellBirthDeath, reference,
                               make_case, copy_contract, sha, fingerprint)
import torch


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    path = ROOT / 'validation-ra4/learned/training/toy/CB64-RA4/checkpoint-2000.pt'
    sources = [Path(__file__), HERE / 'check_paired_copy.py', path,
               args.package_root / 'particlegan/feature_cells.py',
               args.reference_package_root / 'particlegan/feature_cells.py']
    hashes = {str(p):sha(p) for p in sources}
    saved = torch.load(path, map_location='cpu', weights_only=False)['trainer']
    controlled = deepcopy(saved)
    controlled['models']['ema_prior']['z'] = controlled['models']['prior']['z'].clone()
    controlled['birth_death']['lineage_neighbors'].fill_(-1)
    ref_class = reference()
    child, parent = torch.arange(128), torch.arange(256, 384)
    row = copy_contract(controlled, 314159, ref_class, child, parent, 'equal-priors-empty-lineage')
    old_t, old = make_case(controlled, ref_class, 314159)
    new_t, new = make_case(controlled, FeatureCellBirthDeath, 314159)
    old._move(old_t, child.to(args.device), parent.to(args.device))
    new._move(new_t, child.to(args.device), parent.to(args.device))
    assert torch.equal(old_t.ema_prior.z, new_t.ema_prior.z)
    assert fingerprint(old_t.opt_g.state[old_t.prior.z]) == fingerprint(new_t.opt_g.state[new_t.prior.z])
    assert hashes == {str(p):sha(p) for p in sources}
    assert torch.cuda.is_initialized() == (args.device == 'cuda')
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='PASS', device=args.device,
        fixture_control='saved live prior copied into EMA; lineage cleared in private memory only',
        source_sha256=hashes, case=row, equal_priors_preserve_old_ema_bits=True,
        numerical_updates=0, new_seeds=0, fixture_files_unchanged=True)
    assert not args.output.exists()
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', equal_priors_preserve_old_ema_bits=True, output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
