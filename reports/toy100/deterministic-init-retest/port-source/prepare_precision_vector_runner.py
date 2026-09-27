"""Source-only reviewed-host adaptation; no Torch import or model execution."""
from pathlib import Path
import ast,json,hashlib,difflib
PORT=Path(__file__).resolve().parent
old=PORT/'new-init-dv12-unequal-mass';new=PORT/'new-init-vector-screen'
f=new/'vector_screen.py';s=(old/'vector_screen.py').read_text()
s=s.replace('Exact frozen unequal_mass follow-up for the unchanged new-init DV12 package.','Six frozen vector tasks for each unchanged sealed new-init API package.')
s=s.replace('import json\n','import json\nimport math\n')
s=s.replace('def verify(package_root, declaration_path):','def verify(package_root, declaration_path, task_name):')
s=s.replace("    plan = json.loads((ROOT / 'followup-protocol.json').read_text())\n    assert sha(declaration_path.read_bytes()) == plan['candidate_declaration_sha256']\n",'')
s=s.replace("    assert declaration['candidate'] == 'API-DV12-new-init'\n    assert declaration['initial_optimizer_state'] == 'native_lazy'\n    assert declaration['optimizer_step_devices'] == {'G': 'cpu', 'D': 'cpu'}", "    assert declaration['candidate'] in ('API-RP12-new-init', 'API-RP14-new-init', 'API-RP15-new-init')\n    assert declaration['initial_optimizer_state'] == 'declared_eager'\n    assert declaration['optimizer_step_devices'] == {'G': 'parameter', 'D': 'parameter'}")
s=s.replace("    assert declaration['resolved_recipe']['continuous_policy'] == 'dv12'", "    assert declaration['resolved_recipe']['game_update'] == 'secant_resolvent'")
s=s.replace("    task = json.loads((ROOT / 'task.json').read_text())", "    task = json.loads((ROOT / 'task-specs.json').read_text())[task_name]")
s=s.replace("    assert sha((ROOT / 'task.json').read_bytes()) == proof['task_sha256']", "    assert sha((ROOT / 'task-specs.json').read_bytes()) == proof['task_specs_sha256']\n    assert sha((ROOT / 'frozen_shared_critic.py').read_bytes()) == proof['shared_critic_module_sha256']")
a=s.index('    # Exact selected branch of original vector_discriminator; reject other cards.');b=s.index('    assert all(p.device.type',a)
s=s[:a]+'''    # Exact three branches of frozen public_default_verification.vector_discriminator.
    if card is None:
        critic = host.SimpleMLPDiscriminator(2, cfg.get('d_hidden', cfg['hidden']),
                                             cfg.get('d_layers', cfg['layers']), cfg['fourier'])
    elif card['implementation'] == 'shared_batch_feature_v1':
        assert (card['feature'], card['placement'], card['trunk_normalization'], card['name']) == (
            'distance', 'head', 'center', 'batchfeat_center6_distance_head')
        critic = package.BatchDistanceDiscriminator(in_dim=2, hidden_dim=card['width'],
            n_hidden=card['layers'], scales=tuple(card['kernel_scales']),
            beta=card['softplus_beta'], eps=card['eps'])
    elif card['implementation'] == 'shared_critic_v1':
        from frozen_shared_critic import constructor
        critic = constructor(card)(2, card['hidden'], card['layers'], card['fourier'])
    else:
        raise ValueError('unknown frozen discriminator card')
'''+s[b:]
s=s.replace("    assert torch.count_nonzero(first.D.head.weight[:, -first.D.scales.numel():]) == 0", "    if card is not None and card['implementation'] == 'shared_batch_feature_v1':\n        assert torch.count_nonzero(first.D.head.weight[:, -first.D.scales.numel():]) == 0")
s=s.replace("    assert receipt['candidate_declaration_sha256'] == sha(args.declaration.read_bytes())", "    assert receipt['candidate_declaration_sha256'] == sha(args.declaration.read_bytes())\n    assert receipt['task'] == args.task")
s=s.replace("    expected = list(range(50, 1201, 50))\n    assert cfg['steps'] == 1200 and len(batches) == 1200", "    expected = [math.ceil(i * cfg['steps'] / 24) for i in range(1, 25)]\n    assert cfg['steps'] in (1200, 1600) and len(batches) == cfg['steps']")
s=s.replace("                           policy=trainer.controller.diagnostics(), critic=trainer.penalty.diagnostics(),", "                           critic=trainer.penalty.diagnostics(),")
s=s.replace("                rates.write(json.dumps(row, allow_nan=False) + '\\n')", "                row['game_stats'] = material(torch, trainer.game_stats)\n                row['precision'] = material(torch, trainer.precision.state)\n                assert row['game_stats']['field_evaluations'] == 2\n                rates.write(json.dumps(row, allow_nan=False) + '\\n')")
s=s.replace("                if step in (1, 1200):", "                if step in (1, cfg['steps']):")
s=s.replace("task='vector_unequal_mass'", "task=args.task")
s=s.replace("    parser.add_argument('--output', type=Path, required=True)", "    parser.add_argument('--task', choices=tuple(json.loads((ROOT / 'task-specs.json').read_text())), required=True)\n    parser.add_argument('--output', type=Path, required=True)")
s=s.replace('declaration, task = verify(args.package_root, args.declaration)', 'declaration, task = verify(args.package_root, args.declaration, args.task)')
s=s.replace("        result.update(candidate_declaration_sha256=sha(args.declaration.read_bytes()), source_sha256=source_hashes,", "        result.update(task=args.task, candidate_declaration_sha256=sha(args.declaration.read_bytes()), source_sha256=source_hashes,")
assert 'trainer.controller' not in s
ast.parse(s);f.write_text(s)
(PORT/'precision-vector-runner.patch').write_text(''.join(difflib.unified_diff((old/'vector_screen.py').read_text().splitlines(True),s.splitlines(True),fromfile='reviewed-dv12-vector-screen.py',tofile='precision-vector-screen.py')))
(new/'README.md').write_text('Six exact frozen vector cards for RP12/RP14/RP15 unchanged new-init packages. Derived from the independently reviewed DV12 vector adapter; ten prior data/model/scorer definitions unchanged, exact original SimpleMLP critic added, complete original shared_critic_research module retained for three promoted custom cards. No historical candidate learner or score is imported. Original task declarations, cards and source ZIPs are pinned in host-source-proof.json.\n\nPrior CPU std.5/own seed0 is constructed first; CPU G/D follow, then CUDA and actual GANTrainer public factory initialization. Separate CUDA data0/latent1/penalty2 and private noise5 persist. Own candidate/task CPU proof must repeat complete initial state without RNG reset and match actual CUDA model bytes before update1. Batch-distance appended head zeros are explicitly checked for its card only. Custom critic buffers and all parameters are included in full state proof. Eager parameter-device Adam counters belong to each package and are verified without repair.\n\nThe frozen original sample streams are retained: two independent real batches per accepted update. The exact candidate joint_step evaluates its caller G-real callback once and reuses that tensor for both fields, restoring learner latent streams between fields; the data stream is independent of latent draws. No adapter cache or package change is added. Before first update, 1200 or1600 dry data/latent receipts are generated without changing learner state; actual accepted draws/cursors must match every row. Complete public steps are scoped serial. Every accepted game/precision/rate record is retained.\n\nEvaluation retains4096 draws, latent990/target991/projection992/outputnoise2303; all24 original ceil(i*N/24) observations and original bounds/final-five gate. Spiral is1600 updates, other five are1200. Each accepted update performs two game fields; no equal-compute claim. Only host resource dimensions vary; candidate rates/prior objective/noise/controller/EMA unchanged. All source and own CPU proof required; no GPU training run performed during preparation.\n')
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest={f.name:sha(f.read_bytes()) for f in sorted(new.iterdir()) if f.is_file() and f.name!='bundle-sha256.json'}
(new/'bundle-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(bundle=str(new),seal=sha((new/'bundle-sha256.json').read_bytes()))))
