"""CPU inference/planning on saved toy states; no training or CUDA context.

Derived snapshots are not serialized in the trainer checkpoint. Rebuild once
with the existing training seed on CPU, save it, and use exactly that partition,
count comparison and planning RNG for both accounting rules. This does not
recreate CUDA PCA random draws or provide a CUDA quality verdict.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "pkg-CB64-RA2"
VALIDATION = ROOT / "validation"
MODELS = Path("/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation/models_metrics.py")
sys.path.insert(0, str(PACKAGE))
import torch
from particlegan.feature_cells import FeatureCellBirthDeath, FeatureCellSnapshot, BoundedLatentGeometry, Q
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
spec = importlib.util.spec_from_file_location("saved_mass_models", MODELS)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)
SEED = 314159
STEPS = (1000, 2000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


current_source = inspect.getsource(FeatureCellSnapshot.ordinary_transport)
old_source = textwrap.dedent(current_source).replace(
    "clean_counts = torch.bincount(ids[~flags],minlength=self.cells)",
    "clean_counts = torch.bincount(ids,minlength=self.cells)")
assert old_source != textwrap.dedent(current_source)
namespace = dict(inspect.unwrap(FeatureCellSnapshot.ordinary_transport).__globals__)
exec(compile(old_source, "<pre-correction count-only comparison>", "exec"), namespace)
old_transport = namespace["ordinary_transport"]


@torch.no_grad()
def case(step):
    path = VALIDATION / "learned/training/toy/CB64-RA2" / f"checkpoint-{step:04d}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["trainer"]
    G, D = models.networks("toy")
    G.load_state_dict(saved["models"]["G"])
    D.load_state_dict(saved["models"]["D"])
    G.eval(); D.eval()
    prior = SimpleNamespace(z=saved["models"]["prior"]["z"])
    controller = SimpleNamespace(latent_bandwidth=saved["controller"]["latent_bandwidth"])
    trainer = SimpleNamespace(G=G, D=D, prior=prior, controller=controller)
    bd = FeatureCellBirthDeath.__new__(FeatureCellBirthDeath)
    bd.settings = saved["birth_death"]["settings"]
    bd.sample_shape = tuple(saved["birth_death"]["sample_shape"])
    bd._linears = [module for module in D.modules() if isinstance(module, torch.nn.Linear)]
    bd._heads = None
    bd.stream = torch.Generator().manual_seed(SEED)
    bd.latent_geometry = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
    bd.counters = {"feature_forward_rows": 0}
    q = bd._capture_generated(trainer, prior.z)
    real = bd._features(trainer, saved["birth_death"]["reservoir"], chunk=256)
    snap = FeatureCellSnapshot.fit(real, generator=bd.stream, cells=64, rank=8, chunk=256)
    pick = torch.randint(len(q), (len(q),), generator=bd.stream)
    fake = bd._capture_generated(trainer, prior.z[pick],
                                 sigma=checkpoint["record"]["diagnostics"]["output_sigma"], jitter=True)
    flags, pvalues, scores = snap.support(q)
    snap.cache_queries(q)
    comparison = snap.cell_comparison(fake)
    planning_state = bd.stream.get_state().clone()
    payload = dict(snapshot=deepcopy(vars(snap)), q=q, flags=flags, pvalues=pvalues, comparison=deepcopy(comparison),
                   planning_rng=planning_state, real_features=real, fake_features=fake, checkpoint=str(path),
                   checkpoint_sha256=sha(path), seed=SEED, checkpoint_step=step)
    torch.save(payload, HERE / f"snapshot-{step:04d}.pt")
    ids, _ = snap.assign(q)
    groups = snap._mass_topology()
    target = snap._mass_targets(len(q))
    table = torch.bincount(ids, minlength=snap.cells)
    supported = torch.bincount(ids[~flags], minlength=snap.cells)
    eligible = torch.bincount(ids[~flags & (pvalues > Q)], minlength=snap.cells)
    group_target = snap._group_counts(target)
    group_supported = snap._group_counts(supported)
    group_table = snap._group_counts(table)
    oracle_distance = torch.cdist(G(prior.z), models.oracle_centres()).min(1).values
    rows = []
    for name, method in (("pre_correction_table_counts", old_transport),
                         ("current_nonflagged_counts", FeatureCellSnapshot.ordinary_transport)):
        frozen = deepcopy(snap)
        stream = torch.Generator().set_state(planning_state)
        child, parent, detail = method(frozen, q, flags, deepcopy(comparison), generator=stream, pvalues=pvalues)
        iso_child, iso_parent, iso = frozen.select_parents(q, flags, ordinary_children=child,
                                      ordinary_parents=parent, generator=stream, pvalues=pvalues)
        rows.append(dict(policy=name, ordinary_moves=len(child), isolation_moves=len(iso_child),
                         children_flagged=int(flags[child].sum()), unique_parents=len(torch.unique(parent)),
                         supported_children=int((oracle_distance[child] <= .09).sum()),
                         guard_passed=iso["guard_passed"], details=plain(detail)))
    expected_last = saved["birth_death"]["last"]
    result = dict(step=step, checkpoint_sha256=sha(path), reconstructed_cpu_snapshot=True,
                  identical_snapshot_comparison_rng_across_rules=True, planning_rng_sha256=tensor_sha(planning_state),
                  flags=int(flags.sum()), population=len(q), ordinary_budget=int(Q*len(q)),
                  eligible_rows=int((~flags & (pvalues > Q)).sum()), real_groups=snap.mass_groups,
                  groups_at_or_below_supported_target=int((group_supported<=group_target).sum()),
                  groups_with_supported_surplus=int((group_supported>group_target).sum()),
                  groups_with_table_surplus=int((group_table>group_target).sum()),
                  excess_cells=int(comparison["excess"].sum()), deficit_cells=int(comparison["deficit"].sum()),
                  raw_count_certified_death_capacity=int((torch.floor(len(q)*comparison["difference"].clamp_min(0.)+1e-10)*comparison["excess"]).sum()),
                  raw_count_certified_birth_capacity=int((torch.floor(len(q)*(-comparison["difference"]).clamp_min(0.)+1e-10)*comparison["deficit"]).sum()),
                  significant_birth_cells_with_eligible_parent=int((comparison["deficit"]&(eligible>0)).sum()),
                  group_table_counts=group_table.tolist(), group_nonflagged_counts=group_supported.tolist(),
                  group_targets=group_target.tolist(), eligible_counts=eligible.tolist(),
                  frozen_gpu_last=dict(flags=expected_last["iso_flagged"], ordinary_moves=expected_last["ordinary_moves"],
                                       eligible=expected_last["eligible"], discoveries=expected_last["discoveries"]),
                  rules=rows)
    print(json.dumps(plain(result)), flush=True)
    return result


results = [case(step) for step in STEPS]
receipt = dict(status="DIAGNOSED", scope=__doc__, seed=SEED, records=results,
               cuda_initialized=torch.cuda.is_initialized(),
               sources={str(path): sha(path) for path in (Path(__file__), MODELS, PACKAGE/"particlegan/feature_cells.py")})
assert not receipt["cuda_initialized"]
(HERE / "paired-count-rules.json").write_text(json.dumps(plain(receipt), indent=2)+"\n")
