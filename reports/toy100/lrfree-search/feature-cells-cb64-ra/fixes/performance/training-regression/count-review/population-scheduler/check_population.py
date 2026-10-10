"""Population-continuity contracts; saved tensors, no optimizer updates."""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root", type=Path, default=HERE / "pkg-POPULATION")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--composed", action="store_true", help="Check the scheduler splices in a composed package")
args = parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES="" if args.device == "cpu" else "0",
    PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8")
sys.dont_write_bytecode = True

import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import math
import torch
from types import SimpleNamespace

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device(args.device)
sys.path.insert(0, str(args.package_root))
from particlegan import GANTrainer, ParticlePrior, Recipe
from particlegan.continuous import SequentialSettleTest, _T9875


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(str((v.dtype, tuple(v.shape), str(v.device))).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=repr):
                h.update(repr(key).encode()); add(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for x in v: add(x)
        else:
            h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def original_module():
    p = ROOT / "pkg-CB64-RA4/particlegan/continuous.py"
    spec = importlib.util.spec_from_file_location("population_reference", p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cls(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def method(owner, name):
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_proof():
    source = args.package_root / "particlegan"
    base = ROOT / "pkg-CB64-RA4/particlegan"
    old_c, new_c = (ast.parse((p / "continuous.py").read_text()) for p in (base, source))
    changed = []
    for name in ("SequentialSettleTest", "StationarityLR"):
        a, b = cls(old_c, name), cls(new_c, name)
        if name == "SequentialSettleTest":
            changed.append(dict(owner=name, kind="class", base=sha(base / "continuous.py")))
            new_c.body[new_c.body.index(b)] = deepcopy(a)
        else:
            old_init, new_init = method(a, "__init__"), method(b, "__init__")
            b.body[b.body.index(new_init)] = deepcopy(old_init)
    assert ast.dump(old_c, include_attributes=False) == ast.dump(new_c, include_attributes=False)
    old_t, new_t = (ast.parse((p / "training.py").read_text()) for p in (base, source))
    old_tr, new_tr = cls(old_t, "GANTrainer"), cls(new_t, "GANTrainer")
    for name in ("_state_dict", "_load_state_dict"):
        a, b = method(old_tr, name), method(new_tr, name)
        new_tr.body[new_tr.body.index(b)] = deepcopy(a)
    assert ast.dump(old_t, include_attributes=False) == ast.dump(new_t, include_attributes=False)
    different = [p.name for p in sorted(base.glob("*.py")) if sha(p) != sha(source / p.name)]
    if not args.composed:
        assert different == ["continuous.py", "training.py"]
    return dict(continuous_exact_splices=["SequentialSettleTest", "StationarityLR.__init__"],
        training_exact_splices=["GANTrainer._state_dict", "GANTrainer._load_state_dict"],
        all_other_continuous_and_training_AST_unchanged=True, changed_modules=different,
        serving_noise_averaging_update_order_count_API_unchanged=True)


def fresh(n=20, d=2, s=1., b=8.):
    param = torch.nn.Parameter(torch.zeros(n, d, device=DEVICE))
    t = SequentialSettleTest(early_stationary_only=True, final_table=_T9875)
    t.rows = n; t.s = s; t.b = b; t.begin([param])
    return param, t


def window(t, missing=(), scale="both"):
    t.r_b = [torch.full((t.rows,), -.2-.01*i, device=DEVICE) for i in range(12)]
    t.r_2b = [torch.full((t.rows,), -.3-.01*i, device=DEVICE) for i in range(6)]
    if scale == "2b":
        t.r_b = [torch.full((t.rows,), .1 if i % 2 else -.1, device=DEVICE) for i in range(12)]
    for pair in t.r_b + t.r_2b:
        if missing: pair[torch.tensor(missing, device=DEVICE)] = float("nan")
    t.blocks_in_window = 24


def settle(t, missing=(), scale="both", step=1):
    window(t, missing, scale)
    return t._decide(step)


checks = {}
base = original_module()
fixture = ROOT / "validation-ra4/learned/training/toy/CB64-RA4/checkpoint-1250.pt"
saved = torch.load(fixture, map_location="cpu", weights_only=False)["trainer"]
source_paths = [Path(__file__), fixture] + sorted((args.package_root / "particlegan").glob("*.py"))
source_paths += [ROOT / "pkg-CB64-RA4/particlegan/continuous.py", ROOT / "pkg-CB64-RA4/particlegan/training.py"]
hashes = {str(p):sha(p) for p in source_paths}
rng_before = torch.get_rng_state().clone()
if args.device == "cuda":
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.2, 0)
checks["source_scope"] = source_proof()

p,t = fresh()
assert t.stationary_rows.dtype == torch.bool and t.stationary_rows.shape == (20,)
assert not t.population_active and not t.stationary_rows.any()
assert settle(t) == "stationary" and t.s == .5 and t.population_active
assert t.stationary_undo_s == 1. and t.stationary_rows.all()
checks["fresh_population_negative_verdict"] = dict(covered_rows=20, s=.5, undo=1.)
t.r_b = [torch.arange(20, device=DEVICE).float()]
t.r_2b = [torch.arange(20, device=DEVICE).float()+.25]
t.blocks = [torch.arange(40, device=DEVICE).float()]
t.blocks_in_window = 1; t.tau=.75
kept_b, kept_tau = t.b, t.tau
untouched = deepcopy((t.r_b[0][2:],t.r_2b[0][2:],t.blocks[0].view(20,2)[2:]))
t.rebase([p],torch.tensor([0],device=DEVICE))
assert t.population_active and t.s == .5
t.rebase([p],torch.tensor([0,0],device=DEVICE))
assert t.population_active and t.s == .5 and int(t.stationary_rows.sum()) == 19
t.rebase([p],torch.tensor([1],device=DEVICE))
assert not t.population_active and t.s == 1 and t.last_decisive == 0 and t.b_anchor is None
assert t.b == kept_b and t.tau == kept_tau and t.blocks_in_window == 1
assert torch.equal(t.r_b[0][2:],untouched[0]) and torch.equal(t.r_2b[0][2:],untouched[1])
assert torch.equal(t.blocks[0].view(20,2)[2:],untouched[2]) and torch.isnan(t.r_b[0][:2]).all()
t.rebase([p],torch.tensor([2,3],device=DEVICE))
assert t.s == 1 and t.counts["population_expiries"] == 1
checks["Q_distinct_rows_one_release_and_window_preserved"] = dict(Q=.05, allowed=1, expires_at=2,
    unchanged_b_tau_and_untouched_evidence=True, repeated_rows_spend_once=True)

_,partial = fresh()
assert settle(partial,missing=(0,1)) == "inconclusive" and partial.s == 1 and partial.b == 16
assert partial.counts["population_coverage_rejections"] == 1
_,partial2 = fresh()
assert settle(partial2,missing=(0,1),scale="2b") == "inconclusive"
assert partial2.last["verdict_b"] == 0 and partial2.last["verdict_2b"] == -1
_,permitted = fresh()
assert settle(permitted,missing=(0,)) == "stationary" and int(permitted.stationary_rows.sum()) == 19
checks["decisive_scale_coverage_gate"] = dict(eighteen_of_twenty_rejected=True,nineteen_of_twenty_accepted=True,
    scale2b_negative_retained_in_diagnostics=True, test_levels_unchanged=True)

_,excluded = fresh()
excluded.exclude=torch.zeros(20,device=DEVICE,dtype=torch.bool); excluded.exclude[:2]=True
assert settle(excluded) == "inconclusive"
_,held = fresh(); held.hold_descent=True
assert settle(held) == "held" and held.s == 1 and held.b == 8
checks["existing_exclusion_and_hold"] = dict(nonvoters_not_counted=True, existing_broad_hold_preserved=True)

parities=0
for vb,v2 in [(-1,-1),(-1,0),(0,-1),(1,1),(1,0),(0,1),(-1,1),(1,-1),(0,0),(None,0)]:
    _,new=fresh()
    old=base.SequentialSettleTest(early_stationary_only=True,final_table=base._T9875)
    old.rows=20; old.b=8.
    window(new)
    old.r_b=deepcopy(new.r_b);old.r_2b=deepcopy(new.r_2b);old.blocks_in_window=24
    stat=lambda v:dict(verdict=v,n=12,mean=0.,t=0.,log_bf=0.)
    a=new._conclude(stat(vb),stat(v2),8.,1,False)
    b=old._conclude(stat(vb),stat(v2),8.,1,False)
    assert (a,new.s,new.b,new.last_decisive,new.last_decisive_scale)==(b,old.s,old.b,old.last_decisive,old.last_decisive_scale)
    parities+=1
checks["original_direction_and_scale_actions"] = dict(full_coverage_cases=parities, exact=True)

param,bounded=fresh(s=.5)
assert settle(bounded) == "stationary" and bounded.s == .25
bounded.rebase([param],torch.tensor([0,1],device=DEVICE))
assert bounded.s == .5 and bounded.counts["population_expiries"] == 1
assert settle(bounded,step=2) == "stationary" and bounded.s == .25
checks["immediate_pre_descent_rate_and_renewal"] = dict(s_before=.5,s_after_descent=.25,s_after_release=.5,
    no_forced_full_reopen=True, renewal=True)

state=bounded.state_dict(); buf=io.BytesIO();torch.save(state,buf);buf.seek(0)
serialized=torch.load(buf,map_location=DEVICE,weights_only=False)
_,resumed=fresh();resumed.load_state_dict(serialized,param.numel())
assert digest(resumed.state_dict())==digest(state)
serialized["stationary_rows"].zero_()
assert resumed.stationary_rows.all()
for tester in (bounded,resumed):
    tester.rebase([param],torch.tensor([3,4],device=DEVICE))
    settle(tester,step=3)
assert digest(bounded.state_dict())==digest(resumed.state_dict())
checks["tester_serialization_continuation_and_alias"] = dict(exact=True, returned_mask_not_alias=True,
    added_state_tensor_elements=20)

good=resumed.state_dict();before=digest(good);invalid=[]
for label in ("old-law","mask-dtype","mask-shape","law","level","inactive-stamp","bad-undo","low-active-coverage"):
    bad=deepcopy(good)
    if label=="old-law":bad.pop("population_policy")
    elif label=="mask-dtype":bad["stationary_rows"]=bad["stationary_rows"].float()
    elif label=="mask-shape":bad["stationary_rows"]=bad["stationary_rows"][:-1]
    elif label=="law":bad["population_policy"]="other"
    elif label=="level":bad["population_q"]=.051
    elif label=="inactive-stamp":bad["population_active"]=False;bad["stationary_undo_s"]=None
    elif label=="bad-undo":bad["stationary_undo_s"]=1.
    elif label=="low-active-coverage":bad["stationary_rows"][:2]=False
    try:resumed.load_state_dict(bad,param.numel())
    except ValueError:invalid.append(label)
    else:raise AssertionError(f"accepted {label}")
    assert digest(resumed.state_dict())==before
checks["tester_atomic_rejections"] = invalid

p,whole=fresh();settle(whole);window(whole)
whole.restart([p],reopen=False)
assert whole.s==1 and whole.last_decisive==0 and not whole.population_active
assert all(torch.isnan(v).all() for v in whole.r_b+whole.r_2b)
settle(whole);whole.restart([p],reopen=True)
assert whole.s==1 and whole.b==1 and not whole.population_active
checks["whole_group_restart"] = dict(old_lineage_pairs_dropped=True, single_undo=True, existing_reopen_preserved=True)

# Full trainer, built from saved model/table tensors. No gradient or optimizer
# step occurs. Network construction is scoped so global RNG is not advanced.
with torch.random.fork_rng(devices=[]):
    G=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.LeakyReLU(.2),
        torch.nn.Linear(128,128),torch.nn.LeakyReLU(.2),torch.nn.Linear(128,2))
    D=torch.nn.Sequential(torch.nn.Linear(2,128),torch.nn.LeakyReLU(.2),
        torch.nn.Linear(128,128),torch.nn.LeakyReLU(.2),torch.nn.Linear(128,1))
G.load_state_dict(saved["models"]["G"]);D.load_state_dict(saved["models"]["D"])
G=G.to(DEVICE);D=D.to(DEVICE)
prior=ParticlePrior.__new__(ParticlePrior);torch.nn.Module.__init__(prior)
prior.z=torch.nn.Parameter(saved["models"]["prior"]["z"].clone().to(DEVICE))
trainer=GANTrainer(Recipe(**saved["recipe"]),G,D,prior=prior,seed=314159,serial_backward=True)
tt=trainer._table_tester();tt.begin([trainer.prior.z]);tt.b=8
assert settle(tt)=="stationary" and trainer._serve_settled()
trainer._serve_apply();assert trainer._fast is not None
active=trainer.state_dict();assert active["schema"]==5
restored=deepcopy(trainer);restored.load_state_dict(active)
assert digest(restored.state_dict())==digest(active)
for bad in (dict(active,schema=4),deepcopy(active)):
    if bad["schema"]==5:bad["lr_settle"][0][1].pop("population_policy")
    before=digest(restored.state_dict())
    try:restored.load_state_dict(bad)
    except ValueError:pass
    else:raise AssertionError("old trainer law accepted")
    assert digest(restored.state_dict())==before and restored._fast is not None
checks["trainer_schema5_active_roundtrip_and_atomic_rejection"] = dict(exact=True,served_model_preserved=True,
    rejects_schema4=True,rejects_forged_schema5_old_tester=True)

# Compile only the unchanged post-reaction hook from the actual production
# trainer. Two distinct action batches represent ordinary copies and novel
# births; both must enter moved_rows in the compositor.
tr_tree=ast.parse((args.package_root/"particlegan/training.py").read_text())
step=method(cls(tr_tree,"GANTrainer"),"_step")
hook=next(n for n in step.body if isinstance(n,ast.If) and ast.unparse(n.test)=="self.birth_death is not None"
    and any(isinstance(n2,ast.Call) and ast.unparse(n2.func)=="self.birth_death.maybe_apply" for n2 in ast.walk(n)))
fn=ast.FunctionDef(name="reaction_hook",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="self")],
    kwonlyargs=[],kw_defaults=[],defaults=[]),body=[deepcopy(hook)],decorator_list=[])
tree=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace={}
exec(compile(tree,"unchanged_production_reaction_hook","exec"),namespace)
backend=trainer.birth_death
trainer._serve_release()
for label,rows in (("ordinary_copy",torch.arange(40,device=DEVICE)),("novel_birth",torch.arange(40,60,device=DEVICE))):
    trainer.birth_death=SimpleNamespace(moved_rows=rows,maybe_apply=lambda self,sigma:dict(moves=len(rows)))
    namespace["reaction_hook"](trainer)
    if label=="ordinary_copy":assert tt.population_active and tt.s==.5
    else:assert not tt.population_active and tt.s==1. and not trainer._serve_settled()
trainer.birth_death=backend
trainer._serve_apply();assert trainer._fast is None
endpoint=trainer.state_dict();restored.load_state_dict(endpoint)
assert digest(restored.state_dict())==digest(endpoint)
assert tt.counts["population_expiries"]==1 and trainer.row_evidence.counters["resets"]==60
checks["actual_trainer_hook_copy_and_novel_birth"] = dict(ordinary_rows=40,novel_rows=20,expires_once=True,
    row_evidence_resets=60,unchanged_serving_policy_obeys_revoked_stamp=True,full_endpoint_roundtrip=True)
assert torch.equal(rng_before,torch.get_rng_state())
assert all(sha(p)==expected for p,expected in hashes.items())
assert torch.cuda.is_initialized()==(args.device=="cuda")
checks["source_inputs_rng_and_training_untouched"] = dict(source_hashes_unchanged=True,cpu_rng_unchanged=True,
    optimizer_updates=0,new_seeds=0,cuda_initialized=torch.cuda.is_initialized())
result=dict(status="PASS",device=args.device,scope="saved-tensor mechanical scheduler/checkpoint/API contracts",
    checks=checks,source_sha256=hashes,quality_passed=False,optimizer_updates=0,new_seeds=0,
    required_integration="Every copy and novel-birth child enters birth_death.moved_rows before the existing trainer hook.")
args.output.parent.mkdir(parents=True,exist_ok=True)
assert not args.output.exists(),"retain prior receipts"
args.output.write_text(json.dumps(result,indent=2,allow_nan=True)+"\n")
print(json.dumps(dict(status="PASS",contracts=len(checks),output=str(args.output))),flush=True)
