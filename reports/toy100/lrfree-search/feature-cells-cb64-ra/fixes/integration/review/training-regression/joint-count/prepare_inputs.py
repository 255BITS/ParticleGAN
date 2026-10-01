"""Prepare fresh fixed CPU inputs; never rewrite frozen source/evidence."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent
SNAPS=HERE.parent
ROOT=SNAPS.parents[2]
sys.path.insert(0,str(HERE/"pkg-joint-count"))
from particlegan.feature_cells import FeatureCellSnapshot
SEED=314159


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def augment(value,real,fake,seed,planning_rng=None):
    snap=FeatureCellSnapshot.__new__(FeatureCellSnapshot)
    snap.__dict__.update(deepcopy(value["snapshot"]))
    even=snap.transform(real[0::2]);ids,_=snap._assign_metric(even)
    assert torch.equal(torch.bincount(ids,minlength=snap.cells),snap.reference_counts)
    odd=snap.transform(real[1::2]);odd_ids,_=snap._assign_metric(odd)
    assert torch.equal(torch.bincount(odd_ids,minlength=snap.cells),snap.real_calibration_counts)
    snap._fit_count_partition_metric(even,ids)
    odd_categories,_=snap._count_categories_metric(odd,odd_ids)
    snap.real_calibration_category_counts=torch.bincount(odd_categories,minlength=2*snap.cells)
    snap._update_storage()
    return dict(snapshot=deepcopy(vars(snap)),q=value["q"].clone(),flags=value["flags"].clone(),
                pvalues=value["pvalues"].clone(),real_features=real.clone(),fake_features=fake.clone(),
                seed=seed,planning_rng=planning_rng)


output=HERE/"inputs.pt"
if output.exists():raise SystemExit("inputs already exist; preserve the existing artifact")
fixed=[SNAPS/f"snapshot-{step:04d}.pt" for step in (1000,2000)]
mass_path=ROOT/"stability/mass-gpu-inputs.pt"
fixture_path=Path("/ml2/hypergan/gan-attempts/scaling-portability-20260929/scaling_a/shared_toy.py")
sources=[Path(__file__),HERE/"PROTOCOL.md",HERE/"pkg-joint-count/particlegan/feature_cells.py",mass_path,fixture_path]+fixed
before={str(p):sha(p) for p in sources}
cases={}
for step,path in zip((1000,2000),fixed):
    value=torch.load(path,map_location="cpu",weights_only=False)
    cases[f"saved_toy_{step}"]=augment(value,value["real_features"],value["fake_features"],value["seed"],value["planning_rng"])
spec=importlib.util.spec_from_file_location("joint_count_existing_fixture",fixture_path)
fixture=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fixture;spec.loader.exec_module(fixture)
original=torch.load(mass_path,map_location="cpu",weights_only=False)


@torch.no_grad()
def capture(toy,z):
    raw=toy.generate(z)
    features=[]
    for block in raw.split(256):
        hidden=torch.tanh(torch.nn.functional.linear(block,toy.w1.T.contiguous(),toy.b1))
        features.append(torch.nn.functional.softplus(torch.nn.functional.linear(hidden,toy.w2.T.contiguous(),toy.b2)))
    return torch.cat(features).double()


for name,value in original["scenarios"].items():
    toy,z,real,_=fixture.make_fixture(len(value["q"]),name)
    assert torch.equal(z,value["z"]),name
    assert torch.equal(capture(toy,z),value["q"]),name
    r=capture(toy,real)
    # The original mass artifact omitted emitted features. This mechanical
    # case uses its fixed clean table as the count sample, not an iid replay.
    cases[name]=augment(value,r,value["q"],original["seed"])
    cases[name]["count_sample_scope"]="fixed clean table mechanical count comparison, not iid emitted replay"

# A deterministic supported imbalance addresses the concrete2K regression.
offset=torch.linspace(-.04,.04,512,dtype=torch.float64)
left=torch.stack((offset-3.,.01*torch.cos(torch.arange(512,dtype=torch.float64))),1)
right=torch.stack((offset+3.,.01*torch.sin(torch.arange(512,dtype=torch.float64))),1)
real=torch.cat((left,right))
source=torch.Generator().set_state(cases["saved_toy_1000"]["planning_rng"])
snap=FeatureCellSnapshot.fit(real,generator=source,cells=2,rank=2,chunk=256)
q=torch.cat((torch.tensor([[-3.,0.]],dtype=torch.float64).expand(900,-1),
             torch.tensor([[3.,0.]],dtype=torch.float64).expand(124,-1))).clone()
flags,p,_=snap.support(q);assert not bool(flags.any()) and bool((p>.05).all())
snap.cache_queries(q)
cases["supported_mass_imbalance"]=dict(snapshot=deepcopy(vars(snap)),q=q,flags=flags,pvalues=p,
    real_features=real,fake_features=q.clone(),seed=SEED,planning_rng=source.get_state().clone(),
    count_sample_scope="deterministic fixed supported table, not iid quality sample")
cases["supported_mass_small_holes"]=deepcopy(cases["supported_mass_imbalance"])
small=cases["supported_mass_small_holes"]
small["q"][:46,1]+=2.2  # established nominal-hole displacement, no new level
small["fake_features"]=small["q"].clone()
small["flags"],small["pvalues"],_=snap.support(small["q"])
assert int(small["flags"].sum())==46
snap.cache_queries(small["q"]);small["snapshot"]=deepcopy(vars(snap))
cases["supported_balanced_control"]=deepcopy(cases["supported_mass_imbalance"])
control=cases["supported_balanced_control"];control["q"]=real.clone()
control["fake_features"]=q[:1].expand_as(real).clone()
control["flags"]=torch.zeros(len(real),dtype=torch.bool);control["pvalues"]=torch.ones(len(real))
snap.cache_queries(real);control["snapshot"]=deepcopy(vars(snap))

assert before=={p:sha(Path(p)) for p in before}
torch.save(dict(cases=cases,source_sha256=before,
    scope="fixed CPU geometry/even boundary and mechanical saved-input contracts; no new seed or quality verdict"),output)
receipt=dict(status="PREPARED",cases=list(cases),sources_unchanged=True,source_sha256=before,
             input_sha256=sha(output),cuda_initialized=torch.cuda.is_initialized(),new_seeds=0,optimizer_updates=0)
(HERE/"prepare-inputs.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True);assert not receipt["cuda_initialized"]
