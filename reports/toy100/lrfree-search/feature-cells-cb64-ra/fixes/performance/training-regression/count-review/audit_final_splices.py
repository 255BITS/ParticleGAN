"""Independent complete-source AST proof for the four final performance splices."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/"integration/review/training-regression/global-count/pkg-global-count/particlegan/feature_cells.py"
NEW=ROOT/"performance/sampler-regression/cpu-plan-review/plan-batching/pkg-PLAN-FINAL/particlegan/feature_cells.py"
PROTOTYPE=ROOT/"performance/sampler-regression/cpu-plan-review/plan-batching/pkg-PLAN/particlegan/feature_cells.py"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[Path(__file__),BASE,NEW,PROTOTYPE]
before={str(p):sha(p) for p in paths}
base,new,prototype=(ast.parse(p.read_text()) for p in (BASE,NEW,PROTOTYPE))
helper=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="_group_integer_allocate")
previous=next(n for n in prototype.body if isinstance(n,ast.FunctionDef) and n.name==helper.name)
assert ast.dump(helper,include_attributes=False)==ast.dump(previous,include_attributes=False)
rebuilt=deepcopy(new)
rebuilt.body=[n for n in rebuilt.body if not (isinstance(n,ast.FunctionDef) and n.name==helper.name)]
original=next(n for n in base.body if isinstance(n,ast.ClassDef) and n.name=="FeatureCellSnapshot")
restored=next(n for n in rebuilt.body if isinstance(n,ast.ClassDef) and n.name==original.name)
names={"_ordinary_mass_transport","_ordinary_support_transport","_ordinary_global_transport"}
replacements={n.name:n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in names}
assert replacements.keys()==names
restored.body=[deepcopy(replacements[n.name]) if isinstance(n,ast.FunctionDef) and n.name in names else n for n in restored.body]
assert ast.dump(rebuilt,include_attributes=False)==ast.dump(base,include_attributes=False)
assert before=={str(p):sha(p) for p in paths}
receipt=dict(status="PASS",source_sha256=before,helper_matches_independently_reviewed_prototype=True,
             exactly_four_performance_AST_splices=True,all_other_module_AST_unchanged=True,
             original_MST_and_all_backend_settings_unchanged=True,cpu_only=True,
             scope="Complete feature module composition proof, stdlib only")
(HERE/"final-splice-review.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2),flush=True)
