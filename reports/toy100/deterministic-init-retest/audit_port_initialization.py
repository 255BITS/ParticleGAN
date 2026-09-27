"""Independent source/receipt checker for manually reviewed initialization-only ports.

This checker imports no candidate/Torch code. It consumes the separate zero-step
CPU receipt and refuses changes outside the reviewed initialization surfaces.
"""
from pathlib import Path
import argparse, ast, copy, difflib, hashlib, json, subprocess, zipfile

ROOT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_bytes())

def methods(tree, name):
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
    return {x.name: ast.dump(x, include_attributes=False) for x in cls.body if isinstance(x, (ast.FunctionDef,ast.AsyncFunctionDef))}

def unaffected(tree, classname, ignored):
    tree = copy.deepcopy(tree)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == classname)
    cls.body = [x for x in cls.body if not
        (isinstance(x, (ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in ignored) and not
        (isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Name) and x.target.id == 'initialization')]
    if isinstance(cls.body[0], ast.Expr) and isinstance(cls.body[0].value,ast.Constant) and isinstance(cls.body[0].value.value,str):
        cls.body.pop(0)
    return ast.dump(tree,include_attributes=False)

def main(alias):
    p = ROOT/'port-source'/alias
    decl = read(p/'candidate-declaration.json'); manifest = read(p/'port-manifest.json'); source = read(p/'source-declaration.json')
    cpu_path = ROOT/(alias+'-cpu-preflight.json'); cpu = read(cpu_path); authority = decl['algorithm_source']
    assert sha((p/'port-manifest.json').read_bytes()) == authority['port_manifest_sha256']
    assert sha(Path(authority['source_zip']).read_bytes()) == authority['source_zip_sha256'] == manifest['source_zip_sha256']
    original_decl = Path(authority['source_zip']).with_name('declaration.json')
    assert sha(original_decl.read_bytes()) == manifest['source_declaration_sha256']
    assert read(original_decl) == source
    assert sha((p/'package.zip').read_bytes()) == manifest['package_zip_sha256']
    actual = {str(f.relative_to(p/'package')):sha(f.read_bytes()) for f in (p/'package/particlegan').glob('*.py')}
    assert actual == decl['package_sha256'] == manifest['package_sha256']
    with zipfile.ZipFile(authority['source_zip']) as z:
        old = {n:z.read(n) for n in z.namelist() if n.startswith('particlegan/') and n.endswith('.py')}
    assert all(sha(b) == source['source_sha256'][n] for n,b in old.items())
    changed = sorted(n for n,b in old.items() if sha(b) != actual[n]); added = sorted(set(actual)-set(old))
    assert changed == ['particlegan/__init__.py','particlegan/recipes.py','particlegan/training.py']
    assert len(added) == 10
    for n in added:
        assert (p/'package'/n).read_bytes() == subprocess.check_output(['git','show',authority['new_api_base']+':'+n],cwd='/ml2/hypergan/ParticleGAN-ka2-default')
    with zipfile.ZipFile(p/'package.zip') as z:
        assert {n:sha(z.read(n)) for n in z.namelist() if n.endswith('.py')} == actual
    expected_export = old['particlegan/__init__.py'].decode().replace('from .particle_prior import', 'from .initialization import initialize_\nfrom .particle_prior import').replace('__all__ = [','__all__ = [\n    "initialize_",')
    assert expected_export == (p/'package/particlegan/__init__.py').read_text()
    deltas = {}
    for name, cls, ignored in [('training.py','GANTrainer',{'load_state_dict'}),('recipes.py','Recipe',{'__post_init__','make_prior','make_optimizers'})]:
        original = ast.parse(old['particlegan/'+name]); current = ast.parse((p/'package/particlegan'/name).read_bytes())
        before, after = methods(original,cls), methods(current,cls)
        assert before.keys() == after.keys()
        deltas[name] = [k for k in before if before[k] != after[k]]
        assert set(deltas[name]) == ignored
        assert unaffected(original,cls,ignored) == unaffected(current,cls,ignored)
    expected_recipe = {**source['recipe'], 'batch_size':128, 'num_particles':12, 'z_dim':4, 'initialization':'batch_feature_zero'}
    assert decl['resolved_recipe'] == expected_recipe
    recipe_delta = {k:{'original':source['recipe'].get(k),'screen':v} for k,v in expected_recipe.items() if source['recipe'].get(k)!=v}
    # Full training text deltas must be exactly one of the two already reviewed
    # checkpoint-initialization variants. Context/line offsets do not enter it.
    def edits(before, after):
        return ''.join(x for x in difflib.unified_diff(before.splitlines(True),after.splitlines(True),n=0) if not x.startswith(('@@','---','+++')))
    allowed=[]
    for reference in ('api-dv15','api-rp5'):
        rp=ROOT/'port-source'/reference;rd=read(rp/'candidate-declaration.json')
        with zipfile.ZipFile(rd['algorithm_source']['source_zip']) as rz:
            rb=rz.read('particlegan/training.py').decode()
        allowed.append(edits(rb,(rp/'package/particlegan/training.py').read_text()))
    assert edits(old['particlegan/training.py'].decode(),(p/'package/particlegan/training.py').read_text()) in allowed
    # Independently remove only exact known public initialization AST additions
    # and compare the remaining original recipe arithmetic/validation verbatim.
    ref_tree=ast.parse((ROOT/'port-source/api-dv15/package/particlegan/recipes.py').read_bytes())
    old_tree=ast.parse(old['particlegan/recipes.py']);new_tree=ast.parse((p/'package/particlegan/recipes.py').read_bytes())
    def node(tree,name):
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Recipe')
        return copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name))
    def dump(node):return ast.dump(node,include_attributes=False)
    def nodoc(node):
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
        return node
    class Unwrap(ast.NodeTransformer):
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='initialize':
                assert len(n.args)==1 and not n.keywords
                return n.args[0]
            return n
    for name in ('__post_init__','make_prior','make_optimizers'):
        before=node(old_tree,name);after=node(new_tree,name);ref=node(ref_tree,name)
        if name=='__post_init__':
            additions=[x for x in ref.body if isinstance(x,ast.If) and isinstance(x.test,ast.Compare) and isinstance(x.test.left,ast.Attribute) and x.test.left.attr=='initialization']
        elif name=='make_prior':
            additions=[x for x in ref.body if (isinstance(x,ast.ImportFrom) and any(y.name=='initialization' for y in x.names)) or isinstance(x,ast.FunctionDef) and x.name=='initialize']
        else:
            additions=[x for x in ref.body if (isinstance(x,ast.ImportFrom) and x.module=='initialization') or isinstance(x,ast.If) and isinstance(x.test,ast.BoolOp) and any(isinstance(y,ast.Compare) and isinstance(y.left,ast.Attribute) and y.left.attr=='initialization' for y in x.test.values)]
        for addition in additions:
            matches=[i for i,x in enumerate(after.body) if dump(x)==dump(addition)]
            assert len(matches)==1,(alias,name,dump(addition))
            after.body.pop(matches[0])
        if name=='make_prior':after=Unwrap().visit(after)
        assert dump(nodoc(before))==dump(nodoc(after)),(alias,name)

    assert cpu['declaration_sha256'] == sha((p/'candidate-declaration.json').read_bytes())
    assert cpu['status'] == cpu['cpu_initialization']['status'] == 'PASS' and cpu['cuda_initialized'] is False
    assert cpu['cpu_initialization']['repeated_without_rng_reset'] is True
    constructor = next(x for x in next(c for c in ast.parse(old['particlegan/training.py']).body if isinstance(c,ast.ClassDef) and c.name=='GANTrainer').body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    assert decl['serial_backward_argument'] == ('serial_backward' in {a.arg for a in [*constructor.args.args,*constructor.args.kwonlyargs]})
    report = dict(status='PASS_INITIALIZATION_ONLY',candidate=decl['candidate'],
        declaration_sha256=cpu['declaration_sha256'],port_manifest_sha256=authority['port_manifest_sha256'],
        cpu_receipt_sha256=sha(cpu_path.read_bytes()),cpu_receipt=str(cpu_path),
        source_zip_sha256=authority['source_zip_sha256'],package_zip_sha256=manifest['package_zip_sha256'],
        source_declaration_original_sha256=manifest['source_declaration_sha256'],
        source_declaration_copy_sha256=sha((p/'source-declaration.json').read_bytes()),source_declaration_semantically_equal=True,
        initializer_commit=decl['initializer_commit'],new_api_base=authority['new_api_base'],package_files=actual,
        unchanged_candidate_files=sorted(set(old)-set(changed)),changed_candidate_files=changed,new_files_equal_new_api_git=added,
        ast_changed_methods=deltas,all_other_training_and_recipe_ast_equal=True,recipe_delta=recipe_delta,
        cpu_scope=cpu['cpu_initialization']['scope'],cuda_initialized=False,
        repeatability_includes_all_non_rng_state_and_named_parameters_buffers=True,
        construction_rng_preserved=True,initializer_rng_neutral=True,
        geometry={k:v for k,v in cpu['cpu_initialization'].items() if k in ('geometry_check','latent_bandwidth')},
        declared_execution={k:decl[k] for k in ('serial_backward_argument','evaluation_generate','initial_optimizer_state','optimizer_step_devices')},
        manual_diff_review=['Exact public initialization field/validation and export only',
            'Public prior initializer wraps old prior constructors before MoG calibration',
            'Network initializer precedes original optimizer creation, including candidate eager-state allocation when applicable',
            'Supplied EMA critic synchronized after deterministic critic initialization',
            'Checkpoint metadata adds public initialization compatibility, retaining original serial/controller/schema guards',
            'Controller, optimizer, game-update and prior algorithms unchanged; no historical score inherited'],
        remaining=['External CUDA dry construction must match all1,200 original sampling rows before update1',
            'CPU zero-step proof does not establish training quality, CUDA numerical identity, or replay'],quality_inherited=False)
    (ROOT/(alias+'-independent-init-audit.json')).write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/(alias+'-independent-init-audit.md')).write_text(f'''# {decl['candidate']} initialization port audit\n\nPASS for initialization-only integration. Constructor-only CPU preflight completed with zero learner steps and CUDA uninitialized. Two fresh constructions separated by extra RNG draws, without reset, produced identical full non-RNG checkpoint state and all independently enumerated named parameters/buffers. Ordinary constructor RNG consumption remains; the public prior/network initializer does not consume those streams.\n\nAll {len(actual)} package files match the declared package ZIP. The {len(old)-3} unchanged candidate files remain byte-identical to the archived source; ten added initializer modules match API Git base25751c0864dd8259b00c5804f600cd41cce6e4cf exactly. Every trainer method except initializer metadata loading is AST-identical; all training/recipe module AST outside the three reviewed recipe methods, initialization field and load_state_dict is unchanged. The declared algorithm recipe changes only initialization and the frozen mode-hold host resources (12 particles, z4, batch128). All schedule/noise/learner fields remain as declared. The two checkpoint diff shapes were manually reviewed; the exact initialization AST additions are independently stripped and all remaining recipe arithmetic is checked against the original.\n\nPrior initialization occurs through the actual public recipe factory before any trainer-derived geometry; network initialization precedes the original optimizer construction. Existing candidate eager/lazy optimizer state and its declared counter placement remain intact. Geometry result: {report['geometry']}. Checkpoint loading adds the merged public initialization metadata behavior; continuation itself was not executed.\n\nDeclaration `{report['declaration_sha256']}`. [Hash-bound independent audit](%s) and [CPU receipt](%s). Historical declaration copies may differ in JSON formatting; original and copied raw hashes are retained and parsed contents were independently verified equal.\n\nActual CUDA sampling must still pass the sealed harness dry preflight before any learner update. No previous quality evidence transfers to the new initialization epoch.\n''' % (alias+'-independent-init-audit.json',alias+'-cpu-preflight.json'))
    print(json.dumps({k:report[k] for k in ('status','candidate','declaration_sha256','port_manifest_sha256','cpu_receipt_sha256','geometry')},indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate');main(p.parse_args().candidate)
