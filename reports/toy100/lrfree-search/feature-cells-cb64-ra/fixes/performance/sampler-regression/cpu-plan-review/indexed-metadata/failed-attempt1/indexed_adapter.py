"""Declared RA4 indexed API metadata adapter; frozen numerical checks retained."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.machinery
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
VALIDATION=ROOT/'validation-ra4'
SCREEN_ROOT=VALIDATION/'screens'
COLLECTOR=SCREEN_ROOT/'collect.py'
MONITOR=ROOT/'integration/review/monitor_validation.py'
OUTPUT=ROOT/'integration/review/ra4-indexed-api-monitor'
READY=HERE/'READY.json'
HARNESS=Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
PACKAGE=ROOT/'pkg-CB64-RA4'
STATUS_FIELDS={'collected_at','canonical_fixture_validity','acceptance_status','canonical_gpu_acceptance','validity_reasons'}
CURRENT_OUTPUT=OUTPUT
DECLARATION=None
FRESH_OPTIONS=None
FRESH_DETECTED=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_sha(tree):return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def guard():
    declaration=read(READY)
    for path,expected in declaration['exact_source_sha256'].items():
        assert sha(path)==expected,'Declared adapter source changed: '+path
    root_ready=read(ROOT/'integration/iteration-4/READY.json')
    assert root_ready['package_sha256']==declaration['package_sha256']
    for path,expected in root_ready['numerical_source_sha256'].items():
        assert sha(path)==expected,'Frozen RA4 numerical source changed: '+path
    digest=hashlib.sha256()
    for path in sorted((PACKAGE/'particlegan').rglob('*.py')):
        digest.update(str(path.relative_to(PACKAGE/'particlegan')).encode()+b'\0'+path.read_bytes()+b'\0')
    assert digest.hexdigest()==declaration['package_sha256']
    training=ast.parse((PACKAGE/'particlegan/training.py').read_text())
    cls=next(n for n in training.body if isinstance(n,ast.ClassDef) and n.name=='GANTrainer')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate')
    assert ast_sha(method)==declaration['generate_api_ast_sha256']
    assert 'indices' in [a.arg for a in method.args.args]
    assert 'rows' in [a.arg for a in method.args.kwonlyargs]
    return dict(status='VALID',package_sha256=declaration['package_sha256'],
                ready_sha256=sha(READY),root_ready_sha256=sha(ROOT/'integration/iteration-4/READY.json'),
                source_freeze_sha256=sha(VALIDATION/'source-freeze.json'),
                original_collector_sha256=sha(COLLECTOR),original_monitor_sha256=sha(MONITOR))


def option_assignment(tree):
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect')
    assignments=[n for n in ast.walk(function) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='expected_options' for t in n.targets)]
    assert len(assignments)==1
    return assignments[0]


def collector_trees():
    old=ast.parse(COLLECTOR.read_text());new=deepcopy(old)
    assignment=option_assignment(new)
    values=[k.value for k in assignment.value.keywords if k.arg=='evaluation_generate']
    assert len(values)==1 and isinstance(values[0],ast.Constant) and values[0].value=='plain'
    values[0].value='indexed'
    reverted=deepcopy(new)
    next(k.value for k in option_assignment(reverted).value.keywords if k.arg=='evaluation_generate').value='plain'
    assert ast_sha(reverted)==ast_sha(old),'Collector AST differs beyond declared API expectation'
    return old,new,dict(original_ast_sha256=ast_sha(old),adapted_ast_sha256=ast_sha(new),
        reverted_ast_sha256=ast_sha(reverted),exact_ast_delta=dict(function='collect',assignment='expected_options',
            keyword='evaluation_generate',before='plain',after='indexed'),all_other_ast_nodes_exact=True)


def resolve_declared_api():
    global FRESH_OPTIONS,FRESH_DETECTED
    sys.path.insert(0,str(PACKAGE))
    import particlegan
    assert Path(particlegan.__file__).resolve()==PACKAGE/'particlegan/__init__.py'
    tree=ast.parse(HARNESS.read_text())
    defaults=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='DEFAULT_OPTIONS' for t in n.targets))
    resolver=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_options')
    namespace=dict(inspect=inspect)
    exec(compile(ast.Module(body=[defaults,resolver],type_ignores=[]),str(HARNESS),'exec'),namespace)
    FRESH_OPTIONS,FRESH_DETECTED=namespace['resolve_options'](particlegan,{},read(SCREEN_ROOT/'candidate-options.json'))
    assert FRESH_OPTIONS['evaluation_generate']==FRESH_DETECTED['evaluation_generate']=='indexed'
    assert inspect.signature(particlegan.GANTrainer._generate).parameters['indices'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    # The exact frozen native call appends sampled indices and calls positionally.
    draw=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='draw'
        and [a.arg for a in n.args.args]==['n','ema','latent_seed','noise_seed'])
    context=next(n for n in ast.walk(draw) if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute)
        and i.context_expr.func.attr=='fork_rng' for i in n.items))
    start=next(i for i,n in enumerate(context.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='arguments' for t in n.targets))
    fragment=context.body[start:start+3]
    assert ast.unparse(fragment[1].body[0])=='arguments.append(indices)'
    assert ast.unparse(fragment[2])=='clean = trainer._generate(*arguments)'
    return dict(resolved_options=FRESH_OPTIONS,auto_detected=FRESH_DETECTED,
        canonical_resolver_ast_sha256=ast_sha(resolver),native_positional_fragment_ast_sha256=ast_sha(ast.Module(body=fragment,type_ignores=[])),
        fifth_positional_indices=True)


def canonical(value):return json.dumps(value,sort_keys=True,default=str,allow_nan=False)


def annotate(task,strict,adapted,integrity,proof):
    # Preserve every original output field beyond timestamp and acceptance
    # fields whose values follow directly from the one declared expectation.
    keys=set(strict)|set(adapted)
    for key in keys-STATUS_FIELDS:
        assert canonical(strict.get(key))==canonical(adapted.get(key)),task+': non-adapter field changed: '+key
    result=read(SCREEN_ROOT/'runs'/task/'result.json')
    options=result.get('header',{}).get('options',{})
    mismatch='original resolved options differ: '+str(options)
    remaining=Counter(strict.get('validity_reasons',[]))
    if options==FRESH_OPTIONS:
        assert remaining[mismatch]==1,task+': strict interpretation lacks the declared metadata mismatch'
        remaining[mismatch]-=1
        if not remaining[mismatch]:del remaining[mismatch]
        assert remaining==Counter(adapted.get('validity_reasons',[])),task+': other validity discrepancies changed'
    elif options==dict(FRESH_OPTIONS,evaluation_generate='plain'):
        expected=remaining+Counter({mismatch:1})
        assert expected==Counter(adapted.get('validity_reasons',[])),task+': plain API no longer rejected'
    else:
        assert remaining==Counter(adapted.get('validity_reasons',[])),task+': wrong options changed interpretation'
    assert adapted['primary_status']==strict['primary_status']==result.get('status')
    detected=result.get('header',{}).get('auto_detected',{})
    if detected!=FRESH_DETECTED:
        adapted['validity_reasons'].append('declared indexed API auto-detection receipt differs from the exact frozen resolver')
        adapted.update(canonical_fixture_validity='INVALID',acceptance_status='ERROR',canonical_gpu_acceptance='ERROR')
    old_path=ROOT/'integration/review/ra4-validation-monitor/canonical-receipts/screens/runs'/task/'acceptance-receipt.json'
    adapted['indexed_api_expectation_adapter']=dict(
        scope='declared RA4 indexed API metadata expectation; original numerical/data/init/stream/gate checks retained',
        original_strict_status=strict['acceptance_status'],original_strict_validity=strict['canonical_fixture_validity'],
        original_strict_reasons=strict['validity_reasons'],original_primary_status=strict['primary_status'],
        original_error_receipt_path=str(old_path) if old_path.exists() else None,
        original_error_receipt_sha256=sha(old_path) if old_path.exists() else None,
        exact_declared_api_difference=dict(field='header.options.evaluation_generate',original_expected='plain',
            declared_expected='indexed',actual=options.get('evaluation_generate'),basis='pre-frozen named positional indices API'),
        all_other_collector_ast_nodes_exact=True,all_other_output_fields_exact=True,quality_verdict_unchanged=True,
        numerical_reruns=0,source_guard=integrity,collector_ast_proof=proof)
    return adapted


def modules(lane):
    assert Path(lane.__file__).resolve()==SCREEN_ROOT/'lane.py'
    integrity=guard()
    if FRESH_OPTIONS is None:resolve_declared_api()
    old,new,proof=collector_trees()
    strict=ModuleType('ra4_original_strict_collector');strict.__file__=str(COLLECTOR)
    adapted=ModuleType('ra4_declared_indexed_collector');adapted.__file__=str(COLLECTOR)
    exec(compile(old,str(COLLECTOR),'exec'),strict.__dict__)
    exec(compile(new,'<declared-indexed-expectation:'+str(COLLECTOR)+'>','exec'),adapted.__dict__)
    strict.write=lambda path,value:None
    adapted.write=lambda path,value:None
    return strict,adapted,integrity,proof


def install(output=OUTPUT):
    global CURRENT_OUTPUT
    CURRENT_OUTPUT=Path(output).resolve()
    assert CURRENT_OUTPUT==OUTPUT,'Adapter outputs are pinned to a new RA4 review directory'
    guard();api=resolve_declared_api()
    original=importlib.machinery.SourceFileLoader.exec_module
    def hooked(loader,module):
        if Path(loader.path).resolve()!=COLLECTOR:return original(loader,module)
        lane=sys.modules['lane'];strict,adapted,_,proof=modules(lane)
        # Copy the compiled one-literal collector into the module loaded by the
        # unchanged monitor, then wrap only returned receipt annotations.
        module.__dict__.update({k:v for k,v in adapted.__dict__.items() if k not in ('__name__','__spec__','__loader__')})
        raw_collect=module.collect
        # Functions execute in adapted's namespace; share the monitor's later
        # write redirection at call time, without any lane writes.
        def collect(task,require_attempt=False):
            before=guard();strict.write=lambda path,value:None
            strict_record=strict.collect(task,require_attempt=require_attempt)
            adapted.write=module.write
            assert getattr(module.write,'__name__',None)=='redirected_write','Readonly monitor writer was not installed'
            record=raw_collect(task,require_attempt=require_attempt)
            if record.get('primary_status')=='PENDING':return record
            record=annotate(task,strict_record,record,before,proof)
            out=SCREEN_ROOT/'runs'/task
            module.write(out/'strict-plain-acceptance-receipt.json',strict_record)
            module.write(out/'acceptance-receipt.json',record)
            guard()
            import torch
            assert not torch.cuda.is_initialized()
            return record
        module.collect=collect
        # Correct the inherited monitor's checker-identity claims in the NEW
        # output files. Original source/report logic and old outputs stay intact.
        frame=inspect.currentframe().f_back
        while frame and Path(frame.f_globals.get('__file__','/')).resolve()!=MONITOR:frame=frame.f_back
        assert frame is not None
        namespace=frame.f_globals;original_write=namespace['write']
        def declared_write(path,value):
            value=deepcopy(value)
            if isinstance(value,dict):
                if 'write_redirection_only' in value:value['write_redirection_only']=False
                if 'original_canonical_checks_unchanged' in value:value['original_canonical_checks_unchanged']=False
                annotation=dict(
                    declaration_ready_sha256=sha(READY),exact_ast_delta=proof['exact_ast_delta'],
                    all_other_canonical_checks_unchanged=True,quality_gates_unchanged=True,
                    numerical_reruns=0,original_collector_ast_sha256=proof['original_ast_sha256'],
                    adapted_collector_ast_sha256=proof['adapted_ast_sha256'])
                annotation.update(value.get('indexed_api_expectation_adapter',{}))
                value['indexed_api_expectation_adapter']=annotation
            original_write(path,value)
        namespace['write']=declared_write
    importlib.machinery.SourceFileLoader.exec_module=hooked
    return api
