"""Freeze learned CUDA runner inputs without initializing Torch or running quality."""
import common
from common import ROOT, PREV, OLD, SEED, STEPS, DEVICE, GPU_UUID, PROBLEMS, VARIANTS
from common import require, sha, package_sources, package_digest, write_json, utc_now, gpu_inventory
import ast
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import sys

PYTHON = '/tmp/pr38-default-env/bin/python'
LOCAL = ('common.py', 'run_training.py', 'replay.py', 'prepare.py', 'summarize.py', 'PROTOCOL.md', 'COMMANDS.md', 'INPUTS.json')
EXPECTED_CANDIDATE_PACKAGE = '13f5bbe4de824e6899cb28ee4dff8f35d74bbaa6243bfe9c18b8df44d173a1ce'
EXPECTED_CANDIDATE_CONFIG = 'd2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7'


def main():
    require(not list((ROOT/'training').rglob('checkpoint-*.pt'))
            and not list((ROOT/'training').rglob('error.json'))
            and not list((ROOT/'replay').rglob('branch-*.pt'))
            and not list(ROOT.glob('replay-*.json')), 'cannot refreeze after numerical execution evidence exists')
    old_path=OLD/'candidate-freeze.json'
    old=json.loads(old_path.read_text())
    variants=deepcopy(old['variants'])
    candidate=variants['CB64-RA']
    require(candidate['package_sha256']==EXPECTED_CANDIDATE_PACKAGE,'old freeze candidate package identity differs')
    require(candidate['config_sha256']==EXPECTED_CANDIDATE_CONFIG,'old freeze candidate config identity differs')
    for name,selected in variants.items():
        require(package_digest(selected['package_root'])==selected['package_sha256'],f'{name} package bytes changed')
        require(package_sources(selected['package_root'])==selected['source_sha256'],f'{name} source map changed')
        require(sha(selected['config_path'])==selected['config_sha256'],f'{name} config bytes changed')
        require(json.loads(Path(selected['config_path']).read_text())==selected['config'],f'{name} config contents changed')
    ready=old['ready_receipt']
    require(sha(ready['path'])==ready['sha256'],'candidate implementation READY bytes changed')
    ready_contents=json.loads(Path(ready['path']).read_text())
    for name,digest in candidate['source_sha256'].items():
        path=str(Path(candidate['package_root'])/'particlegan'/name)
        require(ready_contents['hashes'][path]==digest,f'implementation READY byte map differs: {name}')
    data_receipt=json.loads((PREV/'data-receipt.json').read_text())
    files=[old_path,Path(ready['path']),OLD/'run_training.py',OLD/'replay.py',OLD/'PROTOCOL.md',
           PREV/'models_metrics.py',PREV/'data-receipt.json',PREV/'evaluator.pt',
           PREV/'data/toy-stream.pt',PREV/'data/image-stream.pt']
    require(sha(PREV/'data/toy-stream.pt')==data_receipt['toy_stream_sha256'],'toy stream differs from original receipt')
    require(sha(PREV/'data/image-stream.pt')==data_receipt['image_stream_sha256'],'image stream differs from original receipt')
    # The data receipt's evaluator_model_sha256 hashes model tensor values;
    # candidate-freeze.previous_sources records serialized evaluator file bytes.
    require(sha(PREV/'evaluator.pt')==old['previous_sources']['evaluator.pt'],
            'serialized real-only evaluator differs from original file-byte freeze')
    for name,expected in data_receipt['files'].items():
        path=PREV/'data/MNIST/raw'/name
        require(sha(path)==expected,f'MNIST byte receipt differs: {name}')
        files.append(path)
    initial={}
    for problem in PROBLEMS:
        path=PREV/'runs'/problem/'E22/config.json'
        previous=json.loads(path.read_text());files.append(path)
        require(previous['visible_devices']=='0' and previous['device_name']=='NVIDIA RTX A6000',
                f'expected initialization receipt is not original GPU E22: {problem}')
        require(previous['source_hashes']['models_metrics.py']==sha(PREV/'models_metrics.py'),
                f'architecture/metric source differs from GPU initialization receipt: {problem}')
        initial[problem]={k:previous[k] for k in ('initial_generator_sha256','initial_critic_sha256','initial_prior_sha256')}
    inputs=dict(prepared_at=utc_now(),variants=variants,
                origin_candidate_freeze=dict(path=str(old_path),sha256=sha(old_path)),
                implementation_ready=dict(path=ready['path'],sha256=ready['sha256'],source_byte_map_verified=True),
                read_only_file_sha256={str(p):sha(p) for p in files},expected_initial_hashes=initial,
                evaluator_hash_scopes=dict(file_sha256=sha(PREV/'evaluator.pt'),
                    archived_model_tensor_hash=data_receipt['evaluator_model_sha256']),
                gpu_policy=dict(device=DEVICE,physical_gpu=0,uuid=GPU_UUID,cuda_memory_fraction=.2,cpu_threads=2),
                training=dict(seed=SEED,updates=STEPS,N=1024,z_dim=128,batch=128,serial_backward=True,
                              checkpoints=list(common.CHECKPOINTS)),
                replay=dict(start_step=1000,updates=10,branches=2,excluded_observational_fields=['birth_death.last.eval_seconds']))
    write_json(ROOT/'INPUTS.json',inputs)
    syntax={}
    for name in LOCAL:
        if name.endswith('.py'):
            tree=ast.parse((ROOT/name).read_text(),filename=name)
            compile(tree,name,'exec');syntax[name]='PASS'
    # A static AST comparison proves the metric formulas were retained.
    old_tree=ast.parse((OLD/'run_training.py').read_text())
    new_tree=ast.parse((ROOT/'run_training.py').read_text())
    def helpers(tree):
        result={node.name:ast.dump(node,include_attributes=False) for node in tree.body
                if isinstance(node,ast.FunctionDef) and node.name in ('frechet','embedding_score','diagnostics')}
        image=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ImageEvaluation')
        result.update({f'ImageEvaluation.{node.name}':ast.dump(node,include_attributes=False) for node in image.body
                       if isinstance(node,ast.FunctionDef) and node.name in ('score','evaluate')})
        return result
    old_helpers,new_helpers=helpers(old_tree),helpers(new_tree)
    metric_identity={k:old_helpers[k]==new_helpers[k] for k in old_helpers}
    require(all(metric_identity.values()),'learned metric/helper formulas changed from frozen CPU runner')
    inventory=gpu_inventory()
    require(next(r for r in inventory if r['physical_index']==0)['uuid']==GPU_UUID,'physical GPU 0 identity differs')
    local_hashes={name:sha(ROOT/name) for name in LOCAL}
    freeze=dict(status='FROZEN_PRE_EXECUTION',frozen_at=utc_now(),local_source_sha256=local_hashes,
                candidate_package_sha256=EXPECTED_CANDIDATE_PACKAGE,candidate_config_sha256=EXPECTED_CANDIDATE_CONFIG,
                variants={name:dict(package_sha256=v['package_sha256'],config_sha256=v['config_sha256'])
                          for name,v in variants.items()},
                package_digest_convention='sorted *.py paths relative to particlegan + NUL + file bytes + NUL')
    write_json(ROOT/'SOURCE-FREEZE.json',freeze)
    commands=[]
    for problem in PROBLEMS:
        for variant in VARIANTS:
            commands.append([PYTHON,'-B','-u',str(ROOT/'run_training.py'),'--problem',problem,'--variant',variant])
    for variant in VARIANTS:commands.append([PYTHON,'-B','-u',str(ROOT/'replay.py'),variant])
    commands.append([PYTHON,'-B',str(ROOT/'summarize.py')])
    versions={}
    for package in ('torch','torchvision','numpy','scipy'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    receipt=dict(status='PREPARED_NO_NUMERICAL_EXECUTION',prepared_at=utc_now(),syntax_checks=syntax,
                 unchanged_metric_helpers=metric_identity,all_frozen_input_bytes_verified=True,
                 initial_hashes_recorded_but_runtime_initialization_not_executed=initial,
                 source_freeze_sha256=sha(ROOT/'SOURCE-FREEZE.json'),inputs_sha256=sha(ROOT/'INPUTS.json'),
                 local_source_sha256=local_hashes,read_only_file_sha256=inputs['read_only_file_sha256'],
                 candidate_implementation_ready=inputs['implementation_ready'],
                 hardware_inventory_read_only=inventory,python=sys.version,package_versions=versions,
                 numerical_execution_owner='root coordinator only',numerical_execution_started=False,
                 preparation_command=[sys.executable,*sys.argv],planned_serial_commands=commands)
    write_json(ROOT/'preparation-receipt.json',receipt)
    write_json(ROOT/'READY.json',dict(status='READY',ready_at=utc_now(),numerical_execution_started=False,
                                    source_freeze_sha256=sha(ROOT/'SOURCE-FREEZE.json'),
                                    preparation_receipt_sha256=sha(ROOT/'preparation-receipt.json'),
                                    local_source_sha256=local_hashes,planned_serial_commands=commands,
                                    physical_gpu=0,cuda_memory_fraction=.2,cpu_threads=2))
    print(json.dumps(dict(status='READY',source_freeze_sha256=sha(ROOT/'SOURCE-FREEZE.json'),
                          preparation_receipt_sha256=sha(ROOT/'preparation-receipt.json'),
                          syntax_checks=syntax,unchanged_metric_helpers=metric_identity,
                          numerical_execution_started=False)),flush=True)


if __name__=='__main__':main()
