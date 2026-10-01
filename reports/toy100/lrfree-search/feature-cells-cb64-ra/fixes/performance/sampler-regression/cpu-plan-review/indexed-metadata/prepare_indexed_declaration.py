"""Freeze the narrow indexed metadata declaration before its saved pilot."""
from datetime import datetime,timezone
import json
from pathlib import Path
import indexed_adapter as adapter


def main():
    if adapter.READY.exists():raise RuntimeError('Declaration already exists')
    # These hashes predate this metadata diagnosis and tie the correction to
    # the exact already-frozen RA4 API, collector, monitor and validation lane.
    frozen={
        'integration/iteration-4/READY.json':'8d5522e1e2bf7ae42baf9666bc63da4da490b03f1b62724b1ab759031341ee87',
        'validation-ra4/source-freeze.json':'c1b09a7acf8ce40690eeaf6e931586c47c08e523dcfc573c7efb1d6f7bb8b443',
        'validation-ra4/screens/source-freeze.json':'329e61941685a35007ab514f59b779f5cf5870fdc2d73cf475dc85eab5e3c681',
        'validation-ra4/screens/READY.json':'1a506432e84f34ea9d4c2abeb1ffb4dcd8b4196a993c70ef057851141a533110',
        'validation-ra4/screens/collect.py':'3500fc335698b16741dc674ad9a0f6c30aa5084637cccd1ba6108b4f69049341',
        'validation-ra4/screens/lane.py':'a2c3b374a2757aca6f27be0237202fb414b7532c707c0f2fa4f58b64fd343bf5',
        'integration/review/monitor_validation.py':'4ee4cae810342b675a39b269112ca907be3ffe88ae9cadab157e4933c487328b',
        'performance/sampler-regression/cpu-plan-review/AXIS-ID-READY.json':'9193167d0184e82ef24ed513dc2d823fb5ee84bcadb3696c1518534d5e39801a',
        'pkg-CB64-RA4/particlegan/training.py':'f5626e4e25f62b37aa76b162ef4d351ed45dad0b5c24dc2c0c385be3d42d6a7a',
        'performance/sampler-regression/cpu-plan-review/composition-review/COMPOSITION-FROZEN.json':'69ae4a80e94503b2094a2d21520403fc45849b70caa398248fd910b0fc1cba97',
    }
    sources={str(adapter.ROOT/path):value for path,value in frozen.items()}
    sources[str(adapter.HARNESS)]='ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c'
    for name in ('indexed_adapter.py','run_indexed_monitor.py','pilot_indexed_adapter.py','prepare_indexed_declaration.py'):
        sources[str(adapter.HERE/name)]=adapter.sha(adapter.HERE/name)
    assert all(adapter.sha(path)==expected for path,expected in sources.items())
    _,_,proof=adapter.collector_trees()
    ready=dict(status='DECLARED_PILOT_REQUIRED_BEFORE_WATCH',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='one expected API metadata literal; all original numerical/data/init/stream/gate checks retained',
        package_sha256='e34bcb21aaa64caa0601cea5dc1f9b8eaebee9578686ebff39b459676063deb2',
        generate_api_ast_sha256='6e404b9fd2382bbbac4677d656dbd8c88457227e8c659d65354411576fac70de',
        exact_source_sha256=sources,collector_ast_proof=proof,validation=str(adapter.VALIDATION),
        original_monitor=str(adapter.MONITOR),new_output=str(adapter.OUTPUT),old_monitor_output_preserved=True,
        cpu_threads=1,cuda_context_allowed=False,numerical_reruns=0,original_quality_gates_unchanged=True,
        pilot_command=['/tmp/pr38-default-env/bin/python','-u','-B',str(adapter.HERE/'pilot_indexed_adapter.py')],
        watch_command=['/tmp/pr38-default-env/bin/python','-u','-B',str(adapter.HERE/'run_indexed_monitor.py'),'--watch'])
    adapter.READY.write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps(dict(ready_sha256=adapter.sha(adapter.READY),ready=str(adapter.READY))),flush=True)


if __name__=='__main__':main()
