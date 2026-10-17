"""Original strict native-control checks, routed through the no-fire source bridge."""
import hashlib
import json
from pathlib import Path
NAME = 'RA16-portability'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def checkpoint_origin(inputs, problem, variant):
    bridge = inputs['restoration_bridge']
    assert variant == NAME == bridge['candidate_variant']
    assert bridge['proof']['candidate_package_sha256'] == inputs['variants'][variant]['package_sha256']
    return bridge['fixtures'][problem]


def compare_native_control(output, inputs, problem):
    origin = checkpoint_origin(inputs, problem, NAME)
    old = read(origin['native_control_result'])['branches'][0]
    updates = [dict(step=new['step'], loss_bits=new['loss_sha256'] == saved['loss_sha256'],
                    semantic_state_bits=new['semantic_state_sha256'] == saved['semantic_state_sha256'],
                    semantic_section_bits=new['semantic_sections'] == saved['semantic_sections'])
               for new, saved in zip(output['update_fingerprints'], old['update_fingerprints'])]
    assert [item['step'] for item in updates] == list(range(1001, 1011))
    checks = dict(restored_semantic_state_bits=output['restored_state_semantic_sha256'] == old['restored_state_semantic_sha256'],
                  loss_bits=output['losses_sha256'] == old['losses_sha256'],
                  endpoint_semantic_state_bits=output['semantic_state_sha256'] == old['semantic_state_sha256'],
                  endpoint_semantic_section_bits=output['semantic_sections'] == old['semantic_sections'],
                  primary_sample_bytes=output['primary_samples_sha256'] == old['primary_samples_sha256'])
    passed = all(checks.values()) and all(all(v for k, v in item.items() if k != 'step') for item in updates)
    return dict(status='PASS' if passed else 'FAIL', origin_result=origin['native_control_result'],
                origin_result_sha256=sha(origin['native_control_result']), checks=checks, per_update=updates)

