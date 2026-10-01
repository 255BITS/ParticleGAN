"""Stdlib post-exit closure; preserve raw outcomes and every attempt byte."""
import argparse
from common import *

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-freeze',type=Path,required=True);parser.add_argument('--root-go',type=Path,required=True)
args=parser.parse_args()
assert not (FINAL_AREA/'receipt.json').exists() and not (FINAL_AREA/'FINAL-FROZEN.json').exists()
seal=read(args.input_freeze);verify(seal['source_and_input_sha256'])
exit_record=read(HERE/'EXIT-attempt1.json');launch=read(HERE/'LAUNCH-attempt1.json')
assert exit_record['returncode']==0 and exit_record['log_closed'] and exit_record['log_sha256']==sha(HERE/'attempt1.log')
require_exited(launch['pid'],launch['startticks'])
result=read(HERE/'accepted-attempt1/receipt.json');assert result['status']==result['evidence_status']=='VALID'
assert result['total_original_jobs']==19 and result['total_canonical_screens']==16 and result['planned_and_actual_PT_loads']==14
assert result['MNIST']['evidence_status']=='VALID' and all(v['evidence_status']=='VALID' for v in result['replay'].values())
assert result['input_freeze_sha256']==sha(args.input_freeze) and result['root_GO_sha256']==sha(args.root_go)
value=dict(result)
value.update(closed_utc=now(),authoritative_runtime_receipt_sha256=sha(HERE/'accepted-attempt1/receipt.json'),
    launch_sha256=sha(HERE/'LAUNCH-attempt1.json'),exit_sha256=sha(HERE/'EXIT-attempt1.json'),logs_closed=True,
    no_general_package_promotion=True,quality_failures_preserved=True)
write_new(FINAL_AREA/'receipt.json',value)
mnist=value['MNIST']['final']['metrics'];active=mnist['active_embedding']
lines=['# RA11 final regression artifact review','','Evidence: **VALID**. Source/API/toy/grid proofs are reused; original MNIST and replay artifacts are checked without model execution.','',
       '| Record | Original result | Evidence |','|---|---|---|',
       '| MNIST training | COMPLETE | VALID |']
for problem,row in value['replay'].items():lines.append(f"| {problem} replay | {row['primary_status']} | VALID |")
for row in value['canonical']['records']:lines.append(f"| {row['task']} | {row['primary_status']} | VALID |")
lines+=['',f"MNIST quality remains a regression: active embedding Fréchet {active['embedding_frechet']:.6f}, precision {active['embedding_precision']:.6f}, recall {active['embedding_recall']:.6f}; confident class coverage {mnist['confident_class_coverage']}, pixel clipping {mnist['pixel_clipping_fraction']:.6f}.",
        '', 'Both strict toy and Grid100 passed. Remaining screen failures and MNIST degradation are preserved; this validates the experiment and does not recommend general package replacement.',
        '', 'Ten MNIST checkpoint loads and four replay endpoint loads use CPU storages and original GPU device tags. No constructors, forwards, draws, training updates, scorer runs or CUDA contexts. Only the original eval_seconds replay exclusion applies.',
        '', 'Reset/inheritance/lineage identities are checked only at actual saved reaction boundaries; historical or compressed row IDs and isolation identities are not reconstructed.']
with (FINAL_AREA/'REPORT.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
mapping=dict(seal['source_and_input_sha256'])
for path in sorted(FINAL_AREA.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.name!='FINAL-FROZEN.json':pin(mapping,path)
verify(mapping)
write_new(FINAL_AREA/'FINAL-FROZEN.json',dict(status='POST_EXIT_VALID_FINAL_RA11_REGRESSION_REVIEW',utc=now(),receipt_sha256=sha(FINAL_AREA/'receipt.json'),files=mapping,
    logs_closed=True,quality_FAILs_preserved=True,CPU_only=True,no_source_or_artifact_mutations=True))
print(json.dumps(dict(status='VALID',receipt_sha256=sha(FINAL_AREA/'receipt.json'),freeze_sha256=sha(FINAL_AREA/'FINAL-FROZEN.json'),guards=len(mapping))))
