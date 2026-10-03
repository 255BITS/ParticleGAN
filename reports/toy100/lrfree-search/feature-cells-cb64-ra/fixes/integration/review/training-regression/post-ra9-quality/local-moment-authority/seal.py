"""Close the exclusive one-run metadata receipt after its process has exited."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    assert not (HERE/'FROZEN.json').exists()
    pre=json.loads((HERE/'PREPARATION-FROZEN.json').read_text())
    for path,value in pre['files'].items():assert sha(path)==value
    path=HERE/'attempt1/result.json';result=json.loads(path.read_text())
    assert result['status']=='PASS_METADATA_AUDIT'
    assert result['new_training_steps']==result['model_forwards']==result['new_emissions']==result['new_actions']==0
    decision=result['proposal_decision'];step=result['inspected_reaction_step'];slots=result['ordinary_copy_slots']
    last=result['last_reaction'];closed=datetime.now(timezone.utc).isoformat()
    consequence=('The reserved count-slot moment proposal is rejected for this final recorded reaction: zero ordinary copy births leave no destination in any supported group to convert.'
                 if slots==0 else
                 'Some ordinary copy births exist, but this saved aggregate metadata cannot determine whether their fine cells/groups reach the biased supports. No targeted authority is certified by this parse.')
    report=f'''# Final native count-slot authority

Metadata audit: **PASS**. Proposal decision: **{decision}**.

- Completed update: {result['completed_steps']}; recorded reaction: {step}, snapshot {result['snapshot_serial']}.
- Actual chart: {result['actual_cells']} cells, rank {result['metric_rank']}, {result['groups']} real topology groups.
- Ordinary copy slots: **{slots}**; novel births: {result['ordinary_novel_births']}.
- Ordinary actions: {last['ordinary_moves']} / shared budget {result['shared_ordinary_budget']}; isolation: {last['iso_moves']}.
- Ordinary discoveries: {last['ordinary_discoveries']}; excess/deficit cells: {last['ordinary_excess_cells']}/{last['ordinary_deficit_cells']}.
- Common family: {result['conditional_family_multiplicity']} hypotheses at cutoff {result['count_cutoff']}.

{consequence}

The proposal only changes births the existing count controller already schedules.
The ordinary budget is an upper bound, not an authorization to fill unused slots.
Unchanged moment/radial noise or lack of count discovery cannot create a birth.
The cumulative run counters do not establish currently available destinations.

The saved backend retains aggregate group count/threshold metadata and last action
summaries; it does not retain the historical fitted chart, cell/group assignments
or ordinary per-cell quota and child/parent ledger. This result concerns only the
recorded reaction above. It does not infer earlier reaction opportunities, local
moment equality, per-group bias significance or current-chart feasibility.

Helper/source/final-state hashes were frozen at {pre['frozen_utc']} before this
one checkpoint parse and remain exact. No model tensor values were evaluated:
only saved shapes, typed diagnostic trees and counters were inspected. There
were no model constructors/forwards, chart refits, emissions, new actions,
optimizer steps, training, RNG draws/restores or CUDA contexts. Global CPU RNG
was unchanged. This is CPU metadata analysis, not numerical replay or quality
acceptance. The original full grid verdict and independent moment-feasibility
diagnostic remain separate. Closed after process exit: {closed}.
'''
    with (HERE/'REPORT.md').open('x') as target:target.write(report)
    receipt=dict(status='PASS_METADATA_AUDIT',proposal_decision=decision,result_path=str(path),result_sha256=sha(path),
                 source_input_sha256=pre['files'],preparation_sha256=sha(HERE/'PREPARATION-FROZEN.json'),
                 process_exited=True,closed_utc=closed,CPU_only=True,new_training_steps=0,new_emissions=0,
                 new_actions=0,model_forwards=0,numerical_replay=False,quality_verdict=None)
    with (HERE/'receipt.json').open('x') as target:target.write(json.dumps(receipt,indent=2)+'\n')
    files={str(p):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    frozen=dict(status='POST_EXIT_FROZEN_SINGLE_METADATA_PARSE',files=files,source_input_sha256=pre['files'],closed_utc=closed,
                proposal_decision=decision,quality_verdict=None)
    with (HERE/'FROZEN.json').open('x') as target:target.write(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],proposal_decision=decision,receipt_sha256=sha(HERE/'receipt.json'),
                         frozen_sha256=sha(HERE/'FROZEN.json'))),flush=True)


if __name__=='__main__':main()
