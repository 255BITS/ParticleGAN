"""Bind the closed authorized invocation/provenance; stdlib only, no new measurements."""
from datetime import datetime, timezone
from pathlib import Path
from bindings import HERE, ROOT, read, sha, write_new, merge_maps, verify


def main():
    attempt = HERE/'attempt1'
    if (attempt/'FINAL-FROZEN.json').exists():
        raise SystemExit('preserve the existing authoritative seal')
    source = read(HERE/'SOURCE-FROZEN.json')
    frozen = read(attempt/'FROZEN.json')
    assert frozen['status']=='PASS' and frozen['process_exited'] and frozen['log_closed']
    review = ROOT/'integration/review/training-regression/post-ra10-quality/mean-law-source-review/helper-preexecution-attempt2'
    assert sha(review/'receipt.json')=='9ea2acbefd96c944caffe748b74d38ca999da80941e529b0943564fb0ab65224'
    assert sha(review/'FROZEN.json')=='0d46cf7b98288fa21e74eaa09401908a2a854f5fdac4b353485a3231cbe3110b'
    proof = read(review/'FROZEN.json')
    maps = [source['source_and_input_sha256'],frozen['local_sha256']]
    for key,value in proof.items():
        if isinstance(value,dict) and key.endswith('sha256'):
            maps.append(value)
    guards=merge_maps(*maps)
    verify(guards)
    launch,exit_record=read(HERE/'LAUNCH-attempt1.json'),read(HERE/'EXIT-attempt1.json')
    assert launch['status']=='AUTHORIZED_ONE_INVOCATION' and launch['CPU_only']
    assert launch['source_seal_sha256']==sha(HERE/'SOURCE-FROZEN.json')
    assert launch['pid']==exit_record['pid']==1131552
    assert launch['startticks']==exit_record['startticks']==168065126
    assert exit_record['status']=='PROCESS_EXITED' and exit_record['exit_code']==0
    assert exit_record['log_closed'] and exit_record['invocations']==1
    process=Path('/proc')/str(launch['pid'])/'stat'
    if process.exists():
        fields=process.read_text().rsplit(')',1)[1].split()
        assert int(fields[19])!=launch['startticks'] or fields[0]=='Z'
    result=read(attempt/'result.json')
    assert result['status']=='COMPLETE' and result['checks']['chart_fits']==1
    assert result['checks']['PT_objects_loaded']==1
    local={str(path):sha(path) for path in HERE.rglob('*') if path.is_file()}
    local[str(review/'receipt.json')]=sha(review/'receipt.json')
    local[str(review/'FROZEN.json')]=sha(review/'FROZEN.json')
    verify(guards)
    seal=dict(status='PASS',scope='AUTHORITATIVE_POST_EXIT_FIXED_DIAGNOSTIC_SEAL',
        utc=datetime.now(timezone.utc).isoformat(),receipt_sha256=sha(attempt/'receipt.json'),
        result_sha256=sha(attempt/'result.json'),source_seal_sha256=sha(HERE/'SOURCE-FROZEN.json'),
        independent_helper_review_sha256=sha(review/'receipt.json'),
        source_and_input_sha256=guards,local_sha256=local,
        process_identity=dict(pid=launch['pid'],startticks=launch['startticks'],exited=True),
        exact_invocations=1,log_closed=True,source_and_original_artifacts_unchanged=True,
        original_grid_fixture_status='VALID',original_grid_quality='FAIL',
        quality_rescored=False,production_qualification=False)
    write_new(attempt/'FINAL-FROZEN.json',seal)
    print('AUTHORITATIVE_CLOSED_PASS',sha(attempt/'receipt.json'),sha(attempt/'FINAL-FROZEN.json'),len(guards),len(local),flush=True)


if __name__=='__main__':
    main()
