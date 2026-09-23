from reports.transfer_suite.host_replication import run
from benchmarks.transfer_suite.linear_skip_refinement_research import LinearSkipCritic
from benchmarks.transfer_suite.smooth_critic_research import SmoothFourierCritic


def test_plan_replays_only_archived_passing_trials():
    jobs = run.plan()
    assert len(jobs) == 46
    assert sum(j['kind'] == 'required' for j in jobs) == 9
    assert len({j['case'] for j in jobs}) == 19
    assert all(run.test_verdict(*run.archived(j)[::2])['passed'] for j in jobs if j['kind'] != 'image')


def test_discriminator_dispatch_matches_archived_architecture():
    by_label = {(j['case'], j['label']): j for j in run.plan()}
    skip = run.archived(by_label[('vector_unequal_mass', 'linear_skip_d96_beta5')])[0]
    smooth = run.archived(by_label[('vector_unequal_width', 'softplus10_d64_l2_f2')])[0]
    plain = run.archived(by_label[('vector_overlap', 'd128_l3_f4')])[0]
    assert isinstance(run.discriminator(skip)(2, 96, 2, 2), LinearSkipCritic)
    assert isinstance(run.discriminator(smooth)(2, 64, 2, 2), SmoothFourierCritic)
    assert run.discriminator(plain) is None


def test_untimed_drops_only_timing():
    value = dict(seconds=1., live=dict(hq=.9), observations=[dict(step=1, seconds=2., hq=.5)])
    assert run.untimed(value) == dict(live=dict(hq=.9), observations=[dict(step=1, hq=.5)])


def test_portability_evidence_matches_archived_artifacts():
    from reports.transfer_suite.host_replication.portability import build
    import json
    paths = [json.loads(line) for line in (build.ROOT / 'paths.jsonl').read_text().splitlines()]
    assert len(paths) == 16 and not any(p['equals_archive'] for p in paths)
    native = build.read(build.ROOT / 'cross_cpu/native_mkl_cnr_avx2' / build.EPISODE)['result']
    emulated = build.read(build.ROOT / 'cross_cpu/emulated_haswell_mkl_cnr_avx2' / build.EPISODE)['result']
    assert run.untimed(native) == run.untimed(emulated)
    trace = json.loads((build.ROOT / 'cross_cpu/trace300_cnr_avx2.json').read_text())
    assert trace['all_300_updates_bit_identical'] and len(trace['per_update_parameter_sha256_prefix']) == 300
    score = build.replication_score(build.ROOT / 'cnr_avx2_replication')
    assert (score['required'], score['practical'], score['unsupported']) == (8, 9, ['mode_hold', 'vector_unequal_mass'])
