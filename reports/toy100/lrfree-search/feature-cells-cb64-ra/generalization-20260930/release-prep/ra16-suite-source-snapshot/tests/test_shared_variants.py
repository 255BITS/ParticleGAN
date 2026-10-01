"""Architecture support must preserve the shared recipe and every failed trial."""
from copy import deepcopy

import pytest

from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.shared_variants import architecture_identity, architecture_spec, select_architecture


@pytest.fixture
def spec():
    return next(j['spec'] for j in plan() if j['spec']['name'] == 'vector_unequal_mass')


@pytest.mark.parametrize('field,value', [('lr', .1), ('steps', 2400), ('hidden', 128),
                                        ('particles', 1024), ('thresholds', [])])
def test_discriminator_card_cannot_change_training_generator_or_measurement(spec, field, value):
    with pytest.raises(ValueError):
        architecture_spec(spec, dict(name='invalid', overrides={field: value}))


def test_explicit_plain_discriminator_removes_old_research_card_only(spec):
    before = deepcopy(spec)
    changed = architecture_spec(spec, dict(name='plain', overrides=dict(
        d_hidden=128, d_layers=3, fourier=0, research_discriminator=None)))
    assert spec == before
    assert 'research_discriminator' not in changed
    assert {k: v for k, v in changed.items() if k not in {'d_hidden', 'd_layers', 'fourier'}} == {
        k: v for k, v in spec.items() if k not in {'d_hidden', 'd_layers', 'fourier', 'research_discriminator'}}


def test_support_selects_live_pass_and_keeps_failed_reference():
    failed = dict(verdict=dict(passed=False, shortfall=0., confirmation_fraction=2.),
                  architecture='reference', artifact='a.gz', discriminator_variant=None)
    passed = dict(verdict=dict(passed=True, shortfall=0., confirmation_fraction=.9),
                  architecture='wide', artifact='b.gz', discriminator_variant=dict(name='wide', overrides=dict(d_hidden=128)))
    selected = select_architecture([failed, passed])
    assert selected['verdict']['passed']
    assert not selected['reference_passed']
    assert [t['artifact'] for t in selected['trials']] == ['a.gz', 'b.gz']
    assert [t['status'] for t in selected['trials']] == ['FAIL', 'PASS']
    assert 'trials' not in passed


def test_renamed_noop_discriminator_has_same_identity(spec):
    renamed = deepcopy(spec['research_discriminator'])
    renamed['name'] = 'another_label'
    changed = architecture_spec(spec, dict(name='repeat', overrides=dict(
        d_hidden=spec.get('d_hidden', spec['hidden']), research_discriminator=renamed)))
    assert architecture_identity(changed) == architecture_identity(spec)


def test_adapter_cannot_be_hidden_under_plain_adam():
    from reports.transfer_suite.unadjusted.build import validate_update_rule
    with pytest.raises(AssertionError, match='Undeclared'):
        validate_update_rule(dict(adapter={}), {})


def test_adapter_receipt_must_match_declared_equation():
    from benchmarks.transfer_suite.relative_step_adapter import mechanism
    from reports.transfer_suite.unadjusted.build import validate_update_rule
    rule = mechanism(.1)
    trace = dict(parameter_rms=1., proposal_rms=1., factor=.5)
    payload = dict(mechanism=rule, adapter=dict(mechanism=rule, trace=[trace]), result={})
    with pytest.raises(AssertionError, match='equation'):
        validate_update_rule(payload, dict(mechanism=rule))
    trace['factor'] = .1/(1.+1e-12)
    assert validate_update_rule(payload, dict(mechanism=rule)) == rule
