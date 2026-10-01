"""Small integration checks for the installed-public-default verification route."""
import pytest

from benchmarks.transfer_suite import public_default_verification as verification


def _declaration(name):
    jobs, profile = verification.load_declaration()
    base = verification.public_default(profile)
    job = next(job for job in jobs if job['spec']['name'] == name)
    spec, card, variant = verification.declared_spec(job, profile, base)
    return jobs, base, spec, card, variant


def test_declared_routes_and_installed_root_guard():
    jobs, base, rare_spec, card, variant = _declaration('vector_unequal_mass')
    assert len(jobs) == 19
    assert {job['spec']['runner'] for job in jobs} == {'vector', 'image', 'legacy'}
    assert base.name == 'gan_v3'
    assert variant['name'] == 'batchfeat_center6_distance_head'
    assert rare_spec['research_discriminator'] == card
    with pytest.raises(ValueError, match='checkout source'):
        verification.public_module_manifest(verification.ROOT)


def test_vector_public_trainer_uses_promoted_discriminator_and_recipe():
    _, base, spec, card, _ = _declaration('vector_unequal_mass')
    result, context = verification.run_vector(spec, card, base, max_steps=1)
    assert type(context['trainer'].D).__name__ == 'BatchDistanceDiscriminator'
    assert context['shapes']['real_batch'] == [128, 2]
    assert context['shapes']['prior'] == [256, 4]
    assert context['shapes']['discriminator_parameters'] == 19013
    assert [row['lr'] for row in context['applied']] == [.00425, .0085, .00425]
    assert all(row['betas'] == [0., .99] for row in context['applied'])
    assert result['update_counts'] == {'g': 1, 'd': 1}
    assert result['actions'] == [{'step': 1, 'multiplier': 1.,
                                  'lr_g': .00425, 'lr_prior': .0085, 'lr_d': .00425}]


def test_image_public_trainer_preserves_frozen_shapes_and_shared_stream():
    _, base, spec, card, _ = _declaration('img_stripes2')
    assert card is None
    result, context = verification.run_image(spec, base, max_steps=1)
    assert context['shapes']['real_batch'] == [32, 1, 8, 8]
    assert context['shapes']['prior'] == [32, 8]
    assert context['shapes']['generator_output'] == [2, 1, 8, 8]
    assert context['trainer'].latent_generator is context['trainer'].penalty_generator
    assert [row['lr'] for row in context['applied']] == [.00425, .0085, .00425]
    assert result['update_counts'] == {'g': 1, 'd': 1}
    assert len(result['actions']) == 1
