"""Run declared minibatch-feature D variants with unchanged shared cap6.

python -u -m benchmarks.transfer_suite.shared_batch_feature_search --plan PLAN --output OUTPUT

The critics are declared in ``shared_batch_feature_research``; everything else is the
canonical ``shared_discriminator_search`` runner.
"""
from . import shared_discriminator_search as runner
from . import shared_batch_feature_research as research

OPTIONS = dict(title='Shared cap6 minibatch-feature discriminator research', version='shared-batch-feature-v1', stop_on_first_pass=True)


def episode(job, card):
    with runner.family(research):
        return runner.episode(job, card)


def run(declaration, output):
    with runner.family(research):
        return runner.run(declaration, output, **OPTIONS)


if __name__ == '__main__':
    runner.main(research, **OPTIONS)
