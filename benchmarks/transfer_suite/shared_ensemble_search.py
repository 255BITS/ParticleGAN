"""Run declared smooth-ensemble D variants with one unchanged shared cap6 recipe.

python -u -m benchmarks.transfer_suite.shared_ensemble_search --plan PLAN --output OUTPUT

The critics are declared in ``shared_ensemble_research``; everything else is the
canonical ``shared_discriminator_search`` runner.
"""
from . import shared_discriminator_search as runner
from . import shared_ensemble_research as research

OPTIONS = dict(title='Shared cap6 smooth ensemble discriminator research', version='shared-ensemble-v1', stop_on_first_pass=False)


def episode(job, card):
    with runner.family(research):
        return runner.episode(job, card)


def run(declaration, output):
    with runner.family(research):
        return runner.run(declaration, output, **OPTIONS)


if __name__ == '__main__':
    runner.main(research, **OPTIONS)
