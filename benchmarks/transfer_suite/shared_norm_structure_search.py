"""Run declared normalization-structure D variants with unchanged shared cap6.

python -u -m benchmarks.transfer_suite.shared_norm_structure_search --plan PLAN --output OUTPUT

The critics are declared in ``shared_norm_structure_research``; everything else is the
canonical ``shared_discriminator_search`` runner.
"""
from . import shared_discriminator_search as runner
from . import shared_norm_structure_research as research

OPTIONS = dict(title='Shared cap6 discriminator normalization structure research', version='shared-norm-structure-v1', stop_on_first_pass=False)


def episode(job, card):
    with runner.family(research):
        return runner.episode(job, card)


def run(declaration, output):
    with runner.family(research):
        return runner.run(declaration, output, **OPTIONS)


if __name__ == '__main__':
    runner.main(research, **OPTIONS)
