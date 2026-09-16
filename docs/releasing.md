# CI and PyPI releases

ParticleGAN is published as `particlegan`, a separate project from legacy
HyperGAN. Use the existing PyPI account associated with 255labs.xyz.

## CI

[Tests](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml) runs
on pull requests, pushes to `master` and `api`, and manual dispatches:

- Python 3.10, 3.11, and 3.12 run the regression suite with CPU PyTorch.
- Every Python version installs and smoke-tests a wheel outside the checkout.
- After tests pass, CI builds a source archive and wheel, checks their metadata
  and README with Twine, and saves the distributions as a downloadable artifact.

Real-data CUDA tests stay opt-in because GitHub's standard runners have no GPU.
For local coverage, run `RUN_CUDA_IMAGE_TESTS=1 python -m pytest -q` on a CUDA
machine with the image dependencies and CIFAR data installed.

## One-time PyPI setup

While logged into the existing account, open
[PyPI Publishing](https://pypi.org/manage/account/publishing/) and add a new
**pending publisher** using the GitHub form:

| Field | Value |
| --- | --- |
| PyPI project name | `particlegan` |
| Owner | `255BITS` |
| Repository name | `ParticleGAN` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Enter the workflow filename, not its full `.github/workflows/` path. The GitHub
repository uses an environment named `pypi`, restricted to tags matching `v*`;
keep that environment name identical in both places. No PyPI password or API
token needs to be stored in GitHub.

The pending publisher creates the PyPI project on the first successful upload;
it does not reserve the name beforehand. See
[PyPI's setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Publishing a version

1. Merge the reviewed API and release workflows into `master` and wait for CI.
2. Set the version in `pyproject.toml` and make the README's installation section
   current for the release. The prepared first version is `0.2.0`.
3. Create a GitHub release from that commit with a matching tag, for example
   `v0.2.0`. Publishing the GitHub release triggers `release.yml`; creating a
   draft or pushing a tag alone does not publish to PyPI.
4. The workflow verifies the tag, reruns the same test matrix, builds and checks
   the distributions, then publishes those artifacts. A failed check prevents
   upload. The publish job uses the `pypi` environment and short-lived GitHub
   identity credentials through
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
5. Verify the release in a fresh environment with
   `python -m pip install particlegan==0.2.0` (substitute the released version).

Downloadable distributions and job logs are attached to the Actions run. The
publish job only downloads the checked artifacts; it does not rebuild them.
PyPI publishing has not been exercised until the first real release succeeds.
