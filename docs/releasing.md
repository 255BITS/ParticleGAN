# CI and PyPI releases

ParticleGAN is published as `particlegan`, separately from HyperGAN. The package
version is declared in `pyproject.toml`; release 0.9.0 changes the public recipe
default. Explicit presets and saved recipe restoration retain their documented
contracts. A software release does not requalify archived scientific results.

## CI

[Tests](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml) runs
on pull requests and pushes to `develop` and `master`, manual dispatches and
calls from the release workflow.

- Python 3.12 runs the regression suite with CPU PyTorch, with the bounded
  late-phase renderer in a separate process.
- Python 3.11 runs the Lunar smoke pipeline and GIF exports.
- Python 3.10, 3.11 and 3.12 each build and smoke-test a wheel outside the
  checkout, including public recipe restoration and checkpoint examples.
- After every matrix job passes, CI builds a source archive and wheel, checks
  their contents, metadata and README with Twine, and saves the distributions.

CUDA optimizer contracts and real-data tests require local GPU verification;
CPU CI does not establish their numerical or training-quality results. Record
focused API and end-to-end HyperGAN smoke/resume checks for the changed paths
before publishing. Preserve archived Forge results instead of rerunning
unchanged scientific studies merely to merge or release.

## Trusted Publishing

The existing publisher uses GitHub identity credentials rather than a stored
PyPI password or token. Its configuration is:

| Field | Value |
| --- | --- |
| PyPI project | `particlegan` |
| GitHub owner/repository | `255BITS/ParticleGAN` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

The repository's `pypi` environment permits release tags matching `v*`. Its
name must match the PyPI publisher. Repository Actions must be enabled, the
publishing job must retain `id-token: write`, and the GitHub actor must be able
to create a release. If environment rules require review, resolve that actual
Actions gate before considering publication complete. The
[PyPI publisher settings](https://pypi.org/manage/account/publishing/) are
managed through the existing PyPI account associated with 255labs.xyz.

## Publishing

1. Merge the reviewed changes into `develop`, including the version, README,
   dated `CHANGELOG.md` entry and release notes. Finish the exact HyperGAN
   integration smoke locally before committing its dependency to a release.
2. Wait for CI on the final `develop` commit and record its immutable SHA as
   the release source. A reviewed `develop` to `master` release PR is optional;
   the release workflow validates either tagged source without relying on the
   target branch label. HyperGAN integration stays on HyperGAN's `develop`.
3. Create and publish the GitHub release with a tag matching the package
   version. For example, with `RELEASE_SHA` set to that verified commit:

   ```sh
   gh release create v0.9.0 --repo 255BITS/ParticleGAN \
     --target "$RELEASE_SHA" --title "ParticleGAN 0.9.0" \
     --notes-file /tmp/particlegan-0.9.0-release-notes.md
   ```

   Publishing the release triggers `release.yml`; a draft or tag push alone
   does not publish to PyPI. The version check resolves the tag to a commit,
   and all test/build jobs check out that same commit. Release checks run
   regardless of whether GitHub describes the target as a branch or a SHA.
4. Monitor the release Actions run through its test, distribution and publish
   jobs. The publisher downloads the tested distributions and does not rebuild
   them. Confirm that **all** jobs ran and passed; skipped tests are not a
   successful release gate.
5. Verify the version and distribution hashes on
   [PyPI](https://pypi.org/project/particlegan/), then install the release in an
   isolated environment outside the source checkout:

   ```sh
   python -m venv /tmp/particlegan-release-smoke
   /tmp/particlegan-release-smoke/bin/python -m pip install particlegan==0.9.0
   cd /tmp
   /tmp/particlegan-release-smoke/bin/python -c 'from particlegan import DEFAULT_RECIPE, get_recipe, Recipe; r = get_recipe(); assert DEFAULT_RECIPE == "bispf_optimized"; assert Recipe(**r.to_dict()) == r'
   ```

   Also run the prepared HyperGAN training/resume smoke using the released
   package, verifying its import path and distribution version rather than an
   editable checkout. Only then launch the long run with a frozen config.

Local package checks can use `python -m build --outdir /tmp/particlegan-dist`
and `python -m twine check --strict /tmp/particlegan-dist/*`. Keep raw logs,
checkpoints and other bulk artifacts outside Git. Actions retains the tested
package distributions and job logs for the release receipt.
