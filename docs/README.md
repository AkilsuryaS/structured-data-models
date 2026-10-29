# Documentation

Install the package and documentation dependencies, then build HTML and run doctests with warnings treated as errors:

```bash
uv pip install --torch-backend=cpu --group=doc .
uv run --no-project sphinx-build -M html docs/source docs/build -W --keep-going
uv run --no-project sphinx-build -M doctest docs/source docs/build -W --keep-going
```

## Read the Docs

[`.readthedocs.yaml`](../.readthedocs.yaml) builds the existing Sphinx site on Python 3.11 with CPU-only PyTorch and the `doc` dependency group. It fetches Git history and tags so `setuptools-scm` can identify the installed package version. Read the Docs handles hosting and its built-in version menu; no custom archive, version index, or selector is required. GitHub Actions continues to validate HTML and doctests without publishing to GitHub Pages.

The Sphinx configuration uses `READTHEDOCS_CANONICAL_URL` for canonical page URLs and the theme's copied-source URLs, including release documentation and pull request previews. Local builds work without this environment variable.

## Maintainer setup

1. Import this repository into the organization's Read the Docs account and enable its GitHub integration, using `.readthedocs.yaml` as the configuration file.
2. Build `latest` from `main` and verify the generated API pages and version menu. Enable pull request builds if preview documentation is desired.
3. Activate supported release versions. Read the Docs provides `stable` for the greatest stable semantic version and `latest` for the default branch; configure the default documentation version and version-activation rules in the dashboard.
4. Plan older-release builds separately: immutable tags predating `.readthedocs.yaml` do not inherit configuration from `main`. Backport documentation configuration to maintained release branches where appropriate, and validate each historical version before advertising it. Do not move existing release tags.
5. Once the hosted site and its final URL are verified, update public documentation links and arrange redirects from the existing GitHub Pages URLs. This change stops Pages publication but leaves the previously published site intact during the migration.

Account setup, version activation, preview builds, and redirects require maintainer access. A local Sphinx build verifies the documentation but cannot verify these hosted settings.

See the Read the Docs guides for [version management](https://docs.readthedocs.com/platform/stable/versions.html), [build customization](https://docs.readthedocs.com/platform/stable/build-customization.html), and the [version menu](https://docs.readthedocs.com/platform/stable/flyout-menu.html).
