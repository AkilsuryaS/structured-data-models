# Documentation

Install the package and documentation dependencies, then build HTML with warnings treated as errors:

```bash
uv pip install --torch-backend=cpu --group=doc .
uv run --no-project sphinx-build -M html docs/source docs/build -W --keep-going
```

## Published versions

- The site root contains **latest (main)**. Existing documentation links continue to work.
- Publishing a non-prerelease GitHub release builds the documentation from that release's tag and stores it at `versions/vX.Y.Z/`. Release tags must use the `vX.Y.Z` format.
- Every publication preserves other release builds and updates `versions.json`. The version selector reads this shared index, so archived pages can discover newly published releases without being rebuilt. Switching versions opens the selected version's home page because an individual page may not exist in another release.
- Pull requests build HTML and run doctests without publishing. Dispatch the workflow on `main` with an empty `release_tag` to rebuild the development documentation, or supply an existing stable tag such as `v0.2.1` to publish or rebuild that release.

To backfill releases that predate this workflow, run **Actions → Documentation → Run workflow** on `main` and enter the release tag. Repeat for each release to archive. Publishing a new stable GitHub release archives it automatically once its tag contains this workflow.

Each build installs the package and documentation dependencies from the requested source checkout and reads that checkout's documentation pages. The Sphinx configuration, theme templates, and version selector come from the workflow checkout, so older tags can use the current publishing tools without changing their model code or documenting the current `main` API under an older version label. Builds and doctests must pass before anything is published. Only the publishing job has write permission.

For a local release preview, install the matching release checkout into a separate environment. Build its documentation with `SDM_DOC_VERSION=vX.Y.Z` and `-c /path/to/current-checkout/docs/source` to use the versioning configuration. Serve the staged site over HTTP to exercise the selector:

```bash
python docs/publish.py docs/build/html docs/build/site vX.Y.Z
python -m http.server --directory docs/build/site
```

Without a published index, the selector displays the current version and remains disabled.
