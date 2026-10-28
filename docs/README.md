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
- Pull requests build HTML and run doctests without publishing. The workflow can also be dispatched on `main` to rebuild the development documentation.

Release publishing starts with tags that contain the versioning workflow and assets. Earlier releases are not automatically backfilled. Each release build installs the package and documentation dependencies from its own checkout; it does not document the current `main` API under an older version label.

For a local release preview, set `SDM_DOC_VERSION=vX.Y.Z` while building the matching release checkout. Serve the staged site over HTTP to exercise the selector:

```bash
python docs/publish.py docs/build/html docs/build/site vX.Y.Z
python -m http.server --directory docs/build/site
```

Without a published index, the selector displays the current version and remains disabled.
