# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage documentation in an existing gh-pages checkout before publishing."""

import argparse
import json
import re
import shutil
from pathlib import Path

RELEASE_TAG = re.compile(r"v\d+\.\d+\.\d+")


def stage(source: Path, site: Path, version: str) -> None:
    """Replace one build while preserving all other published versions."""
    if version != "latest" and not RELEASE_TAG.fullmatch(version):
        raise ValueError(
            f"Expected latest or a stable vX.Y.Z tag, got {version}"
        )
    if not (source / "index.html").is_file():
        raise FileNotFoundError(source / "index.html")

    site.mkdir(parents=True, exist_ok=True)
    if version == "latest":
        destination = site
        for path in site.iterdir():
            if path.name not in {".git", "versions", "CNAME", ".nojekyll"}:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    else:
        destination = site / "versions" / version
        if destination.exists():
            shutil.rmtree(destination)

    shutil.copytree(source, destination, dirs_exist_ok=True)
    releases = sorted(
        (
            path.name
            for path in (site / "versions").glob("v*")
            if RELEASE_TAG.fullmatch(path.name)
            and (path / "index.html").is_file()
        ),
        key=lambda tag: tuple(int(part) for part in tag[1:].split(".")),
        reverse=True,
    )
    versions = [{"version": "latest", "path": "./"}]
    versions.extend(
        {"version": tag, "path": f"versions/{tag}/"} for tag in releases
    )
    (site / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (site / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("site", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()
    stage(args.source, args.site, args.version)
