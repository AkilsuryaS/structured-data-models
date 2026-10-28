# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path

import pytest

stage = runpy.run_path(str(Path(__file__).parents[1] / "docs/publish.py"))[
    "stage"
]


def test_publish_preserves_releases_and_replaces_latest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "html"
    source.mkdir()
    (source / "index.html").write_text("release")
    site = tmp_path / "site"
    stage(source, site, "v0.2.0")
    stage(source, site, "v0.10.0")
    (site / "obsolete.html").write_text("obsolete")
    (site / "CNAME").write_text("docs.example.com")
    (source / "index.html").write_text("development")
    stage(source, site, "latest")

    assert (site / "index.html").read_text() == "development"
    assert not (site / "obsolete.html").exists()
    assert (site / "CNAME").read_text() == "docs.example.com"
    assert (site / "versions/v0.2.0/index.html").read_text() == "release"
    assert (site / "versions/v0.10.0/index.html").read_text() == "release"
    assert json.loads((site / "versions.json").read_text()) == [
        {"version": "latest", "path": "./"},
        {"version": "v0.10.0", "path": "versions/v0.10.0/"},
        {"version": "v0.2.0", "path": "versions/v0.2.0/"},
    ]
    assert (site / ".nojekyll").exists()


def test_republish_replaces_only_requested_release(tmp_path: Path) -> None:
    source = tmp_path / "html"
    source.mkdir()
    (source / "index.html").write_text("original")
    (source / "removed.html").write_text("removed")
    site = tmp_path / "site"
    stage(source, site, "latest")
    stage(source, site, "v0.2.0")
    (source / "removed.html").unlink()
    (source / "index.html").write_text("rebuilt")
    stage(source, site, "v0.2.0")

    assert (site / "index.html").read_text() == "original"
    assert (site / "versions/v0.2.0/index.html").read_text() == "rebuilt"
    assert not (site / "versions/v0.2.0/removed.html").exists()


@pytest.mark.parametrize("version", ["../escape", "v0.2.0rc1", "main"])
def test_invalid_version_does_not_modify_site(
    tmp_path: Path, version: str
) -> None:
    with pytest.raises(ValueError, match=r"stable vX\.Y\.Z"):
        stage(tmp_path / "html", tmp_path / "site", version)
    assert not (tmp_path / "site").exists()


def test_missing_build_does_not_modify_site(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        stage(tmp_path / "html", tmp_path / "site", "latest")
    assert not (tmp_path / "site").exists()
