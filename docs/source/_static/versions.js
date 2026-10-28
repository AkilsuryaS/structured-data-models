// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(async () => {
  const select = document.getElementById("sdm-doc-version");
  if (!select) return;

  const currentRoot = new URL(select.dataset.root, window.location.href);
  const siteRoot = new URL(
    select.dataset.version === "latest" ? "./" : "../../",
    currentRoot,
  );

  try {
    const response = await fetch(new URL("versions.json", siteRoot));
    if (!response.ok) return;
    const versions = await response.json();
    select.replaceChildren(...versions.map(({ version, path }) => {
      const option = new Option(
        version === "latest" ? "latest (main)" : version,
        new URL(path, siteRoot).href,
      );
      option.selected = version === select.dataset.version;
      return option;
    }));
    select.disabled = false;
    // Version home pages are always present; individual APIs may not be.
    select.addEventListener("change", () => window.location.assign(select.value));
  } catch {
    // Local builds and offline readers still see the current version.
  }
})();
