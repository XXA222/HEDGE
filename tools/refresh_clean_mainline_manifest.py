#!/usr/bin/env python3
"""Refresh the clean-mainline SHA-256 manifest after an intentional source update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_clean_mainline import manifest_files, rel, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest_path = root / "CLEAN-MAINLINE-MANIFEST.json"
    version_path = root / "CLEAN-MAINLINE-VERSION.txt"
    if not manifest_path.is_file() or not version_path.is_file():
        raise RuntimeError("clean-mainline manifest/version authority is missing")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    total_bytes = 0
    for path in manifest_files(root, workspace_mode=True):
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": rel(root, path),
                "size": size,
                "sha256": sha256_file(path),
            }
        )
    payload["version"] = version_path.read_text(encoding="utf-8").strip()
    payload["file_count"] = len(rows)
    payload["total_bytes"] = total_bytes
    payload["files"] = rows

    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(f"Manifest refreshed: files={len(rows)} bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
