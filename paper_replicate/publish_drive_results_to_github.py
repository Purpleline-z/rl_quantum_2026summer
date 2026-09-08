"""Copy small, auditable experiment outputs from Drive into paper_replicate/results."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ALLOWED_SUFFIXES = {".csv", ".json", ".md", ".png", ".pdf", ".txt", ".yaml", ".yml"}
MAXIMUM_FILE_BYTES = 15 * 1024 * 1024
SKIPPED_NAME_TERMS = ("checkpoint", "weights", "weight", "resumable", "optimizer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(drive_results_root: Path, repository_root: Path, run_name: str) -> dict:
    if not run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run-name may contain letters, numbers, underscores, and hyphens only.")
    source = drive_results_root.resolve()
    destination = (repository_root / "paper_replicate" / "results" / run_name).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    copied, skipped = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        lower_name = path.name.lower()
        if path.suffix.lower() not in ALLOWED_SUFFIXES or path.stat().st_size > MAXIMUM_FILE_BYTES or any(term in lower_name for term in SKIPPED_NAME_TERMS):
            skipped.append({"path": str(relative), "reason": "unsupported_type_or_large_or_checkpoint"})
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append({"path": str(relative), "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest = {"source_drive_results_root": str(source), "published_run_name": run_name,
                "policy": {"allowed_suffixes": sorted(ALLOWED_SUFFIXES), "maximum_file_bytes": MAXIMUM_FILE_BYTES,
                           "excluded_name_terms": SKIPPED_NAME_TERMS}, "copied_files": copied, "skipped_files": skipped}
    (destination / "publication_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-results-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    manifest = publish(args.drive_results_root, args.repository_root, args.run_name)
    print(f"Published {len(manifest['copied_files'])} files; skipped {len(manifest['skipped_files'])} files.")


if __name__ == "__main__":
    main()
