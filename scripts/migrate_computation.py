"""Apply this boilerplate release to a NeuroFlame computation repository."""

import argparse
import filecmp
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

if __package__:
    from .neuroflame_manifest import (
        MANIFEST_FILE,
        load_manifest,
        merge_compatibility,
        write_manifest,
    )
else:
    from neuroflame_manifest import (
        MANIFEST_FILE,
        load_manifest,
        merge_compatibility,
        write_manifest,
    )

MANAGED_DIRECTORIES = (
    Path("app/code/framework"),
    Path("app/code/runtime"),
    Path("app/config"),
    Path("system/provision/code"),
)
MANAGED_FILES = (
    Path("Dockerfile-dev"),
    Path("Dockerfile-prod"),
    Path("dockerPush.sh"),
    Path("debugger.py"),
    Path("makeJob.py"),
    Path("pyproject.toml"),
    Path("requirements-dev.txt"),
    Path("scripts/neuroflame_manifest.py"),
    Path("scripts/publish_computation_image.py"),
    Path("system/entry_central.py"),
    Path("system/entry_edge.py"),
    Path("system/entry_provision.py"),
)
GENERATED_NAMES = {
    ".git",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "job",
    "simulator_workspace",
    "test_output",
}
EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")


def read_version(repository: Path) -> str:
    """Read a repository's boilerplate compatibility version."""
    if not (repository / MANIFEST_FILE).is_file():
        return "unversioned"
    return load_manifest(repository)["compatibility"]["boilerplateVersion"]


def _ignore_generated(_directory: str, names: list[str]) -> set[str]:
    """Exclude generated content when creating a migrated repository copy."""
    ignored = {name for name in names if name in GENERATED_NAMES}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def _files_equal(source: Path, target: Path) -> bool:
    """Return whether two files have identical bytes."""
    return target.is_file() and filecmp.cmp(source, target, shallow=False)


def _directories_equal(source: Path, target: Path) -> bool:
    """Return whether directory trees contain identical non-cache files."""
    if not target.is_dir():
        return False
    comparison = filecmp.dircmp(
        source,
        target,
        ignore=["__pycache__", ".ruff_cache"],
    )
    if (
        comparison.left_only
        or comparison.right_only
        or comparison.common_funny
        or comparison.funny_files
    ):
        return False
    if any(
        not _files_equal(source / name, target / name)
        for name in comparison.common_files
    ):
        return False
    return all(
        _directories_equal(source / name, target / name)
        for name in comparison.common_dirs
    )


def merge_requirements(source_text: str, target_text: str) -> str:
    """Update framework pins while retaining target-only computation packages."""
    source_pins = {}
    source_order = []
    for line in source_text.splitlines():
        stripped = line.strip()
        match = EXACT_REQUIREMENT.fullmatch(stripped)
        if not match:
            if stripped and not stripped.startswith("#"):
                raise ValueError(f"Unsupported boilerplate requirement: {line}")
            continue
        normalized_name = match.group(1).lower().replace("_", "-")
        source_pins[normalized_name] = stripped
        source_order.append(normalized_name)

    result = []
    updated = set()
    for line in target_text.splitlines():
        match = EXACT_REQUIREMENT.fullmatch(line.strip())
        if not match:
            result.append(line)
            continue
        normalized_name = match.group(1).lower().replace("_", "-")
        if normalized_name in source_pins:
            result.append(source_pins[normalized_name])
            updated.add(normalized_name)
        else:
            result.append(line)

    for normalized_name in source_order:
        if normalized_name not in updated:
            result.append(source_pins[normalized_name])
    return "\n".join(result).rstrip() + "\n"


def merge_dockerignore(source_text: str, target_text: str) -> str:
    """Add boilerplate-required ignores while retaining target-only entries."""
    result = target_text.splitlines()
    existing = {line.strip() for line in result if line.strip()}
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped and stripped not in existing:
            result.append(stripped)
            existing.add(stripped)
    return "\n".join(result).rstrip() + "\n"


def planned_changes(source: Path, target: Path) -> list[str]:
    """List framework-owned paths that differ from this boilerplate release."""
    changes = []
    for relative_path in MANAGED_DIRECTORIES:
        if not _directories_equal(source / relative_path, target / relative_path):
            changes.append(str(relative_path))
    for relative_path in MANAGED_FILES:
        if not _files_equal(source / relative_path, target / relative_path):
            changes.append(str(relative_path))

    source_requirements = (source / "requirements.txt").read_text(encoding="utf-8")
    target_requirements_path = target / "requirements.txt"
    target_requirements = target_requirements_path.read_text(encoding="utf-8")
    if (
        merge_requirements(source_requirements, target_requirements)
        != target_requirements
    ):
        changes.append("requirements.txt")
    source_dockerignore = (source / ".dockerignore").read_text(encoding="utf-8")
    target_dockerignore_path = target / ".dockerignore"
    target_dockerignore = (
        target_dockerignore_path.read_text(encoding="utf-8")
        if target_dockerignore_path.is_file()
        else ""
    )
    if (
        merge_dockerignore(source_dockerignore, target_dockerignore)
        != target_dockerignore
    ):
        changes.append(".dockerignore")
    source_manifest = load_manifest(source)
    try:
        target_manifest = load_manifest(target)
    except ValueError:
        changes.append(MANIFEST_FILE)
    else:
        if merge_compatibility(source_manifest, target_manifest) != target_manifest:
            changes.append(MANIFEST_FILE)
    return changes


def _replace_directory(source: Path, target: Path) -> None:
    """Replace one managed tree and restore the old tree if replacement fails."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.boilerplate-", dir=target.parent
    ) as staging_root:
        staging_path = Path(staging_root) / "new"
        backup_path = Path(staging_root) / "old"
        shutil.copytree(source, staging_path, ignore=_ignore_generated)
        if target.exists():
            target.rename(backup_path)
        try:
            staging_path.rename(target)
        except BaseException:
            if backup_path.exists() and not target.exists():
                backup_path.rename(target)
            raise


def _replace_file(source: Path, target: Path) -> None:
    """Atomically replace one framework-owned file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)


def apply_boilerplate(source: Path, target: Path) -> None:
    """Replace framework-owned content and record the applied release."""
    merged_manifest = merge_compatibility(
        load_manifest(source),
        load_manifest(target),
    )
    source_requirements = (source / "requirements.txt").read_text(encoding="utf-8")
    target_requirements_path = target / "requirements.txt"
    merged_requirements = merge_requirements(
        source_requirements,
        target_requirements_path.read_text(encoding="utf-8"),
    )
    source_dockerignore = (source / ".dockerignore").read_text(encoding="utf-8")
    target_dockerignore_path = target / ".dockerignore"
    target_dockerignore = (
        target_dockerignore_path.read_text(encoding="utf-8")
        if target_dockerignore_path.is_file()
        else ""
    )
    merged_dockerignore = merge_dockerignore(source_dockerignore, target_dockerignore)

    for relative_path in MANAGED_DIRECTORIES:
        _replace_directory(source / relative_path, target / relative_path)
    for relative_path in MANAGED_FILES:
        _replace_file(source / relative_path, target / relative_path)

    target_requirements_path.write_text(merged_requirements, encoding="utf-8")
    target_dockerignore_path.write_text(merged_dockerignore, encoding="utf-8")
    write_manifest(target, merged_manifest)


def create_migrated_copy(source: Path, target: Path, output: Path) -> None:
    """Copy a computation repository and upgrade the copy."""
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    load_manifest(target)
    shutil.copytree(target, output, ignore=_ignore_generated)
    apply_boilerplate(source, output)


def build_parser() -> argparse.ArgumentParser:
    """Build the migration command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply the checked-out NeuroFlame boilerplate while preserving "
            "app/code/computation and computation-only dependencies."
        )
    )
    parser.add_argument("target", type=Path, help="computation repository to update")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="report whether the target matches this boilerplate",
    )
    mode.add_argument(
        "--output",
        type=Path,
        help="create an upgraded repository copy at this path",
    )
    mode.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite framework-owned files in the target repository",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="confirm an in-place overwrite",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a check, copy migration, or explicit in-place migration."""
    args = build_parser().parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    target = args.target.resolve()
    if not target.is_dir():
        raise NotADirectoryError(f"Target repository does not exist: {target}")
    if not (target / "app/code/computation").is_dir():
        raise ValueError(f"Not a NeuroFlame computation repository: {target}")

    changes = planned_changes(source, target)
    print(
        f"Boilerplate {read_version(target)} -> {read_version(source)}; "
        f"{len(changes)} managed path(s) differ."
    )
    for change in changes:
        print(f"  {change}")

    if args.check:
        if args.force:
            raise ValueError("--force is only valid with --in-place")
        return int(bool(changes))
    if args.output:
        if args.force:
            raise ValueError("--force is only valid with --in-place")
        output = args.output.resolve()
        if output == target or target in output.parents:
            raise ValueError("--output must be outside the target repository")
        create_migrated_copy(source, target, output)
        print(f"Migrated copy created at {output}")
        return 0
    if not args.force:
        raise ValueError("--in-place requires --force")
    if target == source:
        raise ValueError("Cannot migrate the boilerplate repository in place")
    apply_boilerplate(source, target)
    print(f"Updated {target} in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
