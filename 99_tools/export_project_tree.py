from __future__ import annotations

from pathlib import Path

EXCLUDE_DIRS = {
    ".git", ".idea", ".venv", "venv",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", "dist", "build",
}
EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_FILES = {"poetry.lock"}  # optioneel

def should_skip(path: Path) -> bool:
    name = path.name

    # Skip excluded directories anywhere in the tree
    if path.is_dir() and name in EXCLUDE_DIRS:
        return True

    # Skip some files
    if path.is_file():
        if name in EXCLUDE_FILES:
            return True
        if path.suffix in EXCLUDE_FILE_SUFFIXES:
            return True

    return False


def write_tree(root: Path, out_file: Path) -> None:
    root = root.resolve()
    lines: list[str] = [str(root)]

    def walk(dir_path: Path, prefix: str = "") -> None:
        entries = []
        for p in dir_path.iterdir():
            if should_skip(p):
                continue
            entries.append(p)
        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))  # dirs first

        for i, p in enumerate(entries):
            is_last = i == (len(entries) - 1)
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + p.name)

            if p.is_dir():
                extension = "    " if is_last else "│   "
                walk(p, prefix + extension)

    walk(root)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]  # assumes 99_tools/ is one level under root
    output_path = project_root / "project_tree.txt"
    write_tree(project_root, output_path)
    print(f"Wrote: {output_path}")
