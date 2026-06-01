"""Reorganize previously-generated comparison figures to match the current
output convention used by ``plot_various_from_experiments.py``:

  figures/comparisons/by_variable/<variable_slug>/<scenario>_<variable_slug>.png

Handles two legacy layouts:
1. Flat files in ``figures/comparisons/`` named ``<scenario>__<slug>.png``
   (double underscore separator).
2. Files in ``figures/comparisons/by_variable/<slug>/`` named
   ``<scenario>.png`` (no slug suffix) or ``<scenario>__<slug>.png``
   (double underscore).

All such files are moved/renamed into the canonical
``<scenario>_<slug>.png`` form inside the matching variable subfolder.
Already-canonical files are left untouched.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_COMPARE = SCRIPT_DIR / "figures" / "comparisons"
FIG_BY_VAR = FIG_COMPARE / "by_variable"


def canonical_name(scenario: str, slug: str) -> str:
    return f"{scenario}_{slug}.png"


def move_flat_files() -> int:
    """Move ``comparisons/<scenario>__<slug>.png`` into the by_variable tree."""
    moved = 0
    for png in sorted(FIG_COMPARE.glob("*.png")):
        name = png.stem
        if "__" not in name:
            print(f"  ? skip (no '__' separator): {png.name}")
            continue
        scenario, slug = name.split("__", 1)
        dest_dir = FIG_BY_VAR / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / canonical_name(scenario, slug)
        if dest.exists():
            print(f"  - dest exists, removing source: {png.name}")
            png.unlink()
        else:
            print(f"  + {png.name} -> {dest.relative_to(FIG_COMPARE)}")
            png.rename(dest)
            moved += 1
    return moved


def rename_inside_subfolders() -> int:
    """Rename files inside each variable subfolder to ``<scenario>_<slug>.png``."""
    renamed = 0
    if not FIG_BY_VAR.is_dir():
        return 0
    for var_dir in sorted(p for p in FIG_BY_VAR.iterdir() if p.is_dir()):
        slug = var_dir.name
        canonical_suffix = f"_{slug}"
        double_suffix = f"__{slug}"
        for png in sorted(var_dir.glob("*.png")):
            stem = png.stem
            if stem.endswith(canonical_suffix):
                continue  # already canonical
            if stem.endswith(double_suffix):
                scenario = stem[: -len(double_suffix)]
            else:
                # Bare scenario filename: <scenario>.png
                scenario = stem
            new_name = canonical_name(scenario, slug)
            dest = var_dir / new_name
            if dest == png:
                continue
            if dest.exists():
                print(f"  - dest exists, removing source: {png.relative_to(FIG_BY_VAR)}")
                png.unlink()
            else:
                print(f"  + {png.relative_to(FIG_BY_VAR)} -> {dest.relative_to(FIG_BY_VAR)}")
                png.rename(dest)
                renamed += 1
    return renamed


def main() -> None:
    if not FIG_COMPARE.exists():
        print(f"Nothing to do: {FIG_COMPARE} does not exist.")
        return
    print(f"Working in: {FIG_COMPARE}")
    print("Step 1: moving flat files into by_variable/...")
    n_moved = move_flat_files()
    print(f"  moved {n_moved} files")
    print("Step 2: renaming files inside variable subfolders...")
    n_renamed = rename_inside_subfolders()
    print(f"  renamed {n_renamed} files")


if __name__ == "__main__":
    main()
