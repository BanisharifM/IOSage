"""Path checks for generated paper and artifact outputs."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
FROZEN_PAPER_DIR = (PROJECT_DIR / "papers" / "SC_2026").resolve()
LEGACY_PAPER_DIR = PROJECT_DIR / "paper"


def checked_output_dir(path):
    """Resolve an output directory and reject the frozen paper repository."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    absolute = candidate.absolute()
    if absolute == LEGACY_PAPER_DIR or LEGACY_PAPER_DIR in absolute.parents:
        raise ValueError(f"output directory uses the retired frozen-paper path: {absolute}")
    resolved = candidate.resolve()
    if resolved == FROZEN_PAPER_DIR or FROZEN_PAPER_DIR in resolved.parents:
        raise ValueError(f"output directory is inside the frozen paper repository: {resolved}")
    return resolved
