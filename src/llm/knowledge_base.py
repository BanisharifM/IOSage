"""Compatibility entry point for measured knowledge-base construction."""

from src.llm.evidence_kb import (
    build_knowledge_base,
    main,
    write_knowledge_base,
)

__all__ = ["build_knowledge_base", "write_knowledge_base", "main"]


def build_kb_entries(*_args, **_kwargs):
    raise RuntimeError(
        "unmeasured KB construction is retired; use build_knowledge_base"
    )


def export_for_tabassum(*_args, **_kwargs):
    raise RuntimeError(
        "per-dimension exports are retired; use the versioned KB document"
    )


if __name__ == "__main__":
    main()
