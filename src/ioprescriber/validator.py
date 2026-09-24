"""Retired fixed-pair validation interface.

The former implementation compared hand-authored benchmark configurations.
It did not apply or execute the recommendation produced by the LLM, so it
cannot establish source-code fix correctness or speedup.
"""


class Validator:
    """Reject use of the retired fixed-pair validation path."""

    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "fixed-pair validation is retired; a source-aware build, correctness, "
            "and execution protocol must be configured before closed-loop claims"
        )
