"""
Job-Level Aggregation
=====================
Handles the transition between per-file Darshan records and job-level
aggregated counters.

In the current pipeline, aggregation is handled by two backends:

  1. **PyDarshan backend** (``parse_darshan._extract_pydarshan_module``):
     Implements all 7 Darshan aggregation rules (SUM, MAX, MIN_NONZERO,
     LAST_VALUE, TOP-4 MERGE, CONDITIONAL, ZEROED) directly from the
     per-file DataFrames.  This is the primary and correct aggregation.

  2. **CLI backend** (``darshan-parser --total``): Darshan's own C code
     performs the aggregation.  We parse the already-aggregated output.

This module previously contained custom aggregation logic, but that has
been superseded by the correct rule implementations in ``parse_darshan.py``.
It now provides only utility functions for working with aggregated data.

See ``docs/darshan_counter_aggregation.md`` for full documentation of the
7 aggregation rules.
"""

import logging

logger = logging.getLogger(__name__)
