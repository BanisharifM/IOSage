# Remaining Tasks — SC 2026 Submission

**Last updated**: 2026-04-06 06:40
**Deadline**: Apr 8, 2026
**Model**: Boost experiment (689 GT, 488 test)

---

## PENDING (Waiting for SLURM results)

| Item | SLURM Job | What it produces | Paper location |
|------|-----------|-----------------|----------------|
| LLM quality table | 17324832 | Groundedness per model (Claude/GPT-4o/Llama) | Table VI, eval text line 149 |
| TraceBench full (per-app + cross-system) | 17324832 | Per-app TP/FP, cross-system F1 | Tables VIII, IX |
| SHAP from new model | 17324832 | New SHAP values + Fig 4 | Fig 4 |
| Fig 7 t-SNE | needs regeneration | t-SNE with 689 benchmark samples | Fig 7 |
| E2E benchmark | NOT submitted yet | May change 6.5x speedup if model detects differently | Eval text, conclusion |

## PENDING (Fair IOAgent comparison)

| Item | Status | Purpose |
|------|--------|---------|
| IOSage with gpt-4.1-mini | needs setup + submit | Fair comparison: same LLM as IOAgent |
| IOAgent with Claude/GPT-4o/Llama | needs setup + submit | Fair comparison: same LLMs as IOSage |

## DECISION NEEDED

| Item | Question |
|------|----------|
| E2E re-run | Does E2E pipeline analysis with new model change the 6.5x? Need to re-run to verify. |
| Adopt boost results? | All paper numbers already updated to boost. If reverting, need to undo all changes. |

---

## COMPLETED (This session)

All paper numbers updated to boost experiment (689 GT):
- Abstract, Introduction, System Design, Dataset, Evaluation, Discussion, Conclusion
- Tables: II, III, IV, V, VII, X, Latency
- Figures: 2 (689 samples), 3 (new F1), 5 (unchanged), 6 (updated)
- All text references: sample counts, F1 values, speedup ranges, latency, etc.
