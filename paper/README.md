# SC 2026 Paper: ML+LLM Hybrid for HPC I/O Bottleneck Detection

LaTeX source for the SC 2026 submission.

**Template:** IEEE Conference (IEEEtran.cls)
**Page limit:** 10 pages excluding bibliography
**Review:** Double-anonymous

## Building

### On Overleaf

This repository is synced to Overleaf via GitHub integration. Pull from GitHub in Overleaf to get the latest changes.

### Locally

```bash
# From the paper/ directory
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Or from the project root
bash scripts/compile_paper.sh
```

## Structure

```
paper/
├── main.tex             # Main document (includes all sections)
├── references.bib       # Bibliography
├── IEEEtran.cls         # IEEE template class
├── IEEEtran.bst         # IEEE bibliography style
├── sections/            # One file per section
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── system_design.tex
│   ├── dataset.tex
│   ├── methodology.tex
│   ├── evaluation.tex
│   ├── discussion.tex
│   └── conclusion.tex
├── figures/
│   ├── preprocessing/   # Pipeline figures (PDF + PNG)
│   └── eda/             # Exploratory analysis plots
└── tables/
    ├── tab_dataset_summary.tex
    └── tab_normalization.tex
```

## Sections

| # | Section | Target | File |
|---|---------|--------|------|
| - | Abstract | 250 words | `sections/abstract.tex` |
| 1 | Introduction | 1.5 pages | `sections/introduction.tex` |
| 2 | Background and Related Work | 1.5 pages | `sections/related_work.tex` |
| 3 | System Design | 2 pages | `sections/system_design.tex` |
| 4 | Dataset and Preprocessing | 1.5 pages | `sections/dataset.tex` |
| 5 | Methodology | 2 pages | `sections/methodology.tex` |
| 6 | Evaluation | 2 pages | `sections/evaluation.tex` |
| 7 | Discussion | 0.5 pages | `sections/discussion.tex` |
| 8 | Conclusion | 0.5 pages | `sections/conclusion.tex` |

## Custom Commands

| Command | Output | Usage |
|---------|--------|-------|
| `\system` | IOSage | System name (placeholder) |
| `\dataset` | Polaris-IO | Dataset name |
| `\ie` | i.e., | With smart spacing |
| `\eg` | e.g., | With smart spacing |
| `\etal` | et al. | With smart spacing |
