# IOSage

IOSage combines multi-label I/O bottleneck detection with benchmark evidence, source-level recommendations, and measured closed-loop validation. The active work targets the IPDPS 2027 resubmission.

The `v1.0.0` tag is the published SC 2026 artifact snapshot. Results stored under historical `results/` directories belong to that snapshot unless a manifest says otherwise. They are not evidence for the active resubmission run.

## Pipeline

1. Parse Darshan logs and build a versioned feature table.
2. Assign heuristic labels to the production corpus.
3. verify benchmark jobs against construction labels and extract the same feature schema.
4. Train the biquality detector with grouped benchmark partitions.
5. Compute per-label attribution for the final detector bundle.
6. Build a knowledge base from accepted development evidence and measured fixes.
7. Detect, retrieve evidence, generate a code change, and measure the result.

The current implementation requires hashes and manifests at stage boundaries. Failed or incomplete stages do not publish a result.

## Current status

The resubmission pipeline and its software checks are under active repair and rerun. The benchmark inventory currently contains accepted and excluded rows, but the full benchmark verification, feature extraction, final detector training, attribution, LLM evaluation, and iterative closed-loop runs have not all completed on the new pipeline. See [`docs/7_resubmission/ROADMAP_STATUS.md`](docs/7_resubmission/ROADMAP_STATUS.md) for the live gates.

Paper figure generation from result data stays blocked until a current validated result manifest exists. Historical metrics remain available through the `v1.0.0` tag and its archived artifacts.

## Installation

The project environment used on Delta is:

```bash
PYTHONNOUSERSITE=1 /work/nvme/bdau/mbanisharifdehkordi/envs/iosage/bin/python scripts/run_tests.py
```

For a separate installation, use the pinned manifests:

```bash
conda env create -f environment.yml
conda activate iosage
python -m pip install -r requirements-test.txt
PYTHONNOUSERSITE=1 python -m pytest
```

Optional requirements are split by purpose:

- `requirements-wisio.txt` for the WisIO comparison.
- `requirements-notebooks.txt` for notebooks.
- `requirements-test.txt` for repository tests.

## Reproduction driver

`scripts/reproduce_all.sh` runs one selected stage or the maintained sequence. It creates an immutable directory below `results/resubmission/reproduction/` and stops when a required artifact is absent.

```bash
bash scripts/reproduce_all.sh --step 1 --run-id environment_check_YYYYMMDD
```

The full sequence needs the real Darshan corpus, verified benchmark logs, a final model bundle, measured knowledge-base evidence, API credentials for live model calls, and Delta access for closed-loop measurements. The driver does not substitute cached or historical outputs when one of those inputs is missing.

## Repository layout

```text
src/data/             Darshan parsing, feature extraction, and preprocessing
src/models/           Biquality training and attribution
src/ioprescriber/     Detection, retrieval, recommendation, and validation
src/llm/              Knowledge-base and iterative optimization components
configs/              Maintained experiment configuration
benchmarks/           Benchmark definitions and SLURM generators
scripts/              Pipeline, verification, and analysis entry points
tests/                Contract and regression tests
docs/7_resubmission/  Active roadmap and audit records
```

The root `paper` path is a compatibility symlink to the frozen SC 2026 repository. Active paper edits belong in the separate `papers/IPDPS_2027` repository.

## Data and provenance

Large inputs and generated outputs are not committed to the public repository. Each current result must identify its input paths, hashes, configuration, run directory, and source revision. `data/processed/README.md` describes the current and historical local layouts.

The published production corpus is available from [Zenodo](https://doi.org/10.5281/zenodo.15052603).

## License

See [LICENSE](LICENSE).
