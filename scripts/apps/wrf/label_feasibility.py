"""Label-feasibility report for WRF Darshan logs (TASK_02, phase 0).

Runs each log through the canonical parser (``parse_darshan_log``), the raw and derived
feature path (``engineer_one``), the target-validity mask and the registered label rules
(``src.data.label_rules``), and writes one JSON report with the input paths and hashes, the
code commit, the feature schema version, the label-definition hash, module completeness,
every rule value, every validity bit and the derived labels. The report also separates the
job-wide counters by file (restart reads, history writes, other files) and by rank, so that
the reason for a label can be read from the log itself.

Usage:
    python scripts/apps/wrf/label_feasibility.py --log <a.darshan> --log <b.darshan> \
        --output results/apps/wrf/phase0/label_feasibility.json
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

from src.data import label_rules  # noqa: E402
from src.data.feature_extraction import FEATURE_SCHEMA_VERSION  # noqa: E402
from src.data.parse_darshan import parse_darshan_log  # noqa: E402
from src.data.preprocessing import engineer_one, load_preprocessing_config  # noqa: E402

logger = logging.getLogger("wrf_label_feasibility")

RULE_FEATURES = [
    "nprocs", "POSIX_READS", "POSIX_WRITES", "POSIX_SEQ_READS", "POSIX_SEQ_WRITES",
    "POSIX_FILE_NOT_ALIGNED", "POSIX_FSYNCS", "POSIX_FDSYNCS",
    "MPIIO_INDEP_READS", "MPIIO_INDEP_WRITES", "MPIIO_COLL_READS", "MPIIO_COLL_WRITES",
    "MPIIO_NB_READS", "MPIIO_NB_WRITES",
    "rank_byte_range_ratio", "SHARED_BYTE_IMBALANCE", "SHARED_TIME_IMBALANCE",
    "FILE_WRITE_IMBALANCE", "FILE_READ_IMBALANCE", "metadata_time_ratio_all", "io_bytes_all",
    "num_files", "num_data_files", "partial_posix", "partial_mpiio", "partial_stdio",
]
SIZE_BINS = ["0_100", "100_1K", "1K_10K", "10K_100K", "100K_1M", "1M_4M", "4M_10M", "10M_100M", "100M_1G", "1G_PLUS"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_file_breakdown(log_path):
    """Bytes, operations, writing and reading ranks per file name and module, plus per-rank
    byte totals, read directly from the Darshan records."""
    import darshan

    report = darshan.DarshanReport(str(log_path), read_all=True)
    names = report.name_records
    files = defaultdict(dict)
    rank_bytes = defaultdict(int)
    for module, tag in (("POSIX", "POSIX"), ("MPI-IO", "MPIIO"), ("STDIO", "STDIO")):
        if module not in report.records:
            continue
        counter_names = report.counters[module]["counters"]
        idx = {name: counter_names.index(name) for name in counter_names}
        for rec in report.records[module]:
            name = names.get(rec["id"], "")
            c = rec["counters"]
            entry = files[name].setdefault(module, {"bytes_written": 0, "bytes_read": 0, "writes": 0, "reads": 0,
                                                    "ranks": set(), "shared": False})
            written = int(c[idx[f"{tag}_BYTES_WRITTEN"]])
            read = int(c[idx[f"{tag}_BYTES_READ"]])
            entry["bytes_written"] += written
            entry["bytes_read"] += read
            if module == "MPI-IO":
                entry["writes"] += int(c[idx["MPIIO_INDEP_WRITES"]]) + int(c[idx["MPIIO_COLL_WRITES"]])
                entry["reads"] += int(c[idx["MPIIO_INDEP_READS"]]) + int(c[idx["MPIIO_COLL_READS"]])
                entry["coll_writes"] = entry.get("coll_writes", 0) + int(c[idx["MPIIO_COLL_WRITES"]])
                entry["coll_reads"] = entry.get("coll_reads", 0) + int(c[idx["MPIIO_COLL_READS"]])
            else:
                entry["writes"] += int(c[idx[f"{tag}_WRITES"]])
                entry["reads"] += int(c[idx[f"{tag}_READS"]])
            if module == "POSIX":
                entry["not_aligned"] = entry.get("not_aligned", 0) + int(c[idx["POSIX_FILE_NOT_ALIGNED"]])
                entry["seq_writes"] = entry.get("seq_writes", 0) + int(c[idx["POSIX_SEQ_WRITES"]])
                entry["seq_reads"] = entry.get("seq_reads", 0) + int(c[idx["POSIX_SEQ_READS"]])
                for direction in ("READ", "WRITE"):
                    entry[f"size_{direction.lower()}"] = [
                        entry.get(f"size_{direction.lower()}", [0] * len(SIZE_BINS))[i]
                        + int(c[idx[f"POSIX_SIZE_{direction}_{b}"]]) for i, b in enumerate(SIZE_BINS)]
            rank = int(rec["rank"])
            if rank < 0:
                entry["shared"] = True
            else:
                entry["ranks"].add(rank)
                if module == "POSIX":
                    rank_bytes[rank] += written + read
    out = {}
    for name, modules in files.items():
        short = name.split("/")[-1] if name.startswith("/") else name
        out[short] = {m: dict(e, ranks=sorted(e["ranks"])) for m, e in modules.items()}
    return {"files": out, "posix_rank_bytes": {str(k): v for k, v in sorted(rank_bytes.items())},
            "modules": {m: {"partial": bool(v.get("partial_flag")), "records": v.get("num_records")}
                        for m, v in report.modules.items()},
            "nprocs": int(report.metadata["job"]["nprocs"]), "run_time_s": float(report.metadata["job"]["run_time"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="append", required=True, help="Darshan log (repeat for several)")
    parser.add_argument("--label", action="append", default=[], help="name for each log, in order")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    labels = args.label + [Path(p).name for p in args.log[len(args.label):]]
    config = load_preprocessing_config()
    commit = subprocess.run(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(PROJECT_DIR), "status", "--porcelain"], capture_output=True,
                           text=True, check=True).stdout.strip().splitlines()
    report = {"code_commit": commit, "working_tree_modified_files": len(dirty),
              "feature_schema_version": FEATURE_SCHEMA_VERSION,
              "label_definitions": str(label_rules.LABEL_DEFINITIONS_PATH),
              "label_definitions_sha256": sha256_file(label_rules.LABEL_DEFINITIONS_PATH),
              "thresholds": label_rules.LABEL_DEFINITIONS["thresholds"], "logs": {}}
    for name, path in zip(labels, args.log):
        parsed = parse_darshan_log(path, strict=True)
        features = engineer_one(parsed, config=config)
        rules = label_rules.rule_frame(features).iloc[0].to_dict()
        derived = label_rules.labels_from_features(features).iloc[0].to_dict()
        validity = label_rules.validity_from_features(features).iloc[0].to_dict()
        entry = {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
                 "partial_modules": parsed["partial_modules"], "modules": parsed["modules"],
                 "rule_features": {k: (float(features[k]) if k in features else None) for k in RULE_FEATURES},
                 "posix_size_read": {b: float(features[f"POSIX_SIZE_READ_{b}"]) for b in SIZE_BINS},
                 "posix_size_write": {b: float(features[f"POSIX_SIZE_WRITE_{b}"]) for b in SIZE_BINS},
                 "mpiio_size_write_agg": {b: float(features[f"MPIIO_SIZE_WRITE_AGG_{b}"]) for b in SIZE_BINS},
                 "mpiio_size_read_agg": {b: float(features[f"MPIIO_SIZE_READ_AGG_{b}"]) for b in SIZE_BINS},
                 "rules": {k: bool(v) for k, v in rules.items()},
                 "validity": {k: int(v) for k, v in validity.items()},
                 "labels": {k: int(v) for k, v in derived.items()},
                 "rule_details": label_rules.rule_details(features),
                 "positive_labels": [k for k, v in derived.items() if v and k != "healthy"],
                 "per_file": per_file_breakdown(path)}
        report["logs"][name] = entry
        logger.info("%s: positive labels %s; validity %s", name, entry["positive_labels"],
                    {k: v for k, v in entry["validity"].items() if not v} or "all valid")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
