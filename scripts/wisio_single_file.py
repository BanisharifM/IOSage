"""Run WisIO on a single darshan file and output JSON result to stdout."""

import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["DASK_DISTRIBUTED__LOGGING__DISTRIBUTED"] = "error"
os.environ["DASK_LOGGING__DISTRIBUTED"] = "error"
logging.disable(logging.CRITICAL)

WISIO_RULES = [
    "excessive_metadata_access",
    "operation_imbalance",
    "random_operations",
    "size_imbalance",
    "small_reads",
    "small_writes",
]


def main():
    trace_path = sys.argv[1]

    from dask.distributed import LocalCluster, Client
    from wisio.darshan import DarshanAnalyzer

    cluster = LocalCluster(
        n_workers=1, threads_per_worker=1, memory_limit="4GB", silence_logs=50
    )
    client = Client(cluster)

    try:
        analyzer = DarshanAnalyzer(
            checkpoint=False,
            checkpoint_dir="",
            bottleneck_dir="/tmp/wisio_bot",
            verbose=False,
        )
        result = analyzer.analyze_trace(
            trace_path=trace_path,
            percentile=0.9,
            view_types=["file_name", "proc_name"],
            metrics=["iops"],
            exclude_bottlenecks=[],
            exclude_characteristics=[],
        )

        if result._bottlenecks is not None:
            bot_df = result._bottlenecks.compute()
            flags = {}
            for rule in WISIO_RULES:
                if rule in bot_df.columns:
                    flags[rule] = bool(bot_df[rule].any())
                else:
                    flags[rule] = False
        else:
            flags = {r: False for r in WISIO_RULES}

        print(json.dumps({"status": "ok", "flags": flags}))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
