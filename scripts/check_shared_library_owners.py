"""Fail if two installed packages own the same shared-library path.

pip lets two distributions ship the same file and silently keeps whichever was installed
last. The CUDA 11 and CUDA 12 NVIDIA wheels do exactly that (for example both provide
nvidia/nccl/lib/libnccl.so.2), so an orphan CUDA 11 wheel can replace the library a CUDA 12
PyTorch needs and the failure only appears at import time on a GPU node.

Usage: python scripts/check_shared_library_owners.py   (exit 1 on any collision)
"""
import collections
import glob
import logging
import os
import sys
import sysconfig

logger = logging.getLogger("check_shared_library_owners")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    site = sysconfig.get_paths()["purelib"]
    owners = collections.defaultdict(set)
    for record in glob.glob(os.path.join(site, "*.dist-info", "RECORD")):
        package = os.path.basename(os.path.dirname(record))[: -len(".dist-info")]
        with open(record, errors="ignore") as f:
            for line in f:
                path = line.split(",")[0]
                if ".so" in os.path.basename(path):
                    owners[path].add(package)
    collisions = {p: sorted(o) for p, o in owners.items() if len(o) > 1}
    for path, pkgs in sorted(collisions.items()):
        logger.error("%s is owned by %s", path, ", ".join(pkgs))
    if collisions:
        logger.error("%d shared-library path(s) have more than one owner", len(collisions))
        sys.exit(1)
    logger.info("no shared-library path has more than one owner (%d checked)", len(owners))


if __name__ == "__main__":
    main()
