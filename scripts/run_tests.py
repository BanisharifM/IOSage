#!/usr/bin/env python3
"""Run every zero-argument repository test without an external test runner."""

import argparse
import importlib.util
import inspect
import logging
import sys
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_DIR / "tests"
logger = logging.getLogger("run_tests")
sys.path.insert(0, str(PROJECT_DIR))


def load_module(path):
    name = f"iosage_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="test files; default: tests/test_*.py")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = [Path(value) for value in args.paths] if args.paths else sorted(TEST_DIR.glob("test_*.py"))
    if not paths:
        parser.error("no test files selected")
    passed = 0
    failed = []
    for path in paths:
        module = load_module(path.resolve())
        functions = [
            function for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_") and function.__module__ == module.__name__
        ]
        if not functions:
            failed.append((str(path), "no test functions found"))
            continue
        for function in functions:
            if inspect.signature(function).parameters:
                failed.append((f"{path}:{function.__name__}", "requires runner fixtures"))
                continue
            try:
                function()
                passed += 1
            except Exception:
                failed.append((f"{path}:{function.__name__}", traceback.format_exc()))
    logger.info("passed=%d failed=%d", passed, len(failed))
    for name, error in failed:
        logger.error("FAIL %s\n%s", name, error)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
