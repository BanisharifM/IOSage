"""
Validate and build benchmark commands from LLM-proposed parameter changes.

Safety layer: ensures LLM output maps to valid, safe IOR/mdtest/h5bench/DLIO commands.
Uses allowlists from configs/iterative.yaml to constrain parameter ranges.
Prevents execution of harmful or nonsensical commands.

References:
  - STELLAR (SC'25): Constrains LLM to known Lustre parameters
  - PerfCodeGen (2025): Validation phase before execution
"""

import json
import logging
import os
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class BenchmarkCommandBuilder:
    """Validates LLM-proposed benchmark parameters and builds safe commands."""

    def __init__(self, config_path=None, scratch_dir=None):
        config_path = config_path or PROJECT_DIR / "configs" / "iterative.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.allowlist = config.get("ior_allowlist", {})
        self.scratch_dir = scratch_dir or config["slurm"]["scratch_dir"]

        # Valid IOR APIs
        self.valid_apis = set(self.allowlist.get("api", ["POSIX", "MPIIO"]))

        # Transfer size range
        ts_range = self.allowlist.get("transfer_size_range", [64, 16777216])
        self.min_transfer_size = ts_range[0]
        self.max_transfer_size = ts_range[1]

        # Block size range
        bs_range = self.allowlist.get("block_size_range", [65536, 1073741824])
        self.min_block_size = bs_range[0]
        self.max_block_size = bs_range[1]

        # Segments range
        seg_range = self.allowlist.get("segments_range", [1, 1000])
        self.min_segments = seg_range[0]
        self.max_segments = seg_range[1]

        # Allowed flags
        self.allowed_flags = set(self.allowlist.get("max_flags", []))

    def parse_size(self, size_str):
        """Parse size string like '1M', '64K', '4096' to integer bytes."""
        if isinstance(size_str, (int, float)):
            return int(size_str)
        size_str = str(size_str).strip().upper()
        multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
        for suffix, mult in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-1]) * mult)
        return int(size_str)

    def validate_ior_params(self, params):
        """Validate LLM-proposed IOR parameters against allowlists.

        Args:
            params: dict with keys like api, transfer_size, block_size, segments,
                    file_per_proc, extra_flags, collective, etc.

        Returns:
            (valid, sanitized_params, errors) tuple
        """
        errors = []
        sanitized = {}

        # API
        api = params.get("api", "POSIX").upper()
        if api not in self.valid_apis:
            errors.append(f"Invalid API '{api}', must be one of {self.valid_apis}")
            api = "POSIX"
        sanitized["api"] = api

        # Transfer size
        try:
            ts = self.parse_size(params.get("transfer_size", 65536))
            if ts < self.min_transfer_size:
                errors.append(f"Transfer size {ts} below minimum {self.min_transfer_size}")
                ts = self.min_transfer_size
            if ts > self.max_transfer_size:
                errors.append(f"Transfer size {ts} above maximum {self.max_transfer_size}")
                ts = self.max_transfer_size
            sanitized["transfer_size"] = str(ts)
        except (ValueError, TypeError):
            errors.append(f"Invalid transfer_size: {params.get('transfer_size')}")
            sanitized["transfer_size"] = "65536"

        # Block size
        try:
            bs = self.parse_size(params.get("block_size", "100M"))
            bs = max(self.min_block_size, min(bs, self.max_block_size))
            sanitized["block_size"] = str(bs)
        except (ValueError, TypeError):
            sanitized["block_size"] = str(100 * 1024 * 1024)

        # Segments
        try:
            seg = int(params.get("segments", 10))
            seg = max(self.min_segments, min(seg, self.max_segments))
            sanitized["segments"] = str(seg)
        except (ValueError, TypeError):
            sanitized["segments"] = "10"

        # File per proc
        sanitized["file_per_proc"] = bool(params.get("file_per_proc", True))

        # Extra flags - only allow known safe flags
        extra = params.get("extra_flags", "-e -C -w -r")
        # Pre-process: merge "-O useO_DIRECT=1" into single token "-O useO_DIRECT=1"
        extra_str = str(extra)
        extra_str = re.sub(r'-O\s+useO_DIRECT=1', '-O useO_DIRECT=1', extra_str)
        tokens = []
        for part in extra_str.split():
            if part == "-O":
                # Standalone -O; will be merged with next token if it's useO_DIRECT=1
                tokens.append(part)
            elif tokens and tokens[-1] == "-O" and "useO_DIRECT" in part:
                tokens[-1] = f"-O {part}"
            else:
                tokens.append(part)
        safe_flags = []
        for flag in tokens:
            # Always include -w -r (write and read)
            if flag in ("-w", "-r"):
                safe_flags.append(flag)
            elif flag in self.allowed_flags:
                safe_flags.append(flag)
            elif flag.startswith("-O") and "useO_DIRECT" in flag:
                safe_flags.append(flag)
            else:
                errors.append(f"Filtered unsafe flag: {flag}")
        # Ensure we always have -w -r
        if "-w" not in safe_flags:
            safe_flags.append("-w")
        if "-r" not in safe_flags:
            safe_flags.append("-r")
        sanitized["extra_flags"] = " ".join(safe_flags)

        # Collective (MPI-IO only)
        if api == "MPIIO" and params.get("collective", False):
            if "-c" not in sanitized["extra_flags"]:
                sanitized["extra_flags"] += " -c"

        # O_DIRECT constraint: requires transfer_size >= 4096
        ts_int = int(sanitized["transfer_size"])
        if "useO_DIRECT" in sanitized["extra_flags"] and ts_int < 4096:
            errors.append(f"O_DIRECT requires transfer_size >= 4096, got {ts_int}")
            sanitized["extra_flags"] = sanitized["extra_flags"].replace(
                "-O useO_DIRECT=1", ""
            ).strip()

        # IOR constraint: block_size must be a multiple of transfer_size
        ts_int = int(sanitized["transfer_size"])
        bs_int = int(sanitized["block_size"])
        if ts_int > 0 and bs_int % ts_int != 0:
            # Round block_size up to next multiple of transfer_size
            new_bs = ((bs_int + ts_int - 1) // ts_int) * ts_int
            new_bs = min(new_bs, self.max_block_size)
            if new_bs % ts_int != 0:
                # If rounding up exceeds max, round down
                new_bs = (self.max_block_size // ts_int) * ts_int
            errors.append(
                f"block_size {bs_int} not a multiple of transfer_size {ts_int}, "
                f"adjusted to {new_bs}"
            )
            sanitized["block_size"] = str(new_bs)

        valid = len(errors) == 0
        return valid, sanitized, errors

    def build_ior_command(self, params, output_dir=None):
        """Build a safe IOR command string from validated parameters.

        Args:
            params: dict from validate_ior_params
            output_dir: override output directory

        Returns:
            command string
        """
        out = output_dir or self.scratch_dir
        cmd = f"ior -a {params['api']}"
        cmd += f" -t {params['transfer_size']}"
        cmd += f" -b {params['block_size']}"
        cmd += f" -s {params['segments']}"
        if params.get("file_per_proc"):
            cmd += " -F"
        cmd += f" {params['extra_flags']}"
        cmd += f" -o {out}/ior_test_file"
        return cmd

    def build_mdtest_command(self, params, output_dir=None):
        """Build a safe mdtest command from parameters."""
        out = output_dir or self.scratch_dir
        cmd = "mdtest"
        cmd += f" -n {params.get('items_per_rank', 1000)}"
        if params.get("write_bytes", 0) > 0:
            cmd += f" -w {params['write_bytes']}"
        if params.get("read_bytes", 0) > 0:
            cmd += f" -e {params['read_bytes']}"
        if params.get("files_only", True):
            cmd += " -F"
        if params.get("unique_dir", False):
            cmd += " -u"
        cmd += f" -d {out}/mdtest_dir"
        return cmd

    # =========================================================================
    # HACC-IO Validation and Command Building
    # =========================================================================

    HACC_EXECUTABLES = {"posix_shared", "mpiio_shared", "fpp"}
    HACC_IO_DIR = "/work/hdd/bdau/mbanisharifdehkordi/hacc-io"

    def validate_hacc_params(self, params):
        """Validate LLM-proposed HACC-IO parameters.

        Args:
            params: dict with keys like executable, num_particles,
                    collective_buffering.

        Returns:
            (valid, sanitized_params, errors) tuple
        """
        errors = []
        sanitized = {}

        # Executable
        exe = params.get("executable", "posix_shared")
        if exe not in self.HACC_EXECUTABLES:
            errors.append(
                f"Invalid executable '{exe}', must be one of {self.HACC_EXECUTABLES}"
            )
            exe = "posix_shared"
        sanitized["executable"] = exe

        # Number of particles
        try:
            np = int(params.get("num_particles", 200))
            if np < 50:
                errors.append(f"num_particles {np} below minimum 50")
                np = 50
            if np > 10_000_000:
                errors.append(f"num_particles {np} above maximum 10000000")
                np = 10_000_000
            sanitized["num_particles"] = np
        except (ValueError, TypeError):
            errors.append(f"Invalid num_particles: {params.get('num_particles')}")
            sanitized["num_particles"] = 200

        # Collective buffering (pass-through, handled at SLURM level)
        sanitized["collective_buffering"] = params.get("collective_buffering", "disabled")

        valid = len(errors) == 0
        return valid, sanitized, errors

    def build_hacc_command(self, params, output_dir=None):
        """Build a safe HACC-IO command string from validated parameters.

        Args:
            params: dict from validate_hacc_params
            output_dir: override output directory

        Returns:
            command string
        """
        out = output_dir or self.scratch_dir
        exe = params.get("executable", "posix_shared")
        np = params.get("num_particles", 200)
        cmd = f"{self.HACC_IO_DIR}/hacc_io_{exe} {np} {out}/hacc_checkpoint"
        return cmd

    # =========================================================================
    # Custom (load_imbalance) Validation and Command Building
    # =========================================================================

    CUSTOM_SCRIPT = (
        "/work/hdd/bdau/mbanisharifdehkordi/SC_2026/benchmarks/custom/load_imbalance.py"
    )
    CUSTOM_PYTHON = "/projects/bdau/envs/sc2026/bin/python"

    def validate_custom_params(self, params):
        """Validate LLM-proposed custom load_imbalance parameters.

        Args:
            params: dict with keys like imbalance_factor, base_size_mb.

        Returns:
            (valid, sanitized_params, errors) tuple
        """
        errors = []
        sanitized = {}

        # Imbalance factor
        try:
            factor = float(params.get("imbalance_factor", 10))
            if factor < 1.0:
                errors.append(f"imbalance_factor {factor} below minimum 1.0")
                factor = 1.0
            if factor > 100.0:
                errors.append(f"imbalance_factor {factor} above maximum 100.0")
                factor = 100.0
            sanitized["imbalance_factor"] = factor
        except (ValueError, TypeError):
            errors.append(f"Invalid imbalance_factor: {params.get('imbalance_factor')}")
            sanitized["imbalance_factor"] = 10.0

        # Base size MB
        try:
            base = int(params.get("base_size_mb", 10))
            if base < 1:
                errors.append(f"base_size_mb {base} below minimum 1")
                base = 1
            if base > 500:
                errors.append(f"base_size_mb {base} above maximum 500")
                base = 500
            sanitized["base_size_mb"] = base
        except (ValueError, TypeError):
            errors.append(f"Invalid base_size_mb: {params.get('base_size_mb')}")
            sanitized["base_size_mb"] = 10

        valid = len(errors) == 0
        return valid, sanitized, errors

    def build_custom_command(self, params, output_dir=None):
        """Build a safe custom load_imbalance command string.

        Args:
            params: dict from validate_custom_params
            output_dir: override output directory

        Returns:
            command string
        """
        out = output_dir or self.scratch_dir
        factor = params.get("imbalance_factor", 10)
        base = params.get("base_size_mb", 10)
        cmd = (
            f"{self.CUSTOM_PYTHON} {self.CUSTOM_SCRIPT}"
            f" --imbalance-factor {factor}"
            f" --base-size-mb {base}"
            f" --output-dir {out}"
        )
        return cmd

    # =========================================================================
    # h5bench Validation and Command Building
    # =========================================================================

    H5BENCH_DIR = "/work/hdd/bdau/mbanisharifdehkordi/h5bench/build"

    VALID_MEM_PATTERNS = {"CONTIG", "INTERLEAVED"}
    VALID_FILE_PATTERNS = {"CONTIG", "INTERLEAVED"}

    def validate_h5bench_params(self, params):
        """Validate LLM-proposed h5bench parameters.

        Args:
            params: dict with keys like DIM_1, COLLECTIVE_DATA, TIMESTEPS,
                    MEM_PATTERN, FILE_PATTERN.

        Returns:
            (valid, sanitized_params, errors) tuple
        """
        errors = []
        sanitized = {}

        # DIM_1 (elements per rank per timestep; each element is 8 bytes double)
        try:
            dim1 = int(params.get("DIM_1", 1024))
            if dim1 < 64:
                errors.append(f"DIM_1 {dim1} below minimum 64")
                dim1 = 64
            if dim1 > 16_777_216:
                errors.append(f"DIM_1 {dim1} above maximum 16777216")
                dim1 = 16_777_216
            sanitized["DIM_1"] = dim1
        except (ValueError, TypeError):
            errors.append(f"Invalid DIM_1: {params.get('DIM_1')}")
            sanitized["DIM_1"] = 1024

        # COLLECTIVE_DATA
        coll = str(params.get("COLLECTIVE_DATA", "NO")).upper()
        if coll not in ("YES", "NO"):
            errors.append(f"Invalid COLLECTIVE_DATA '{coll}', must be YES or NO")
            coll = "NO"
        sanitized["COLLECTIVE_DATA"] = coll

        # COLLECTIVE_METADATA
        coll_meta = str(params.get("COLLECTIVE_METADATA", coll)).upper()
        if coll_meta not in ("YES", "NO"):
            coll_meta = coll
        sanitized["COLLECTIVE_METADATA"] = coll_meta

        # TIMESTEPS
        try:
            ts = int(params.get("TIMESTEPS", 5))
            if ts < 1:
                errors.append(f"TIMESTEPS {ts} below minimum 1")
                ts = 1
            if ts > 100:
                errors.append(f"TIMESTEPS {ts} above maximum 100")
                ts = 100
            sanitized["TIMESTEPS"] = ts
        except (ValueError, TypeError):
            errors.append(f"Invalid TIMESTEPS: {params.get('TIMESTEPS')}")
            sanitized["TIMESTEPS"] = 5

        # MEM_PATTERN
        mem_pat = str(params.get("MEM_PATTERN", "CONTIG")).upper()
        if mem_pat not in self.VALID_MEM_PATTERNS:
            errors.append(f"Invalid MEM_PATTERN '{mem_pat}'")
            mem_pat = "CONTIG"
        sanitized["MEM_PATTERN"] = mem_pat

        # FILE_PATTERN
        file_pat = str(params.get("FILE_PATTERN", "CONTIG")).upper()
        if file_pat not in self.VALID_FILE_PATTERNS:
            errors.append(f"Invalid FILE_PATTERN '{file_pat}'")
            file_pat = "CONTIG"
        sanitized["FILE_PATTERN"] = file_pat

        valid = len(errors) == 0
        return valid, sanitized, errors

    def build_h5bench_config(self, params, output_dir=None, config_path=None):
        """Generate h5bench JSON config file and .write/.read sub-configs.

        Args:
            params: dict from validate_h5bench_params
            output_dir: directory for HDF5 output files
            config_path: where to write the JSON config (and sub-configs)

        Returns:
            (write_command, read_command, config_file_path) tuple
        """
        out = output_dir or self.scratch_dir
        dim1 = str(params.get("DIM_1", 1024))
        coll = params.get("COLLECTIVE_DATA", "NO")
        coll_meta = params.get("COLLECTIVE_METADATA", coll)
        timesteps = str(params.get("TIMESTEPS", 5))
        mem_pat = params.get("MEM_PATTERN", "CONTIG")
        file_pat = params.get("FILE_PATTERN", "CONTIG")

        # Write configuration
        write_cfg = {
            "MEM_PATTERN": mem_pat,
            "FILE_PATTERN": file_pat,
            "TIMESTEPS": timesteps,
            "DELAYED_CLOSE_TIMESTEPS": "0",
            "COLLECTIVE_DATA": coll,
            "COLLECTIVE_METADATA": coll_meta,
            "EMULATED_COMPUTE_TIME_PER_TIMESTEP": "0 s",
            "NUM_DIMS": "1",
            "DIM_1": dim1,
            "DIM_2": "1",
            "DIM_3": "1",
            "CSV_FILE": "output.csv",
        }

        # Read configuration
        read_cfg = dict(write_cfg)
        read_cfg["READ_OPTION"] = "FULL"

        # Main JSON config
        main_config = {
            "mpi": {"command": "srun", "ranks": "WILL_BE_SET_BY_SLURM"},
            "vol": {},
            "file-system": {},
            "directory": out,
            "benchmarks": [
                {
                    "benchmark": "write",
                    "file": "h5bench_output.h5",
                    "configuration": write_cfg,
                },
                {
                    "benchmark": "read",
                    "file": "h5bench_output.h5",
                    "configuration": read_cfg,
                },
            ],
        }

        # Write main JSON config
        if config_path is None:
            config_path = os.path.join(out, "h5bench_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(main_config, f, indent=4)

        # Write .write sub-config (flat key=value format)
        write_sub_path = config_path + ".write"
        with open(write_sub_path, "w") as f:
            for k, v in write_cfg.items():
                f.write(f"{k}={v}\n")

        # Write .read sub-config (flat key=value format)
        read_sub_path = config_path + ".read"
        with open(read_sub_path, "w") as f:
            for k, v in read_cfg.items():
                f.write(f"{k}={v}\n")

        h5_file = os.path.join(out, "h5bench_output.h5")
        write_cmd = f"{self.H5BENCH_DIR}/h5bench_write {write_sub_path} {h5_file}"
        read_cmd = f"{self.H5BENCH_DIR}/h5bench_read {read_sub_path} {h5_file}"

        return write_cmd, read_cmd, config_path

    # =========================================================================
    # DLIO Validation and Command Building
    # =========================================================================

    DLIO_BIN = "/projects/bdau/envs/sc2026/bin/dlio_benchmark"

    VALID_DLIO_FORMATS = {"npz", "hdf5", "csv", "tfrecord"}
    VALID_DLIO_SHUFFLES = {"off", "random", "seed"}

    def validate_dlio_params(self, params):
        """Validate LLM-proposed DLIO benchmark parameters.

        Args:
            params: dict with keys like record_length, num_files_train,
                    batch_size, read_threads, computation_time, epochs,
                    format, sample_shuffle, file_shuffle.

        Returns:
            (valid, sanitized_params, errors) tuple
        """
        errors = []
        sanitized = {}

        # record_length (bytes per sample)
        try:
            rl = int(params.get("record_length", 1024))
            if rl < 64:
                errors.append(f"record_length {rl} below minimum 64")
                rl = 64
            if rl > 16_777_216:
                errors.append(f"record_length {rl} above maximum 16777216")
                rl = 16_777_216
            sanitized["record_length"] = rl
        except (ValueError, TypeError):
            errors.append(f"Invalid record_length: {params.get('record_length')}")
            sanitized["record_length"] = 1024

        # num_files_train
        try:
            nf = int(params.get("num_files_train", 1000))
            if nf < 10:
                errors.append(f"num_files_train {nf} below minimum 10")
                nf = 10
            if nf > 10_000:
                errors.append(f"num_files_train {nf} above maximum 10000")
                nf = 10_000
            sanitized["num_files_train"] = nf
        except (ValueError, TypeError):
            errors.append(f"Invalid num_files_train: {params.get('num_files_train')}")
            sanitized["num_files_train"] = 1000

        # num_samples_per_file
        try:
            ns = int(params.get("num_samples_per_file", 1))
            ns = max(1, min(ns, 1000))
            sanitized["num_samples_per_file"] = ns
        except (ValueError, TypeError):
            sanitized["num_samples_per_file"] = 1

        # batch_size
        try:
            bs = int(params.get("batch_size", 1))
            if bs < 1:
                errors.append(f"batch_size {bs} below minimum 1")
                bs = 1
            if bs > 256:
                errors.append(f"batch_size {bs} above maximum 256")
                bs = 256
            sanitized["batch_size"] = bs
        except (ValueError, TypeError):
            errors.append(f"Invalid batch_size: {params.get('batch_size')}")
            sanitized["batch_size"] = 1

        # read_threads
        try:
            rt = int(params.get("read_threads", 1))
            rt = max(1, min(rt, 16))
            sanitized["read_threads"] = rt
        except (ValueError, TypeError):
            sanitized["read_threads"] = 1

        # computation_time (seconds of simulated compute per batch)
        try:
            ct = float(params.get("computation_time", 0.01))
            ct = max(0.0, min(ct, 10.0))
            sanitized["computation_time"] = ct
        except (ValueError, TypeError):
            sanitized["computation_time"] = 0.01

        # epochs
        try:
            ep = int(params.get("epochs", 2))
            ep = max(1, min(ep, 10))
            sanitized["epochs"] = ep
        except (ValueError, TypeError):
            sanitized["epochs"] = 2

        # format
        fmt = str(params.get("format", "npz")).lower()
        if fmt not in self.VALID_DLIO_FORMATS:
            errors.append(f"Invalid format '{fmt}'")
            fmt = "npz"
        sanitized["format"] = fmt

        # sample_shuffle
        ss = str(params.get("sample_shuffle", "off")).lower()
        if ss not in self.VALID_DLIO_SHUFFLES:
            ss = "off"
        sanitized["sample_shuffle"] = ss

        # file_shuffle
        fs = str(params.get("file_shuffle", "off")).lower()
        if fs not in self.VALID_DLIO_SHUFFLES:
            fs = "off"
        sanitized["file_shuffle"] = fs

        # seed
        sanitized["seed"] = int(params.get("seed", 42))

        valid = len(errors) == 0
        return valid, sanitized, errors

    def build_dlio_command(self, params, data_dir=None):
        """Build DLIO benchmark Hydra override command string.

        Args:
            params: dict from validate_dlio_params
            data_dir: directory for DLIO data files

        Returns:
            (datagen_command, training_command) tuple
            Both use the same Hydra overrides but differ in workflow flags.
        """
        out = data_dir or self.scratch_dir

        # Build common Hydra overrides
        overrides = (
            f"++workload.dataset.data_folder={out}"
            f" ++workload.dataset.record_length={params['record_length']}"
            f" ++workload.dataset.num_files_train={params['num_files_train']}"
            f" ++workload.dataset.num_samples_per_file={params['num_samples_per_file']}"
            f" ++workload.reader.batch_size={params['batch_size']}"
            f" ++workload.reader.read_threads={params['read_threads']}"
            f" ++workload.train.computation_time={params['computation_time']}"
            f" ++workload.train.epochs={params['epochs']}"
            f" ++workload.train.seed={params['seed']}"
            f" ++workload.dataset.format={params['format']}"
        )

        # Shuffle overrides
        if params.get("sample_shuffle", "off") != "off":
            overrides += f" ++workload.reader.sample_shuffle={params['sample_shuffle']}"
        if params.get("file_shuffle", "off") != "off":
            overrides += f" ++workload.reader.file_shuffle={params['file_shuffle']}"

        datagen_cmd = (
            f"{self.DLIO_BIN} workload=unet3d"
            f" ++workload.workflow.generate_data=True"
            f" ++workload.workflow.train=False"
            f" {overrides}"
        )

        training_cmd = (
            f"{self.DLIO_BIN} workload=unet3d"
            f" ++workload.workflow.generate_data=False"
            f" ++workload.workflow.train=True"
            f" {overrides}"
        )

        return datagen_cmd, training_cmd

    def parse_llm_config_changes(self, llm_response):
        """Extract IOR/mdtest parameter changes from LLM response.

        The LLM is asked to output a JSON with 'config_changes' dict.
        This method extracts and validates those changes.

        Args:
            llm_response: parsed JSON dict from LLM

        Returns:
            (params_dict, errors_list)
        """
        changes = llm_response.get("config_changes", {})
        if not changes:
            # Try to infer from 'changes_made' list
            changes_list = llm_response.get("changes_made", [])
            if changes_list:
                changes = self._infer_params_from_text(changes_list)

        return changes

    def _infer_params_from_text(self, changes_list):
        """Try to infer parameter changes from text descriptions.

        Handles cases like "increase transfer size to 1MB", "use MPI-IO collective".
        """
        params = {}
        for change in changes_list:
            change_lower = str(change).lower()

            # Transfer size
            size_match = re.search(
                r"transfer\s+size\s+(?:to\s+)?(\d+[kmg]?b?)", change_lower
            )
            if size_match:
                params["transfer_size"] = size_match.group(1).rstrip("b")

            # API change
            if "mpi-io" in change_lower or "mpiio" in change_lower:
                params["api"] = "MPIIO"
            if "collective" in change_lower:
                params["collective"] = True

            # Flags
            if "remove" in change_lower and "fsync" in change_lower:
                params["remove_Y"] = True
            if "remove" in change_lower and "random" in change_lower:
                params["remove_z"] = True
            if "sequential" in change_lower:
                params["remove_z"] = True
            if "o_direct" in change_lower or "direct" in change_lower:
                params["add_o_direct"] = True

            # HACC-IO inference
            if "mpiio" in change_lower and ("shared" in change_lower or "hacc" in change_lower):
                params["executable"] = "mpiio_shared"
            elif "posix" in change_lower and ("shared" in change_lower or "hacc" in change_lower):
                params["executable"] = "posix_shared"
            elif "file-per-process" in change_lower or "fpp" in change_lower:
                if "hacc" in change_lower:
                    params["executable"] = "fpp"

            particle_match = re.search(r"particles?\s+(?:to\s+)?(\d+)", change_lower)
            if particle_match:
                params["num_particles"] = int(particle_match.group(1))

            if "collective" in change_lower and "enable" in change_lower:
                params["collective_buffering"] = "enabled"
            elif "collective" in change_lower and "disable" in change_lower:
                params["collective_buffering"] = "disabled"

            # Custom inference
            imb_match = re.search(r"imbalance\s+(?:factor\s+)?(?:to\s+)?(\d+\.?\d*)", change_lower)
            if imb_match:
                params["imbalance_factor"] = float(imb_match.group(1))

            base_match = re.search(r"base[\s_-]size[\s_-]?(?:mb)?\s+(?:to\s+)?(\d+)", change_lower)
            if base_match:
                params["base_size_mb"] = int(base_match.group(1))

            # h5bench inference
            dim1_match = re.search(r"dim[_\s]?1\s+(?:to\s+)?(\d+)", change_lower)
            if dim1_match:
                params["DIM_1"] = int(dim1_match.group(1))

            if "collective_data" in change_lower and ("yes" in change_lower or "enable" in change_lower):
                params["COLLECTIVE_DATA"] = "YES"
            elif "collective_data" in change_lower and ("no" in change_lower or "disable" in change_lower):
                params["COLLECTIVE_DATA"] = "NO"

            if "interleaved" in change_lower and "mem" in change_lower:
                params["MEM_PATTERN"] = "INTERLEAVED"
            elif "contig" in change_lower and "mem" in change_lower:
                params["MEM_PATTERN"] = "CONTIG"

            if "interleaved" in change_lower and "file" in change_lower:
                params["FILE_PATTERN"] = "INTERLEAVED"
            elif "contig" in change_lower and "file" in change_lower:
                params["FILE_PATTERN"] = "CONTIG"

            timesteps_match = re.search(r"timesteps?\s+(?:to\s+)?(\d+)", change_lower)
            if timesteps_match:
                params["TIMESTEPS"] = int(timesteps_match.group(1))

            # DLIO inference
            rl_match = re.search(r"record[_\s]?length\s+(?:to\s+)?(\d+)", change_lower)
            if rl_match:
                params["record_length"] = int(rl_match.group(1))

            nf_match = re.search(r"num[_\s]?files[_\s]?train\s+(?:to\s+)?(\d+)", change_lower)
            if nf_match:
                params["num_files_train"] = int(nf_match.group(1))

            bs_match = re.search(r"batch[_\s]?size\s+(?:to\s+)?(\d+)", change_lower)
            if bs_match:
                params["batch_size"] = int(bs_match.group(1))

            rt_match = re.search(r"read[_\s]?threads?\s+(?:to\s+)?(\d+)", change_lower)
            if rt_match:
                params["read_threads"] = int(rt_match.group(1))

        return params

    def apply_changes_to_config(self, base_config, changes):
        """Apply LLM-proposed changes to a base benchmark config.

        Args:
            base_config: dict with current IOR/mdtest parameters
            changes: dict of proposed changes from LLM

        Returns:
            new_config dict
        """
        new_config = dict(base_config)

        # Direct parameter overrides (IOR)
        for key in ["api", "transfer_size", "block_size", "segments", "extra_flags"]:
            if key in changes:
                new_config[key] = changes[key]

        # Direct parameter overrides (mdtest)
        for key in ["items_per_rank", "write_bytes", "read_bytes"]:
            if key in changes:
                new_config[key] = changes[key]

        # Direct parameter overrides (HACC-IO)
        for key in ["executable", "num_particles", "collective_buffering"]:
            if key in changes:
                new_config[key] = changes[key]

        # Direct parameter overrides (custom load_imbalance)
        for key in ["imbalance_factor", "base_size_mb"]:
            if key in changes:
                new_config[key] = changes[key]

        # Direct parameter overrides (h5bench)
        for key in ["DIM_1", "COLLECTIVE_DATA", "COLLECTIVE_METADATA",
                     "TIMESTEPS", "MEM_PATTERN", "FILE_PATTERN"]:
            if key in changes:
                new_config[key] = changes[key]

        # Direct parameter overrides (DLIO)
        for key in ["record_length", "num_files_train", "num_samples_per_file",
                     "batch_size", "read_threads", "computation_time", "epochs",
                     "format", "sample_shuffle", "file_shuffle", "seed"]:
            if key in changes:
                new_config[key] = changes[key]

        # Boolean overrides
        if "file_per_proc" in changes:
            new_config["file_per_proc"] = changes["file_per_proc"]
        if "collective" in changes:
            if changes["collective"] and new_config.get("api") == "MPIIO":
                flags = new_config.get("extra_flags", "")
                if "-c" not in flags:
                    new_config["extra_flags"] = flags + " -c"
        if "unique_dir" in changes:
            new_config["unique_dir"] = changes["unique_dir"]
        if "files_only" in changes:
            new_config["files_only"] = changes["files_only"]

        # Flag removal
        if changes.get("remove_Y"):
            flags = new_config.get("extra_flags", "")
            new_config["extra_flags"] = flags.replace("-Y", "").strip()
        if changes.get("remove_z"):
            flags = new_config.get("extra_flags", "")
            new_config["extra_flags"] = flags.replace("-z", "").strip()

        # Flag addition
        if changes.get("add_o_direct"):
            flags = new_config.get("extra_flags", "")
            if "useO_DIRECT" not in flags:
                new_config["extra_flags"] = flags + " -O useO_DIRECT=1"

        return new_config
