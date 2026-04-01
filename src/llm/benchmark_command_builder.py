"""
Validate and build benchmark commands from LLM-proposed parameter changes.

Safety layer: ensures LLM output maps to valid, safe IOR/mdtest commands.
Uses allowlists from configs/iterative.yaml to constrain parameter ranges.
Prevents execution of harmful or nonsensical commands.

References:
  - STELLAR (SC'25): Constrains LLM to known Lustre parameters
  - PerfCodeGen (2025): Validation phase before execution
"""

import logging
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
