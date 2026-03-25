#!/usr/bin/env python3
"""
Run IONavigator (ION) baseline on our benchmark Darshan traces.

This script faithfully reproduces ION's RAG-free pipeline:
  1. Parse .darshan -> per-module CSVs (using ION_Extractor's darshan_parser)
  2. Compute module summaries (using ION's POSIX/STDIO/MPIIO classes)
  3. Generate per-fragment LLM summaries (code interpretation + context)
  4. Generate per-fragment RAG-free diagnoses
  5. Intra-module merge (pairwise merge of fragments per module)
  6. Inter-module merge (pairwise merge across modules)
  7. Extract detected issues from final diagnosis text
  8. Evaluate against our ground-truth labels

Uses OpenAI API directly (bypassing litellm/llama-index for Python 3.9 compat).
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# OpenAI
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'external', 'IONavigator', 'ION'))

# We import ION's darshan module classes directly (they work on 3.9)
from ion.Steps.Utils.darshan_modules import (
    get_darshan_modules, extract_class_methods,
    process_trace_header, summarize_trace_header,
    POSIX, STDIO, MPIIO, DarshanModules
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ionavigator_baseline')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_LOGS = PROJECT_ROOT / 'data' / 'benchmark_logs'
LABELS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'benchmark' / 'labels.parquet'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'ionavigator_baseline'

# Our 7 bottleneck dimensions (excluding 'healthy')
LABEL_DIMS = [
    'access_granularity', 'metadata_intensity', 'parallelism_efficiency',
    'access_pattern', 'interface_choice', 'file_strategy',
    'throughput_utilization'
]

# Mapping from ION-style issue keywords to our dimensions
ION_TO_DIM_KEYWORDS = {
    'access_granularity': [
        'small read', 'small write', 'small i/o', 'small request',
        'misalign', 'unaligned', 'alignment', 'request size',
        'tiny', 'granularity', 'sub-optimal request',
        'small-sized', 'small io', 'fragmented',
        'size_read_0_100', 'size_write_0_100',
        'small data transfer', 'inefficient request size',
        'numerous smaller request', 'small and medium',
        'below 1mb', 'below 1 mb', 'under 1 mb',
        'request sizes are small', 'many small',
    ],
    'metadata_intensity': [
        'metadata', 'meta-data', 'open', 'stat', 'seek',
        'high metadata', 'excessive metadata', 'metadata overhead',
        'metadata time', 'metadata operation', 'metadata load',
        'file creation', 'directory', 'inode',
    ],
    'parallelism_efficiency': [
        'load imbalance', 'imbalance', 'single rank',
        'one rank', 'rank 0', 'stragglers', 'straggler',
        'uneven distribution', 'rank imbalance',
        'skewed', 'hotspot', 'server load',
        'all i/o operations', 'only one rank',
        'single process', 'single-rank',
        'load balancing', 'unbalanced',
    ],
    'access_pattern': [
        'random', 'non-sequential', 'non-contiguous',
        'random access', 'random read', 'random write',
        'scattered', 'non sequential', 'stride',
        'discontinuous', 'non-consecutive',
    ],
    'interface_choice': [
        'no collective', 'independent i/o', 'without mpi',
        'collective i/o', 'low-level', 'posix instead',
        'missing mpi-io', 'not using mpi', 'mpi-io absent',
        'direct posix', 'bypassing mpi', 'without collective',
        'independent write', 'independent read',
        'individual i/o', 'lack of collective',
        'multi-process without mpi',
        'not leveraging', 'could benefit from mpi',
    ],
    'file_strategy': [
        'shared file', 'file-per-process', 'single shared',
        'shared-file', 'all processes writing to same',
        'contention', 'lock contention', 'file sharing',
        'many files', 'too many files', 'excessive files',
        'single file', 'common file',
    ],
    'throughput_utilization': [
        'low throughput', 'bandwidth', 'underutiliz',
        'low bandwidth', 'poor throughput', 'slow i/o',
        'throughput', 'transfer rate', 'sustained',
        'peak performance', 'bottleneck',
        'write time', 'read time', 'i/o time',
    ],
}


# ---------------------------------------------------------------------------
# OpenAI wrapper
# ---------------------------------------------------------------------------
class OpenAIClient:
    """Thin wrapper around OpenAI API for ION-style completions."""

    def __init__(self, api_key, model='gpt-4.1-mini'):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_cost = 0.0

    async def complete(self, messages, json_mode=False):
        """Make a chat completion call."""
        kwargs = {}
        if json_mode:
            kwargs['response_format'] = {"type": "json_object"}

        t0 = time.time()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        elapsed = time.time() - t0

        content = resp.choices[0].message.content
        usage = resp.usage
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_calls += 1

        # Cost estimation for gpt-4.1-mini
        # Input: $0.40/1M tokens, Output: $1.60/1M tokens
        cost = (usage.prompt_tokens * 0.40 + usage.completion_tokens * 1.60) / 1_000_000
        self.total_cost += cost

        return content, elapsed

    def get_metrics(self):
        return {
            'total_api_calls': self.total_calls,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_cost_usd': round(self.total_cost, 4),
        }


# ---------------------------------------------------------------------------
# Step 1: Parse .darshan to per-module CSVs
# ---------------------------------------------------------------------------
def parse_darshan_to_csvs(darshan_path, output_dir):
    """Use darshan-parser CLI to extract module data, then parse to CSVs.
    Replicates ION_Extractor behavior."""

    os.makedirs(output_dir, exist_ok=True)

    # Run darshan-parser
    result = subprocess.run(
        ['darshan-parser', '--show-incomplete', darshan_path],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(f"darshan-parser failed for {darshan_path}: {result.stderr}")

    log_data = result.stdout

    # Extract header
    header = {}
    for line in log_data.splitlines():
        if line.startswith('# nprocs'):
            header['nprocs'] = int(line.split()[-1])
        elif line.startswith('# run time'):
            header['runtime'] = float(line.split()[-1])
        elif line.startswith('# start_time:'):
            header['start'] = float(line.split()[-1])
        elif line.startswith('# end_time:'):
            header['end'] = float(line.split()[-1])

    # Extract modules
    modules = {}
    module = None
    in_module = False
    column_names = None
    for line in log_data.splitlines():
        if line.startswith('#<module>'):
            column_names = re.findall(r'<(.*?)>', line)
            in_module = True
            module = None
        elif in_module:
            if not line.strip():
                in_module = False
                module = None
            else:
                fields = line.split()
                if not module:
                    module = fields[0]
                    modules[module] = {'columns': column_names, 'data': [fields]}
                else:
                    modules[module]['data'].append(fields)

    # Convert to DataFrames and save as CSV
    for mod_name, mod_data in modules.items():
        cols = mod_data['columns']
        rows = mod_data['data']
        df = pd.DataFrame(rows, columns=cols)
        index_cols = [c for c in cols if c not in ['counter', 'value']]
        try:
            df = df.pivot_table(
                index=index_cols, columns='counter', values='value',
                aggfunc='first'
            ).reset_index()
            df.to_csv(os.path.join(output_dir, f'{mod_name}.csv'), index=False)
        except Exception as e:
            logger.warning(f"Failed to pivot module {mod_name}: {e}")

    # Save header
    with open(os.path.join(output_dir, 'header.json'), 'w') as f:
        json.dump([header], f, indent=4)

    return output_dir


# ---------------------------------------------------------------------------
# Step 2: Compute module summaries using ION classes
# ---------------------------------------------------------------------------
def compute_module_summaries(trace_csv_dir):
    """Load CSVs and compute summaries using ION's module classes."""
    modules, header = get_darshan_modules(trace_csv_dir)
    if header is None:
        return None, None, None

    header = process_trace_header(header)

    module_size_summaries = {}
    module_summaries = {}

    for module_name in modules:
        if module_name not in DarshanModules:
            continue
        darshan_module_class = DarshanModules[module_name]
        try:
            if module_name == 'MPI-IO':
                instance = darshan_module_class(modules[module_name], header['nprocs'])
            else:
                instance = darshan_module_class(modules[module_name])

            methods = extract_class_methods(darshan_module_class)[1:-1]
            summary_info = instance.summarize()

            if module_name in ['STDIO', 'POSIX', 'MPI-IO']:
                module_size_summaries[module_name] = summary_info[0]

            module_summaries[module_name] = {
                'methods': methods,
                'summary_values': summary_info
            }
        except Exception as e:
            logger.warning(f"Failed to summarize module {module_name}: {e}")

    broad_context = None
    if module_size_summaries:
        try:
            broad_context = summarize_trace_header(header, module_size_summaries)
        except Exception as e:
            logger.warning(f"Failed to summarize trace header: {e}")

    return module_summaries, broad_context, header


# ---------------------------------------------------------------------------
# Step 3-6: ION RAG-free pipeline via OpenAI
# ---------------------------------------------------------------------------
DARSHAN_CONTEXT = (
    "Note that, for the rank data column, -1 indicates that the file is shared. "
    "For other data columns, -1 indicates that the data is not available and the value should be ignored. "
    "Also, note that requests from the MPI-IO module are translated to POSIX requests, "
    "but the broader context will already have removed any double counting of requests."
)


async def run_ion_pipeline_rag_free(client, module_summaries, broad_context):
    """Run ION's RAG-free pipeline: summarize -> diagnose -> merge."""

    if not module_summaries:
        return "No module data available for analysis."

    # Step 3: Generate per-fragment summaries and diagnoses
    module_diagnoses = {}

    for module_name, mod_data in module_summaries.items():
        methods = mod_data['methods']
        summary_values = mod_data['summary_values']
        fragment_diagnoses = []

        for idx, method in enumerate(methods):
            method_return_values = summary_values[idx]

            # 3a: Code interpretation
            code_interp_msgs = [
                {"role": "system", "content":
                    "You are an expert in code interpretation. You will be given snippets of code "
                    "and must describe what they are intended to do. Specifically the snippets of code "
                    "will be part of an analysis and data extraction process of HPC application trace logs "
                    "collected using Darshan I/O Profiler. Be sure to keep the descriptions to the point "
                    "and only contain relevant information while still maintaining a high level of accuracy and detail."},
                {"role": "user", "content":
                    f"Given the following code snippet, give a brief, 2-5 sentence interpretation of what the code is doing: \n\n{method}"}
            ]
            code_interpretation, _ = await client.complete(code_interp_msgs)

            # Build summary document (like ION does)
            summary_document = (
                f"In this analysis, the code was described as follows: {code_interpretation}. \n\n"
                f"This is the actual code used to generate this analysis: \n\n{method}. \n\n"
                f"The return value of this analysis is the following: \n\n{method_return_values}"
            )

            # 3b: Context-aware interpretation
            context_msgs = [
                {"role": "system", "content":
                    f"You will be given a code description, a code snippet, and the return value for the code. "
                    f"The snippets of code will be part of an analysis process of HPC application trace logs collected using Darshan. "
                    f"You will also be given some context of the application trace as a whole, beyond just the {module_name} "
                    f"information to provide context for the scale and scope of the application. "
                    f"While later analysis will investigate other modules in more detail, note that this "
                    f"part of the analysis only looks at one portion of the {module_name} Darshan module data extracted from the trace log. "
                    f"Your task is to give a precise and accurate interpretation of the output values in the context "
                    f"of what was analyzed and the larger view of the application without suggesting any improvements or "
                    f"potential problems. "
                    f"It is crucial that your judgments of scale are guided by overall application context provided. "
                    f"{DARSHAN_CONTEXT} "
                    f"Be sure to keep your responses concise and to the point."},
                {"role": "user", "content":
                    f"Given the following code description, code snippet, and code output analyzing a specific aspect of "
                    f"I/O trace data from just the {module_name} module, provide an interpretation of the output in the context "
                    f"of what was analyzed and the larger view of the application without suggesting any improvements or "
                    f"potential problems: \n\n "
                    f"Here is the analysis: \n{summary_document} \n\n "
                    f"Here is the broader context of the application trace: \n{broad_context}"}
            ]
            rag_document, _ = await client.complete(context_msgs)

            # Build full fragment text
            method_name = re.search(r'def (\w+)', method)
            method_name = method_name.group(1) if method_name else f"method_{idx}"
            fragment_text = (
                f"The {method_name} method in the {module_name} module resulted in the following calculated values: \n\n"
                f"{method_return_values}\n\n"
                f"The broader context of the application is as follows: \n\n{broad_context}\n\n"
                f"The resultant interpretation of the output values given the broader context of the application is as follows: \n\n{rag_document}"
            )

            # 3c: RAG-free diagnosis for this fragment
            diag_msgs = [
                {"role": "system", "content":
                    "You are an expert in HPC I/O performance diagnosis. "
                    "A user has run a process to extract key summary information from a Darshan trace log. "
                    "You will be given an analysis summary investigating an aspect of the trace log. "
                    "Your task is to: \n"
                    "1. Provide context regarding the I/O behavior of the application.\n"
                    "2. Determine whether the summary information indicates any I/O performance issues.\n"
                    "3. Ensure all information is useful to the diagnosis.\n"
                    "4. Do not recommend further analyses.\n\n"},
                {"role": "user", "content":
                    f"Here is the analysis summary of the trace log: \n{fragment_text}"}
            ]
            diagnosis, _ = await client.complete(diag_msgs)
            fragment_diagnoses.append(diagnosis)

        # Step 5: Intra-module merge (pairwise)
        if len(fragment_diagnoses) == 1:
            module_diagnoses[module_name] = fragment_diagnoses[0]
        else:
            merged = await pairwise_merge_intra(client, module_name, fragment_diagnoses)
            module_diagnoses[module_name] = merged

    # Step 6: Inter-module merge
    if len(module_diagnoses) == 1:
        final = list(module_diagnoses.values())[0]
    else:
        final = await pairwise_merge_inter(client, module_diagnoses)

    return final


async def pairwise_merge_intra(client, module_name, fragments):
    """Pairwise merge of fragment diagnoses within a module."""
    current = fragments
    while len(current) > 1:
        new_level = []
        for i in range(0, len(current) - 1, 2):
            merged = await merge_two_intra(client, module_name, current[i], current[i+1],
                                           final=(len(current) <= 3))
            new_level.append(merged)
        if len(current) % 2 == 1:
            new_level.append(current[-1])
        current = new_level

    # Final merge if 2 remain
    if len(current) == 2:
        return await merge_two_intra(client, module_name, current[0], current[1], final=True)
    return current[0]


async def merge_two_intra(client, module_name, frag1, frag2, final=False):
    """Merge two intra-module fragments."""
    task = "comprehensive analysis summary" if final else "new summary"
    msgs = [
        {"role": "system", "content":
            f"You are an expert in HPC I/O performance diagnosis. "
            f"I have conducted an analysis of various aspects of an HPC application I/O trace log collected using Darshan. "
            f"You will be given two analysis summaries investigating two different aspects of the trace log pertaining only to the {module_name} module. "
            f"Your task is to: \n"
            f"1. Create a {task} based on the two provided summaries.\n"
            f"2. Remove redundant information and resolve any contradictions between the analyses but keep all relevant information.\n"
            f"3. Highlight any potential I/O performance issues found in either analysis.\n"
            f"4. Do not forget any important information from the provided analysis summaries.\n\n"
            f"{DARSHAN_CONTEXT}"},
        {"role": "user", "content":
            f"Create a {task} of the {module_name} module based on the following two performance analysis summaries: \n"
            f"Analysis 1: \n{frag1} \n\n"
            f"Analysis 2: \n{frag2} \n\n"
            f"New merged analysis:"}
    ]
    result, _ = await client.complete(msgs)
    return result


async def pairwise_merge_inter(client, module_diagnoses):
    """Pairwise merge across modules."""
    remaining = list(module_diagnoses.items())

    while len(remaining) > 1:
        new_level = []
        for i in range(0, len(remaining) - 1, 2):
            name1, diag1 = remaining[i]
            name2, diag2 = remaining[i+1]
            combined_name = f"{name1}_{name2}"
            merged = await merge_two_inter(client, [name1, name2], diag1, diag2,
                                           final=(len(remaining) <= 3))
            new_level.append((combined_name, merged))
        if len(remaining) % 2 == 1:
            new_level.append(remaining[-1])
        remaining = new_level

    # Final merge if 2 remain
    if len(remaining) == 2:
        name1, diag1 = remaining[0]
        name2, diag2 = remaining[1]
        return await merge_two_inter(client, [name1, name2], diag1, diag2, final=True)
    return remaining[0][1]


async def merge_two_inter(client, module_names, diag1, diag2, final=False):
    """Merge two inter-module diagnoses."""
    all_mods = ", ".join(module_names)
    task_desc = "comprehensive final diagnosis" if final else "combined diagnosis"
    msgs = [
        {"role": "system", "content":
            f"You are an expert in HPC I/O performance diagnosis. "
            f"You will be given two analysis summaries investigating different aspects of the trace log pertaining to various Darshan modules. "
            f"Your task is to: \n"
            f"1. Create a {task_desc} based on the two provided summaries.\n"
            f"2. Remove redundant information and resolve any contradictions between the analyses but keep all relevant information.\n"
            f"3. Highlight any potential I/O performance issues found in the analysis summary.\n"
            f"4. Do not forget any important information from the provided analysis summaries.\n\n"
            f"{DARSHAN_CONTEXT}"},
        {"role": "user", "content":
            f"Create a diagnosis that combines the information from the following two summaries containing "
            f"information from these modules: {all_mods}.\n\n"
            f"Analysis 1:\n{diag1}\n\n"
            f"Analysis 2:\n{diag2}\n\n"
            f"New merged analysis:"}
    ]
    result, _ = await client.complete(msgs)
    return result


# ---------------------------------------------------------------------------
# Step 7: Extract detections from ION text output
# ---------------------------------------------------------------------------
def extract_detections_keyword(diagnosis_text):
    """Keyword-based extraction (fallback). Tends to over-detect."""
    text_lower = diagnosis_text.lower()
    detections = {}
    for dim, keywords in ION_TO_DIM_KEYWORDS.items():
        detected = False
        for kw in keywords:
            if kw in text_lower:
                detected = True
                break
        detections[dim] = int(detected)
    return detections


async def extract_detections_llm(client, diagnosis_text):
    """Use LLM to classify which dimensions are flagged as issues.
    More accurate than keyword matching, and faithful to TraceBench's
    LLM-as-judge evaluation approach."""
    msgs = [
        {"role": "system", "content":
            "You are an expert evaluator of HPC I/O performance diagnoses. "
            "Given a diagnosis report from an I/O analysis tool, you must determine which "
            "of the following I/O performance issues are EXPLICITLY IDENTIFIED AS PROBLEMS "
            "in the diagnosis. Only mark an issue as detected if the diagnosis clearly states "
            "it is a problem or bottleneck -- do NOT mark it if the diagnosis merely mentions "
            "the concept in passing or says it is NOT an issue.\n\n"
            "The 7 issue dimensions are:\n"
            "1. access_granularity: Small or misaligned I/O requests (sub-optimal request sizes)\n"
            "2. metadata_intensity: Excessive metadata operations causing overhead\n"
            "3. parallelism_efficiency: Load imbalance, single-rank I/O, poor parallel distribution\n"
            "4. access_pattern: Random or non-sequential access patterns\n"
            "5. interface_choice: Using POSIX instead of MPI-IO, lack of collective I/O\n"
            "6. file_strategy: Shared file contention or excessive file-per-process overhead\n"
            "7. throughput_utilization: Low I/O throughput relative to system capability\n\n"
            "Respond with a JSON object with exactly these 7 keys, each mapping to 0 or 1.\n"
            "1 = the diagnosis identifies this as a performance issue/bottleneck.\n"
            "0 = the diagnosis does NOT identify this as a problem (or says it's fine)."},
        {"role": "user", "content":
            f"Based on the following I/O performance diagnosis, classify which issues are "
            f"identified as problems:\n\n{diagnosis_text}"}
    ]
    response, _ = await client.complete(msgs, json_mode=True)
    try:
        detections = json.loads(response)
        # Ensure all keys present with int values
        result = {}
        for dim in LABEL_DIMS:
            result[dim] = int(detections.get(dim, 0))
        return result
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"LLM extraction failed, falling back to keywords: {e}")
        return extract_detections_keyword(diagnosis_text)


# ---------------------------------------------------------------------------
# Step 8: Evaluation metrics
# ---------------------------------------------------------------------------
def compute_metrics(results_df, label_dims):
    """Compute per-dimension and overall metrics."""
    per_dim = {}
    total_tp = total_fp = total_fn = total_tn = 0

    for dim in label_dims:
        gt_col = f'gt_{dim}'
        pred_col = f'pred_{dim}'
        if gt_col not in results_df.columns or pred_col not in results_df.columns:
            continue
        gt = results_df[gt_col].values
        pred = results_df[pred_col].values
        tp = int(((gt == 1) & (pred == 1)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        tn = int(((gt == 0) & (pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_dim[dim] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0)

    f1_values = [v['f1'] for v in per_dim.values() if (v['tp'] + v['fn']) > 0]
    macro_f1 = np.mean(f1_values) if f1_values else 0

    return {
        'per_dimension': per_dim,
        'overall': {
            'micro_precision': round(micro_precision, 4),
            'micro_recall': round(micro_recall, 4),
            'micro_f1': round(micro_f1, 4),
            'macro_f1': round(float(macro_f1), 4),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_tn': total_tn,
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def select_traces(labels_df, n_per_bench=10, seed=42):
    """Select representative traces: n_per_bench from each benchmark."""
    rng = np.random.RandomState(seed)
    selected = []
    for bench in ['ior', 'mdtest', 'dlio', 'h5bench', 'hacc_io']:
        sub = labels_df[labels_df['benchmark'] == bench]
        if len(sub) == 0:
            continue
        n = min(n_per_bench, len(sub))
        chosen = sub.sample(n=n, random_state=rng)
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def find_darshan_file(job_id, benchmark):
    """Find darshan file for a given job_id and benchmark."""
    bench_dir = BENCHMARK_LOGS / benchmark
    if not bench_dir.exists():
        return None
    pattern = str(bench_dir / f'*_id{job_id}-*.darshan')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Try broader search
    pattern = str(bench_dir / f'*{job_id}*.darshan')
    matches = glob.glob(pattern)
    return matches[0] if matches else None


async def process_single_trace(client, darshan_path, job_id, temp_dir):
    """Process one trace through the ION pipeline."""
    trace_name = Path(darshan_path).stem
    csv_dir = os.path.join(temp_dir, trace_name)

    t0 = time.time()

    # Step 1: Parse
    try:
        parse_darshan_to_csvs(darshan_path, csv_dir)
    except Exception as e:
        logger.error(f"Failed to parse {darshan_path}: {e}")
        return None, time.time() - t0

    # Step 2: Compute summaries
    module_summaries, broad_context, header = compute_module_summaries(csv_dir)
    if not module_summaries:
        logger.warning(f"No module summaries for {job_id}")
        return None, time.time() - t0

    # Steps 3-6: ION pipeline
    try:
        final_diagnosis = await run_ion_pipeline_rag_free(
            client, module_summaries, broad_context
        )
    except Exception as e:
        logger.error(f"ION pipeline failed for {job_id}: {e}")
        return None, time.time() - t0

    elapsed = time.time() - t0

    # Step 7: Extract detections (LLM-based, like TraceBench)
    detections = await extract_detections_llm(client, final_diagnosis)

    return {
        'job_id': job_id,
        'trace_name': trace_name,
        'diagnosis': final_diagnosis,
        'detections': detections,
        'elapsed_seconds': round(elapsed, 2),
        'api_calls_so_far': client.total_calls,
    }, elapsed


async def main():
    parser = argparse.ArgumentParser(description='Run IONavigator baseline evaluation')
    parser.add_argument('--n-per-bench', type=int, default=10,
                        help='Number of traces per benchmark (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trace selection')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini',
                        help='OpenAI model to use')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / 'temp_csvs'
    temp_dir.mkdir(exist_ok=True)

    # Load labels
    labels_df = pd.read_parquet(LABELS_PATH)
    logger.info(f"Loaded {len(labels_df)} labels")

    # Select traces
    selected = select_traces(labels_df, n_per_bench=args.n_per_bench, seed=args.seed)
    logger.info(f"Selected {len(selected)} traces for evaluation")

    # Find darshan files
    trace_list = []
    for _, row in selected.iterrows():
        darshan_path = find_darshan_file(row['job_id'], row['benchmark'])
        if darshan_path:
            trace_list.append({
                'job_id': row['job_id'],
                'benchmark': row['benchmark'],
                'darshan_path': darshan_path,
                'labels': {dim: int(row[dim]) for dim in LABEL_DIMS}
            })
        else:
            logger.warning(f"No darshan file found for job {row['job_id']} ({row['benchmark']})")

    logger.info(f"Found darshan files for {len(trace_list)}/{len(selected)} traces")

    # Initialize OpenAI client
    client = OpenAIClient(api_key=api_key, model=args.model)

    # Process traces sequentially (to respect rate limits and track costs)
    all_results = []
    for i, trace in enumerate(trace_list):
        logger.info(f"Processing trace {i+1}/{len(trace_list)}: {trace['job_id']} ({trace['benchmark']})")

        result, elapsed = await process_single_trace(
            client, trace['darshan_path'], trace['job_id'], str(temp_dir)
        )

        if result is None:
            logger.warning(f"Skipping trace {trace['job_id']} due to error")
            continue

        # Add ground truth
        result['benchmark'] = trace['benchmark']
        result['ground_truth'] = trace['labels']
        all_results.append(result)

        logger.info(
            f"  Completed in {elapsed:.1f}s, "
            f"API calls so far: {client.total_calls}, "
            f"Cost so far: ${client.total_cost:.4f}"
        )

        # Save intermediate results after each trace
        with open(output_dir / 'raw_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Build results DataFrame for evaluation
    rows = []
    for r in all_results:
        row = {'job_id': r['job_id'], 'benchmark': r['benchmark']}
        for dim in LABEL_DIMS:
            row[f'gt_{dim}'] = r['ground_truth'].get(dim, 0)
            row[f'pred_{dim}'] = r['detections'].get(dim, 0)
        row['elapsed_seconds'] = r['elapsed_seconds']
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Compute metrics
    metrics = compute_metrics(results_df, LABEL_DIMS)
    metrics['api_metrics'] = client.get_metrics()
    metrics['n_traces_processed'] = len(all_results)
    metrics['n_traces_attempted'] = len(trace_list)
    metrics['model'] = args.model
    metrics['avg_time_per_trace'] = round(
        results_df['elapsed_seconds'].mean(), 2) if len(results_df) > 0 else 0
    metrics['avg_api_calls_per_trace'] = round(
        client.total_calls / max(len(all_results), 1), 1)

    # Save all outputs
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    results_df.to_csv(output_dir / 'predictions.csv', index=False)

    with open(output_dir / 'raw_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("IONavigator (ION) Baseline Evaluation Results")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Traces processed: {len(all_results)}/{len(trace_list)}")
    print(f"Avg time per trace: {metrics['avg_time_per_trace']}s")
    print(f"Avg API calls per trace: {metrics['avg_api_calls_per_trace']}")
    print(f"Total API cost: ${client.total_cost:.4f}")
    print(f"\nOverall Metrics:")
    print(f"  Micro-F1:     {metrics['overall']['micro_f1']:.4f}")
    print(f"  Macro-F1:     {metrics['overall']['macro_f1']:.4f}")
    print(f"  Micro-Prec:   {metrics['overall']['micro_precision']:.4f}")
    print(f"  Micro-Recall: {metrics['overall']['micro_recall']:.4f}")
    print(f"\nPer-Dimension F1:")
    for dim, vals in metrics['per_dimension'].items():
        print(f"  {dim:30s}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f} "
              f"(TP={vals['tp']} FP={vals['fp']} FN={vals['fn']} TN={vals['tn']})")
    print(f"\nCompare with our ML system: Micro-F1 = 0.923")
    print("=" * 70)

    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    asyncio.run(main())
