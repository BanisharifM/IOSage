#!/bin/bash

benchmark_run()
{
    local run_name=$1
    local executable_name=$2
    local manifest=$3
    shift 3
    local status
    local candidate
    local known
    local is_new
    local command_text
    local expected_logs=${BENCHMARK_EXPECTED_LOGS:-1}
    local joined_logs
    local -a before=()
    local -a after=()
    local -a new_logs=()

    : "${SLURM_JOB_ID:?benchmark_run requires a SLURM allocation}"
    : "${DARSHAN_LOGPATH:?benchmark_run requires DARSHAN_LOGPATH}"
    [[ $expected_logs =~ ^[1-9][0-9]*$ ]] || {
        printf 'invalid BENCHMARK_EXPECTED_LOGS: %s\n' "$expected_logs" >&2
        return 93
    }
    mapfile -t before < <(compgen -G "$DARSHAN_LOGPATH/*${executable_name}*id${SLURM_JOB_ID}*.darshan" || true)
    if "$@"; then
        status=0
    else
        status=$?
    fi
    if (( status != 0 )); then
        printf '%s failed with exit code %d\n' "$run_name" "$status" >&2
        return "$status"
    fi

    mapfile -t after < <(compgen -G "$DARSHAN_LOGPATH/*${executable_name}*id${SLURM_JOB_ID}*.darshan" || true)
    for candidate in "${after[@]}"; do
        is_new=1
        for known in "${before[@]}"; do
            if [[ "$candidate" == "$known" ]]; then
                is_new=0
                break
            fi
        done
        if (( is_new )); then
            new_logs+=("$candidate")
        fi
    done
    if (( ${#new_logs[@]} != expected_logs )); then
        printf '%s expected %d new job-owned Darshan logs, found %d\n' \
            "$run_name" "$expected_logs" "${#new_logs[@]}" >&2
        return 91
    fi
    printf -v command_text '%q ' "$@"
    joined_logs=$(IFS=,; echo "${new_logs[*]}")
    printf '%s\t%s\t%s\t%s\t%s\n' "$SLURM_JOB_ID" "$run_name" "$executable_name" \
        "$joined_logs" "$command_text" \
        >> "$manifest"
}

benchmark_record_executable()
{
    local executable=$1
    local manifest=$2
    local resolved

    resolved=$(command -v "$executable") || {
        printf 'required executable not found: %s\n' "$executable" >&2
        return 92
    }
    printf '# executable\t%s\t%s\n' "$executable" "$(readlink -f "$resolved")" >> "$manifest"
}
