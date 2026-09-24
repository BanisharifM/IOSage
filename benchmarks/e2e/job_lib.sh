#!/bin/bash

e2e_build()
{
    make -C "$E2E_DIR"
}

e2e_run_and_record()
{
    local config_id=$1
    local fix_id=$2
    local binary=$3
    local output_prefix=$4
    local manifest=$5
    shift 5
    local executable
    local status=0
    local candidate
    local known
    local is_new
    local -a before=()
    local -a after=()
    local -a new_logs=()

    executable=$(basename "$binary")
    mapfile -t before < <(compgen -G "$DARSHAN_LOGPATH/*${executable}*id${SLURM_JOB_ID}*.darshan" || true)
    srun --cpu-bind=none --export="ALL,LD_PRELOAD=$DARSHAN_LIB" \
        "$binary" "$output_prefix" "$@" || status=$?
    if (( status != 0 )); then
        printf 'E2E run failed with exit code %d: %s\n' "$status" "$executable" >&2
        return "$status"
    fi
    if [[ ! -s "${output_prefix}.nc4" ]]; then
        printf 'E2E run did not create a nonempty output: %s.nc4\n' "$output_prefix" >&2
        return 90
    fi

    mapfile -t after < <(compgen -G "$DARSHAN_LOGPATH/*${executable}*id${SLURM_JOB_ID}*.darshan" || true)
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
    if (( ${#new_logs[@]} != 1 )); then
        printf 'Expected one new Darshan log for %s and job %s, found %d\n' \
            "$executable" "$SLURM_JOB_ID" "${#new_logs[@]}" >&2
        return 91
    fi
    if [[ ! -e "$manifest" ]]; then
        printf 'config_id\tfix_id\tjob_id\texecutable\toutput\tdarshan_log\tcompleted_utc\n' > "$manifest"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$config_id" "$fix_id" "$SLURM_JOB_ID" "$binary" \
        "${output_prefix}.nc4" "${new_logs[0]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$manifest"
}

e2e_verify_pair()
{
    local before_config=$1
    local after_config=$2
    local before_output=$3
    local after_output=$4
    local manifest=$5
    local status=0

    command -v h5diff >/dev/null 2>&1 || {
        echo "h5diff is required for E2E correctness validation" >&2
        return 93
    }
    h5diff --quiet "$before_output" "$after_output" || status=$?
    if [[ ! -e "$manifest" ]]; then
        printf 'before_config\tafter_config\tjob_id\tbefore_output\tafter_output\tchecker\tpassed\tcompleted_utc\n' > "$manifest"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$before_config" "$after_config" "$SLURM_JOB_ID" \
        "$before_output" "$after_output" "h5diff --quiet" \
        "$([[ $status -eq 0 ]] && echo true || echo false)" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$manifest"
    if (( status != 0 )); then
        printf 'E2E output comparison failed with h5diff status %d\n' "$status" >&2
        return "$status"
    fi
}
