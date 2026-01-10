#!/bin/bash

COLLECTION_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd -P)"

# default to the COLLECTION_PATH if no argument is provided
log_dir="${1:-$COLLECTION_PATH}"

# argument checking
if [[ $# -gt 1 ]]; then
    echo "Error: Too many arguments provided."
    exit 1
fi

# make sure directory exists
if [[ ! -d "$log_dir" ]]; then
    echo "Error: $dir is not a valid directory."
    exit 1
fi

find "$log_dir" -name "logs.tar.gz" -print0 |\
	xargs -0 -P 16 -I{} sh -c \
	'echo "Extracting: {}"; cd "$(dirname "{}")" && tar -xzf logs.tar.gz && echo "Done: {}"'
