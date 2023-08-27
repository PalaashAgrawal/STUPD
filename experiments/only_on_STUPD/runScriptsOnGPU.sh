#!/bin/bash

# If an argument is given, it is considered as the GPU ID.
# Otherwise, it defaults to 0.
GPU_ID=${1:-0}

for script in *.py; do
    # We use python to set the environment variable and run the script.
    CUDA_VISIBLE_DEVICE=$GPU_ID python ${script}
done