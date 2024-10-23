#!/bin/bash

# More efficient with a job array, but it is not needed if you set slurm env variables to 1


LOAD_DIR="" # path to the generated samples: must contain "x1x3x4" if moses and "x1x{2-4}" if gdb
TRAINING_FILE="" # path to a training file from the shepeherd-gdb or shepherd-moses sets
echo "Loading and saving to $LOAD_DIR" # must contain "moses" or "gdb" in the file name
echo "My task ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

python -u ./consistency_evaluation.py --load-dir "$LOAD_DIR" --task-id "$SLURM_ARRAY_TASK_ID" --training-data "$TRAINING_FILE"
