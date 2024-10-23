#!/bin/bash

# More efficient with a job array, but it is not needed if you set slurm env variables to 1

SAMPLE_IDX='0' # CHANGE ME
LOAD_DIR="" # Path to directory containing generated samples files
echo "Loading and saving to $LOAD_DIR and running idx $SAMPLE_IDX"
echo "My task ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

python -u ./docking_evaluation.py --load-dir-path "$LOAD_DIR" --sample-idx "$SAMPLE_IDX" --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$SLURM_ARRAY_TASK_COUNT"
