#!/bin/bash

# More efficient with a job array, but it is not needed if you set slurm env variables to 1
# Our tests were run on 7 nodes with 12 tasks and 4 cpus/task (Intel Xeon Platinum 8260 -> 4GB RAM/cpu)

SAVE_DIR="" # directory to save results to
echo "Saving to $SAVE_DIR"

echo "My task ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

python -u ./docking_benchmark.py --save-dir-path "$SAVE_DIR" --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$SLURM_ARRAY_TASK_COUNT"