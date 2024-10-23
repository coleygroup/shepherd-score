#!/bin/bash

# More efficient with a job array, but it is not needed if you set slurm env variables to 1
# Our tests were run on 1 node with 12 tasks and 4 cpus/task (Intel Xeon Platinum 8260 -> 4GB RAM/cpu)

SAMPLE_ID='0' # CHANGE ME
LOAD_FILE="" # path to generated samples
echo "Loading and saving to $LOAD_FILE"
echo "Running idx $SAMPLE_ID"
echo "My task ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

python -u ./conditional_evaluation.py --load-file-path "$LOAD_FILE" --sample-id "$SAMPLE_ID" --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$SLURM_ARRAY_TASK_COUNT"
