#!/bin/bash


SAVE_DIR="" # Directory to save to
DATA_NAME="gdb" # gdb, moses_aq

NUM_PROCESSES=12

# TRAINING_SET_DIR="./conformers/${DATA_NAME}"

if [[ "$DATA_NAME" == "gdb" ]]; then
    RELEVANT_SETS="0,1,2" # gdb: "0,1,2"
    SOLVENT=""
else
    RELEVANT_SETS="0,1,2,3,4" # moses_aq: "0,1,2,3,4"
    SOLVENT="water"
fi

echo "Saving to $SAVE_DIR"

python -u ./consistency_benchmark.py --solvent "$SOLVENT" --num-processes "$NUM_PROCESSES" --data-name "$DATA_NAME" --save_dir "$SAVE_DIR" --relevant-sets "$RELEVANT_SETS" #--training-set-dir "$TRAINING_SET_DIR"