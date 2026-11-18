# Evaluation scripts

These contain evaluation scripts for ShEPhERD. For evaluations of ShEPhERD-score, please visit [Shepherd-Score-Benchmarks](https://github.com/Nate-Somers/Shepherd-Score-Benchmarks).

Install the package with `python -m pip install -e .`. These scripts evaluate generated ShEPhERD samples.
Run commands from the repository root. Shell wrappers accept the same arguments.

```bash
python scripts/consistency_evaluation.py --load-dir /data/gdb_samples --training-data /data/training.pkl --task-id 0
python scripts/conditional_evaluation.py --task NP --load-file-path /data/samples.pkl --sample-id 0
python scripts/conditional_evaluation.py --task GDB --load-file-path /data/gdb_samples --sample-id 0
python scripts/consistency_benchmark.py --training-set-dir /data/training --save_dir outputs/consistency --relevant-sets '1,2,3' --data-name gdb
python scripts/docking_screen.py --load-file /data/compounds.smi --save-dir-path outputs/docking
python scripts/docking_evaluation.py --load-dir-path /data/docking_samples --sample-idx 0
```

Consistency evaluation expects the original sample directory layout (`x1x2`,
`x1x3`, `x1x4` for GDB or `x1x3x4` for MOSES); its task IDs 0–2 select the GDB
representation. Conditional evaluation accepts `NP`, `frag`, or `GDB`.
Conditional and docking jobs default to one task (`--task-id 0 --num-tasks 1`);
set both arguments to partition a run. Outputs are written to the sample
directory unless a save directory is accepted explicitly.

xTB must be on `PATH` for the quantum-chemical evaluations. Docking also needs
the `docking` extra, which installs the Vina Python API. Use each script's `--help` for its
arguments. Pickled input must come from a trusted source.
