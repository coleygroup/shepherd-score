import pickle
import os
import time

import numpy as np

from rdkit import Chem

import shepherd_score.conformer_generation
from shepherd_score.conformer_generation import optimize_conformer_with_xtb


from collections.abc import Callable, Iterable
from typing import Literal, Optional

import tqdm
from joblib import delayed, Parallel, parallel_config


# https://gist.github.com/tsvikas/5f859a484e53d4ef93400751d0a116de
class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: Optional[int] = None,
        desc: Optional[str] = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.progress_bar is not None:
            if self.total_tasks is None and self._original_iterator is None:
                self.total_tasks = self.n_dispatched_tasks
                self.progress_bar.total = self.total_tasks
                self.progress_bar.refresh()
            # update progressbar
            self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def joblib_map(
    func: Callable,
    iterable: Iterable,
    n_jobs: int = 1,
    inner_max_num_threads: Optional[int] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    backend: Literal["sequential", "loky", "threading", "multiprocessing"] = "loky",
) -> list:
    if backend != "loky" and inner_max_num_threads is not None:
        print(f"{backend=} does not support {inner_max_num_threads=}, setting to None.")
        inner_max_num_threads = None

    if backend != "sequential":
        with parallel_config(
            backend=backend, inner_max_num_threads=inner_max_num_threads
        ):
            if backend == "loky":
                results = ParallelTqdm(n_jobs=n_jobs, total_tasks=total, desc=desc)(
                    delayed(func)(i) for i in iterable
                )
            else:
                results = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in iterable)
    else:
        results = [func(i) for i in tqdm.tqdm(iterable, desc=desc, total=total)]
    return results


if __name__ == "__main__":
    print(f'File: {shepherd_score.conformer_generation.__file__}')

    # get number of cores
    num_cores = os.cpu_count()
    print(f'Number of cores: {num_cores}')

    print('Loading data...')
    with open('/nfs/ccoleylab001/kento/conformers/mcp/5000_steps_96cores_100/aimnet2_default_relaxed.pkl', 'rb') as f:
        mb_charges = pickle.load(f)

    print('Converting to RDKit molecules...')
    mols = [Chem.MolFromMolBlock(mb, removeHs=False) for i, (mb, _) in enumerate(mb_charges[:30]) if i != 11]

    # benchmarking config
    # n_jobs = 22
    repeats = 0
    tmp_dir = os.environ.get('TMPDIR', '/tmp')
    num_tasks = len(mols)
    print(f'Number of tasks: {num_tasks}')

    n_jobs = np.min([24, num_tasks])
    n_jobs = 10
    print(f'Number of jobs: {n_jobs}')

    # will hold warmup timings for later comparison
    warmup_seq_time = None
    warmup_par_time = None

    def run_sequential_once():
        energies = []
        start = time.perf_counter()
        for i in tqdm.tqdm(range(num_tasks), desc='Sequential', total=num_tasks):
            _, energy, _ = optimize_conformer_with_xtb(
                mols[i], 'water', 4, charge=0, temp_dir=tmp_dir
            )
            energies.append(float(energy))
        elapsed = time.perf_counter() - start
        return elapsed, energies

    def run_sequential_with_breakdown_once():
        """Run sequentially and record per-molecule durations."""
        energies = []
        per_molecule_durations = []
        start_total = time.perf_counter()
        for i in tqdm.tqdm(range(num_tasks), desc='Sequential (breakdown)', total=num_tasks):
            start_i = time.perf_counter()
            _, energy, _ = optimize_conformer_with_xtb(
                mols[i], 'water', 4, charge=0, temp_dir=tmp_dir
            )
            per_molecule_durations.append(time.perf_counter() - start_i)
            energies.append(float(energy))
        elapsed_total = time.perf_counter() - start_total
        return elapsed_total, energies, per_molecule_durations

    def run_parallel_once():
        start = time.perf_counter()
        results = ParallelTqdm(
            n_jobs=n_jobs,
            total_tasks=num_tasks,
            desc='Parallel',
        )(
            delayed(optimize_conformer_with_xtb)(
                mols[i], 'water', 4, charge=0, temp_dir=tmp_dir
            )
            for i in range(num_tasks)
        )
        elapsed = time.perf_counter() - start
        energies = [float(energy) for _, energy, _ in results]
        return elapsed, energies

    # warmup (timed once)
    print('Warming up...')
    try:
        # parallel warmup timing
        time_par, energies_par = run_parallel_once()
        warmup_par_time = time_par

        # sequential with per-molecule breakdown
        time_seq, energies_seq, per_mol_times = run_sequential_with_breakdown_once()
        warmup_seq_time = time_seq

        print(f'Sequential: {time_seq:.2f}s')
        print(f'Parallel: {time_par:.2f}s')
        print(f'Speedup: {time_seq / time_par:.2f}x')
        print(f'Efficiency: {(time_seq / time_par / n_jobs) * 100:.1f}%')

        # Energy comparison
        seq_arr = np.asarray(energies_seq, dtype=float)
        par_arr = np.asarray(energies_par, dtype=float)
        if seq_arr.shape != par_arr.shape:
            print(f'Energy check: shape mismatch {seq_arr.shape} vs {par_arr.shape}')
        else:
            diffs = np.abs(seq_arr - par_arr)
            allclose = np.allclose(seq_arr, par_arr, rtol=1e-6, atol=1e-8)
            max_abs_diff = float(diffs.max()) if diffs.size else 0.0
            print(f'Energy check: allclose={allclose}, max |ΔE|={max_abs_diff:.3e}')

        # Per-molecule sequential times and slowest vs parallel (using warmup parallel)
        if per_mol_times:
            print('Per-molecule sequential times:')
            for i, t in enumerate(per_mol_times):
                try:
                    smi_i = Chem.MolToSmiles(Chem.RemoveHs(mols[i]))
                except Exception:
                    smi_i = 'N/A'
                print(f'  idx={i}, time={t:.2f}s, SMILES={smi_i}')

            slowest_idx = int(np.argmax(per_mol_times))
            slowest_time = float(per_mol_times[slowest_idx])
            try:
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mols[slowest_idx]))
            except Exception:
                smiles = 'N/A'
            print(f'Slowest molecule: idx={slowest_idx}, SMILES={smiles}')
            print(f'Slowest molecule sequential time: {slowest_time:.2f}s')
            print(f'Parallel wall time after warmup (n_jobs={n_jobs}): {warmup_par_time:.2f}s')
            if slowest_time:
                ratio = warmup_par_time / slowest_time
                print(f'Parallel time vs slowest single-molecule time: {ratio:.2f}x')
    except Exception as exc:
        print(f'Warmup encountered an error: {exc}')

    # timed repeats (optional)
    if repeats and repeats > 0:
        print('Timing...')
        seq_times = [run_sequential_once()[0] for _ in range(repeats)]
        print(f'Sequential: {seq_times}')
        par_times = [run_parallel_once()[0] for _ in range(repeats)]
        print(f'Parallel: {par_times}')

        seq_mean = float(np.mean(seq_times)) if seq_times else float('nan')
        par_mean = float(np.mean(par_times)) if par_times else float('nan')
        seq_std = float(np.std(seq_times, ddof=1)) if len(seq_times) > 1 else 0.0
        par_std = float(np.std(par_times, ddof=1)) if len(par_times) > 1 else 0.0

        speedup = seq_mean / par_mean if par_mean and par_mean > 0 else float('nan')
        efficiency = (speedup / n_jobs) * 100 if n_jobs else float('nan')

        print(f'Sequential: {seq_mean:.2f}s ± {seq_std:.2f}s (n={repeats})')
        print(f'Parallel (n_jobs={n_jobs}): {par_mean:.2f}s ± {par_std:.2f}s (n={repeats})')
        print(f'Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1f}%')
    else:
        print('Skipping repeat timing; using warmup measurements for comparisons.')
        seq_mean = warmup_seq_time if warmup_seq_time is not None else float('nan')
        par_mean = warmup_par_time if warmup_par_time is not None else float('nan')
