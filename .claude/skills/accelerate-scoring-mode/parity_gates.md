# Validation gates

Validate a new objective in layers. A later mismatch can originate in either
the reference or accelerated path; do not loosen tolerances to conceal it.
Do not treat optimized score agreement as a substitute for a fixed-pose check.

## 1. Reference mathematics

Verify values and autograd derivatives against direct formulas and finite
differences on distinct pairs and nontrivial poses. Check normalization,
weights, empty channels, asymmetry, and expected clamping. Use explicit atom
assignment checks for fields and retained-isotope-H inputs where applicable.

## 2. Kernel and reduction parity

Compare CPU and GPU value/gradient outputs against the same frozen inputs
and poses. Compare tangent-projected quaternion gradients when the reference
normalizes its quaternion. Exercise real-count masks, non-power-of-two sizes,
empty channels, and size-gated fused/separate paths.

For reused kernels, the new blend/reduction still needs a fixed-pose check
against the reference, including its derivative. Existing kernel tests establish
the reused primitive, not the new objective. Choose tolerances from observed
precision and the nearest current tests, not a historical decimal claim.

## 3. Optimizer execution

Use identical features and explicitly recorded effective seed counts, step
budgets, learning rates, and stopping settings. Compare eager GPU and graph
with matched seeds, then CPU and GPU. Confirm which path ran: graph/fused
failures can fall back silently. With early stopping disabled for comparison,
use a sufficiently large patience, not zero across all paths.

The reference and accelerated seed generators/optimizers differ, so they need
not converge to the identical pose. First establish fixed-pose correctness;
then assess recovery of planted poses and score differences on multiple inputs.
Re-score the returned transform independently. A unit-self-score test alone
does not validate the global search or a composite objective.

## 4. Batch and preparation invariants

Freeze conformers, surfaces, charges, pharmacophores, and descriptors. Check
scores and poses across repeated calls and a fresh process. Record dtype,
Numba/SVML, CUDA/Triton versions, and frozen tuning profile where used.
Changing batch membership can change early stopping; use fixed budgets when
isolating padding or upload behavior from stopping policy.

## 5. Store and worker paths

Follow [screen_wiring.md](screen_wiring.md). Verify that required arrays and
offsets reach disk, array/object paths agree under matched conditions, and
canonical transforms re-score correctly in the input frame. Test any new
third input through screening and worker dispatch, not just a single pair.

## Existing tests to consult

| Test | Evidence it targets |
|---|---|
| `tests/modes/test_mode_registry.py` | derived mode maps, public bindings, result slots |
| `tests/modes/test_esp_agreement_grad.py` | surface-potential agreement gradients |
| `tests/accel/test_graph_early_stop.py` | replay stopping schedule and cached patience |
| `tests/accel/test_cpu_fine_loops_agree.py` | fused/eager CPU comparisons |
| `tests/accel/test_trans_init_accel.py` | translation-initialized execution |
| `tests/screen/test_screen_arrays.py` | array/object screening and transforms |
| `tests/screen/test_screen_const_seeds.py` | canonical seed routing |
| `tests/accel/test_triton_tuning.py` | frozen launch configuration handling |

Run applicable tests on the intended backend. Mark every GPU test (or GPU
parameter, via `pytest.param(..., marks=pytest.mark.cuda)`) with `cuda`:
`tests/conftest.py` skips it without CUDA, and `pytest -m cuda` must select
every GPU test since CI has no GPU. Import Triton with `pytest.importorskip`.
Report what ran, what passed, tolerances, and pending checks. Benchmark only
after correctness, with untimed setup/warm-up and the user's requested timing
protocol; never reuse archived throughput as proof of new-mode correctness.
