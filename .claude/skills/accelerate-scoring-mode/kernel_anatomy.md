# Kernels, optimizer, and memory

## Value and gradient contract

The dispatched kernels return overlap/agreement, `dO/dq`, and `dO/dt` for
padded inputs at a specified unit quaternion and translation. Mask padding
using real counts, not an incidental feature-type sentinel. Match CPU/GPU
signatures and verify value-only and gradient-bearing calls separately.

Gradients emitted in unit-quaternion coordinates can contain a radial
component. Compare tangent-projected gradients with a reference that normalizes
its quaternion. Positions and pharmacophore directions both contribute to
rotation derivatives; directions do not contribute a translation derivative.

`drivers/terms.py` evaluates terms and self-overlaps. `engine.py` applies the
reduction derivative and weight and accumulates the score and descent gradient.
A penalty has a negative score weight. Do not normalize an already reduced
score or omit a term's gradient from a composite objective.

`esp_cmp` now calls `esp_agreement_grad_se3_batch`: the combined ESP objective
uses both shape and electrostatic gradients at every evaluation. Its exclusion
masks make it piecewise smooth; finite differences should avoid mask crossings.
Use `tests/modes/test_esp_agreement_grad.py` to locate the current reference and tests.

## Dispatch and fused execution

`kernels/dispatch.py` chooses CPU or Triton from the input tensor device and
imports GPU modules lazily. The fused small-pad shape/color kernel is a separate
CUDA optimization selected through `ModeSpec.fused_pair`; other shapes use
the separate terms. Test both sides of such size gates if your change touches them.

For a new family, update `terms.evaluate`, any needed `terms.self_overlap`,
and `cpu_fused._term_closure`. If fused CPU execution is not supported, select
that deliberately in the spec rather than relying on an exception fallback.
Inspect the path actually executed: `engine.align` can fall back after a graph
or fused-loop exception, so successful output alone does not prove acceleration.

## Fine-loop schedule

`engine.align` assembles inputs and seeds, then tries CUDA graph, fused CPU,
and eager paths under the spec's gates. The loop tracks the best evaluated
pose before advancing Adam. Its custom Adam and reference torch optimizer
are not interchangeable merely because their learning rates match.

The convergence check uses each pair's best score over seeds. The sub-batch
stops after the specified consecutive checks at which no pair improves beyond
the tolerance. Checks are five evaluations apart after the initial baseline.
Graph patience has no added margin. Combined ESP specs set `graph_full_steps=True`.
Zero patience disables stopping only on the graph path; for a fixed-step
cross-path comparison use a patience larger than the total number of checks.

The graph cache retains buffers until eviction/reset; `reset_graph_cache`
does not clear Triton tuning. `accel/_stats.py` records effort when enabled,
but its counters are process-local. An empty record does not prove a path ran.

## Seeds and coordinate frames

Reference and accelerated seed sets differ. `_common.batched_seeds_torch`
uses principal-axis alignments and structured rotations as well as Fibonacci
seeds. Canonical stores can use `_common.canonical_seed_quats` where the
spec's seed channel allows it. Translation initialization adds another search
path; test it explicitly if the public method supports it.

Hoist pose-invariant self-overlaps per bucket using `engine.term_self_overlaps`
and slice them consistently for chunks. Recomputing them per chunk changes
overhead; reusing them across different feature arrays changes the score.

## Memory and reproducibility

`_batch_upload` shares cold tensors by molecule object identity, but tensors
already populated by pair construction bypass that upload. Do not claim all
pair inputs are deduplicated. Cached inputs must be treated as read-only.

`_pad._subbatched_align` measures GPU allocation growth, estimates a chunk
from available memory, respects an optional pose cap, and halves chunks on
OOM. CPU chunks use a fixed pair count. This limits fine-loop workspace, not
the memory needed to construct all pair objects, upload inputs, or retain
results. Large requested comparisons may need caller-created blocks.

Different chunks may stop at different iterations. Padding preserves the
mathematical score at a fixed pose, but changing buckets/chunks does not
guarantee identical optimized scores under early stopping.

Use `kernels.tuning.autotune`, not a direct Triton decorator, for new kernels.
Warm representative shapes, export with `export_configurations`, and validate
replay in a new process with `FSS_TRITON_CONFIGS` set before import. A profile
freezes launch choices, not wall time or optimality. Record hardware, versions,
inputs, profile hash, paths taken, and timing boundaries for comparisons.

CPU precision and speed depend on the kernel path. The fused loop's fp32 kernels
(`kernels/cpu_soa.py`) evaluate exp2 with a polynomial and do not depend on SVML;
the fp64 kernels in `kernels/cpu.py` (eager loop) vectorize `exp` only when
`numba.core.config.USING_SVML` is true. A new fused term needs a `cpu_soa.py` twin
checked against its `cpu.py` reference (see `tests/accel/test_cpu_soa_kernels.py`). Inspect
the actual kernel implementation; do not
borrow old machine-specific tolerances or speedups as universal guarantees.
