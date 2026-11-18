# Accelerated implementation map

Paths are under `shepherd_score/` unless stated otherwise.

| Concern | Source |
|---|---|
| Terms, modes, default budgets | `accel/_modes.py` |
| Feature readers, tensor attributes, bases, schema flags | `accel/channels.py` |
| Generated pairwise aligners and uploads | `accel/batch/aligners.py` |
| Array-native screening | `accel/batch/_arrays.py` |
| Size buckets and adaptive sub-batches | `accel/batch/_bucket.py`, `accel/batch/_pad.py` |
| Worker tensor specifications | `accel/batch/_dispatch.py` |
| Shared optimizer | `accel/drivers/engine.py` |
| Term evaluation, self-overlaps, lookup tables | `accel/drivers/terms.py` |
| CUDA capture/replay and graph cache | `accel/drivers/_graphed.py` |
| Seeds, quaternions, coarse translations | `accel/drivers/_common.py` |
| Kernel device dispatch | `accel/kernels/dispatch.py` |
| CPU wrappers and fused optimizer | `accel/kernels/cpu.py`, `accel/kernels/cpu_fused.py` |
| GPU kernels | `accel/kernels/*_triton.py` |
| Frozen launch configuration replay | `accel/kernels/tuning.py` |
| Store/profile/screen front end | `screen.py` |
| In-memory CPU screening workers | `accel/screen_parallel.py` |
| Public batch methods, `_align_mode`, per-pair fallback | `container/_batch.py` |
| Reference modes the fallback runs | `modes.py` |

`CANONICAL_MODES`, `MODE_ATTRS`, `MODE_SEEDS`, `MODE_STEPS`, `PROCESS_MODES`,
and `CONST_SEED_MODES` derive from `SPECS`. Preserve those derivations rather
than adding parallel routing lists. Some special cases and test fixtures still
need inspection when extending the registry; search consumers of the nearest mode.

Read these neighboring specs for common extensions:

- `vol` / `surf`: one shape term on different coordinate channels.
- `vol_esp` / `surf_esp`: one scalar-field term, different width conventions.
- `vol_lipo`: shape plus scalar field.
- `vol_color` / `vol_pharm`: shape plus directionless/directional pharmacophores.
- `vol_atomtype`: shape plus element-labelled overlap.
- `vol_and_surf_esp`: shape plus differentiated surface-potential agreement.
- `vol_avoid`: negative-weight raw penalty and a query-side third input.

## Execution controls

`FSS_TRITON_CONFIGS` selects a frozen launch profile before kernel import.
Without it, ordinary cached Triton autotuning remains enabled. GPU model,
capability, Triton version, kernel identity, and key coverage are checked on replay.
New math requires fresh numerical validation even if an old profile still loads.

Graph work budgets/cache limits live in `_graphed.py`; CPU chunk size and
optional pose cap live in `_pad.py`; per-mode gates live in `ModeSpec`.
There is no extra graph patience margin and no periodic ESP-score stride in
the current combined-ESP optimizer. Read the implementation before proposing
changes to any of these controls.
