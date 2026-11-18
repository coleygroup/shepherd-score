# Screening support for a new mode

`screen.py` derives mode support and array dispatch from `SPECS` and
`CHANNELS`. Reusing existing channels normally requires no new per-mode
screening branch. A new molecular feature requires more than a spec.

## New stored features

1. Add `Channel` entries in `accel/channels.py`, including readers, tensor
   stems, coordinate kind, basis, storage keys, schema flags, and padding.
2. Add a basis/offset table when needed. Strict-heavy atomic fields must not
   use RemoveHs-based atom offsets: isotope-labelled hydrogens can survive
   RemoveHs. Field positions and values must share the same indices.
3. Update `screen.py::_BASIS_TABLE` and `MoleculeProfile` slots, initialization,
   accessors, and centring behavior for the new feature. Read `center_to` and
   `_profile_from_schema`; verify that positions translate/rotate and vectors
   rotate only. Handle absent flags in older schemas explicitly.
4. Verify shard writes and reads through `ProfileStore.read_shard`, which
   supports both NPY and NPZ layouts. Do not assume a particular filename glob.

Canonical stores retain transforms into principal frames. Test the returned
hit transform by applying it in the original input frame and rescoring the
result. Comparing only the stored scalar can miss incorrect frame composition.

## Third inputs

`vol_avoid` carries its cloud through `screen(..., avoid_points=...)` and
`_with_avoid`, with a pair-level channel rather than a per-library field.
Follow this route for query-side constants. Verify both direct pair calls
and screening; the direct method requires its own `avoid_points` argument.

## Execution paths to verify

- Array-native screening for a pre-centered store and supported fast backend.
- Object-fast comparison using `_arrays.ENABLED=False` as a temporary test
  seam, restoring it afterwards.
- Object fallback where supported, such as translation initialization.
- Canonical and noncanonical stores, with their actual seed choices recorded.
- Multi-GPU worker dispatch for a new channel or required argument, if supported.

Useful tests are `tests/screen/test_screen_arrays.py`, `tests/screen/test_screen.py`,
`tests/screen/test_screen_const_seeds.py`, `tests/screen/test_screen_ndev_dispatch.py`,
`tests/screen/test_screen_pipeline.py`, and `tests/screen/test_screen_worker_threads.py`.
Inspect fixtures rather than assuming they generate arbitrary new features.
Compare under matched seeds and stopping settings; different coordinate
frames or chunk boundaries can change the optimization trajectory.

For scaling measurements, fix host compute threads per GPU worker explicitly
with `worker_threads`, freeze launch configurations, and state cache residency
and read-ahead conditions. A device-count change alone does not isolate GPU scaling.
