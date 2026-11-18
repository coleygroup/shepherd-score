"""Early-stop fixture: a converged self-pair must not halt the other pairs in its bucket.

A helper, not a collected test; run it directly with ``python tests/_es_fixture.py``. Per mode
it prints the full-budget reference, the default behaviour and a control with the self-pair
dropped, on both the fused numba and the eager torch CPU fine loops.
"""
from __future__ import annotations

import contextlib
import os
import sys
import warnings

# Run-as-a-script bootstrap: ``python tests/_es_fixture.py`` puts tests/ on sys.path, not the
# repo root, so the package would not import. Harmless when pytest imports this instead.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.accel import _stats
from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
from shepherd_score.container import Molecule, MoleculePair


def _stats_calls():
    return _stats.summary().get("steps_mean", 0) * _stats.summary().get("calls", 0)


def _stats_steps():
    """Total value+grad evaluations the recorder has seen (0 when recording is off)."""
    s = _stats.summary()
    return int(s.get("steps_mean", 0) * s.get("calls", 0)) if s else 0


MODES = ("vol", "vol_color", "vol_lipo")

# Heavy-atom counts (9-14) are close enough that all five pairs share one bucket; entry 0 is
# the reference itself, so pair 0 is a self-overlap that scores 1.0.
SMILES = (
    "c1ccccc1CCO",                    # 2-phenylethanol   (ref, and the self-pair fit)
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "c1ccc(cc1)S(=O)(=O)N",           # benzenesulfonamide
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",   # caffeine
    "OCC1OC(O)C(O)C(O)C1O",           # glucopyranose
)
EMBED_SEED = 0xC0FFEE

# Per-mode result attribute and batched-aligner seam on MoleculePair.
_SCORE_ATTR = {"vol": "sim_aligned_vol_noH",
               "vol_color": "sim_aligned_vol_color",
               "vol_lipo": "sim_aligned_vol_lipo"}
_ALIGNER = {"vol": "_align_batch_vol",
            "vol_color": "_align_batch_vol_color",
            "vol_lipo": "_align_batch_vol_lipo"}


def _embed(smiles: str) -> Chem.Mol:
    """ETKDGv3 embed with a fixed seed -> identical coordinates on every run."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    assert AllChem.EmbedMolecule(mol, params) == 0, f"embed failed: {smiles}"
    return mol


def _molecule(smiles: str) -> Molecule:
    """A Molecule with directionless pharmacophores (for vol_color) and no surface (no open3d)."""
    return Molecule(_embed(smiles), pharm_multi_vector=False, feature_set="rdkit_base")


def build_pairs() -> list[MoleculePair]:
    """Five pairs sharing one reference. Pair 0 is the self-pair that pins the global max."""
    cpu = torch.device("cpu")
    return [MoleculePair(_molecule(SMILES[0]), _molecule(s), do_center=True, device=cpu)
            for s in SMILES]


# Patience large enough that the early-stop branch can never fire (reference variant only).
_NO_EARLY_STOP = 1 << 30


@contextlib.contextmanager
def _instrumented(force_eager: bool, disable_early_stop: bool = False):
    """Yield per-fine-loop ``[path, n_evals, n_terms]`` records; optionally force eager or disable early stop."""
    from shepherd_score.accel.drivers import engine, terms
    from shepherd_score.accel.kernels import cpu_fused

    records: list[list] = []
    saved: list[tuple] = []

    def patch(module, name, factory):
        original = getattr(module, name)
        saved.append((module, name, original))
        setattr(module, name, factory(original))

    def open_record(original):
        """Wrapper that starts a new record for each fine-loop invocation."""
        def wrapper(*args, **kwargs):
            records.append(["-", 0, 1])
            if disable_early_stop:
                kwargs["early_stop_patience"] = _NO_EARLY_STOP
            return original(*args, **kwargs)
        return wrapper

    def count_eager(original):
        """Wrapper on the value+grad evaluator: one call per gradient term per iteration."""
        def wrapper(*args, **kwargs):
            records[-1][1] += 1
            return original(*args, **kwargs)
        return wrapper

    def count_fused(original):
        """Wrapper on run_fused: the fused loop reports its own iteration count."""
        def wrapper(pr, steps, lr, es_patience, es_tol, *a, **k):
            slot = records[-1]
            slot[0] = "fused"
            before = _stats_calls()
            out = original(pr, steps, lr, es_patience, es_tol, *a, **k)
            slot[1] += _stats_steps() - before
            return out
        return wrapper

    def count_eager_loop(original):
        def wrapper(pr, *a, **k):
            records[-1][0] = "eager"
            records[-1][2] = len([t for t in pr.terms if t.spec.grad])
            return original(pr, *a, **k)
        return wrapper

    patch(engine, "align", open_record)
    patch(engine, "_eager", count_eager_loop)
    patch(terms, "evaluate", count_eager)
    patch(cpu_fused, "run_fused", count_fused)

    if force_eager:
        def refuse(_original):
            def wrapper(*args, **kwargs):
                raise RuntimeError(
                    "_es_fixture: fused CPU path disabled to force the eager loop")
            return wrapper
        patch(cpu_fused, "run_fused", refuse)

    try:
        yield records
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


def run_mode(mode: str, force_eager: bool, disable_early_stop: bool = False,
             drop_self_pair: bool = False):
    """Align one bucket in ``mode`` (``drop_self_pair``: the control without pair 0); return (scores, records)."""
    pairs = build_pairs()
    if drop_self_pair:
        pairs = pairs[1:]
    _stats.reset()                       # the fused loop reports through the step recorder
    aligner = getattr(MoleculePair, _ALIGNER[mode])
    with _instrumented(force_eager, disable_early_stop) as records:
        aligner(pairs, steps_fine=MODE_STEPS[mode])
    scores = [float(getattr(p, _SCORE_ATTR[mode])) for p in pairs]
    # the eager counter fires once per GRADIENT TERM per iteration; divide it back out
    return scores, [(r[0], r[1] // max(1, r[2]) if r[0] == "eager" else r[1]) for r in records]


# (label, force_eager, disable_early_stop). The reference runs first so the two BASELINE
# variants can be printed with their deficit against it.
VARIANTS = (
    ("REFERENCE full budget, default route", False, True),
    ("REFERENCE full budget, eager",         True,  True),
    ("BASELINE default CPU route",           False, False),
    ("BASELINE eager (fused forced off)",    True,  False),
)


def main() -> None:
    warnings.filterwarnings("ignore")           # the one-shot SVML notice is env, not signal
    import numba.core.config as nbcfg

    print("=" * 78)
    print("EARLY-STOP BASELINE FIXTURE  (tests/_es_fixture.py)")
    print("=" * 78)
    print(f"torch {torch.__version__}   rdkit {Chem.rdBase.rdkitVersion}   "
          f"numba SVML={bool(nbcfg.USING_SVML)}")
    print(f"device cpu   pairs 5 (pair 0 = self-overlap)   embed seed 0x{EMBED_SEED:X}")
    print("step counts are VALUE+GRAD EVALUATIONS per fine-loop invocation")
    print("REFERENCE variants raise early_stop_patience only; seeds/steps are untouched")
    print()

    reference: dict[str, list[float]] = {}
    for mode in MODES:
        seeds, steps = MODE_SEEDS[mode], MODE_STEPS[mode]
        for label, force_eager, no_es in VARIANTS:
            scores, records = run_mode(mode, force_eager, no_es)
            print(f"--- mode={mode}  seeds={seeds}  steps_configured={steps}  [{label}]")
            for i, (path, n) in enumerate(records):
                print(f"    fine-loop[{i}] path={path:<5s} steps_executed={n:3d} of {steps}")
            if no_es and not force_eager:
                reference[mode] = scores            # the full-budget reference
            for i, score in enumerate(scores):
                tag = "self " if i == 0 else "cross"
                line = (f"    pair[{i}] {tag} {SMILES[0]:<24s} vs {SMILES[i]:<30s} "
                        f"score={score:.10f}")
                if not no_es:
                    line += f"   deficit={score - reference[mode][i]:+.10f}"
                print(line)
            print()

        # --- attribution control: same cross-pairs, self-pair dropped, nothing else changed
        scores, records = run_mode(mode, force_eager=False, drop_self_pair=True)
        print(f"--- mode={mode}  seeds={seeds}  steps_configured={steps}  "
              f"[CONTROL self-pair dropped, early stop untouched]")
        for i, (path, n) in enumerate(records):
            print(f"    fine-loop[{i}] path={path:<5s} steps_executed={n:3d} of {steps}")
        for i, score in enumerate(scores):
            ref = reference[mode][i + 1]
            print(f"    pair[{i + 1}] cross {SMILES[0]:<24s} vs {SMILES[i + 1]:<30s} "
                  f"score={score:.10f}   vs_reference={score - ref:+.10f}")
        print()

    print("=" * 78)
    print("END")
    print("=" * 78)


if __name__ == "__main__":
    main()
