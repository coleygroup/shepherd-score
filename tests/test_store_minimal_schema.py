"""Every screenable mode must work from a store built for that mode alone.

A shared store can hide a mode whose schema flags do not cover everything ``_flush`` writes,
because another mode in the list supplies the missing arrays. One store per mode cannot.
"""
import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")

N_SURF = 32
SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
          "CC(=O)Oc1ccccc1C(=O)O"]

# keyword a mode demands of screen(); everything else takes registry defaults
KW = {"surf": {"alpha": 0.81}, "surf_esp": {"alpha": 0.81},
      "surf_tversky": {"alpha": 0.81}, "surf_esp_tversky": {"alpha": 0.81},
      "vol_and_surf_esp": {"alpha": 0.81}, "vol_and_surf_esp_tversky": {"alpha": 0.81},
      "vol_esp": {"lam": 0.3}}


def _mol(smiles, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        assert AllChem.EmbedMolecule(rd, p) == 0, smiles
    rng = np.random.default_rng(seed)
    n = rd.GetNumAtoms()
    return Molecule(rd, pharm_multi_vector=False,
                    surface_points=(rng.standard_normal((N_SURF, 3)) * 3.0).astype(np.float32),
                    electrostatics=rng.standard_normal(N_SURF).astype(np.float32),
                    fukui=(rng.standard_normal(n) * 0.1).astype(np.float32),
                    partial_charges=(rng.standard_normal(n) * 0.2).astype(np.float32))


def _screenable():
    from shepherd_score import screen as S
    from shepherd_score.accel._modes import CANONICAL_MODES
    out = []
    for m in CANONICAL_MODES:
        try:
            if S._store_supports(S._schema_from_modes([m]), m):
                out.append(m)
        except Exception:                                           # noqa: BLE001
            pass
    return out


@pytest.mark.parametrize("mode", _screenable())
def test_mode_screens_from_a_store_built_for_itself_alone(mode):
    """A store created with ``modes=[mode]`` must actually serve that mode."""
    from shepherd_score.accel._modes import uses_avoid
    from shepherd_score.screen import ProfileStore, screen

    mols = [_mol(s, i) for i, s in enumerate(SMILES)]
    tmp = tempfile.mkdtemp(prefix="fss_minstore_")
    try:
        path = os.path.join(tmp, "s")
        os.makedirs(path, exist_ok=True)
        st = ProfileStore.create(path, num_surf_points=N_SURF, modes=[mode])
        for i, m in enumerate(mols):
            st.add(m, id="m%d" % i)
        st.close()

        st = ProfileStore.open(path)
        assert st.supports(mode), (
            f"a store created with modes=[{mode!r}] reports it cannot serve {mode}")

        kw = dict(KW.get(mode, {}))
        if uses_avoid(mode):
            # the modes with a third, non-molecule input: a cloud in the QUERY's frame
            kw["avoid_points"] = (np.asarray(mols[0].atom_pos, dtype=np.float32) + 40.0)
        hits = screen(mols[0], st, mode=mode, top_k=len(mols), **kw)
        assert len(hits) == len(mols)
        vals = []
        for h in hits:
            a, b = h[0], h[1]
            vals.append(float(b) if isinstance(a, str) else float(a))
        assert all(np.isfinite(vals)), f"{mode}: non-finite scores {vals}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_flag_a_mode_declares_is_one_the_writer_knows():
    """Every schema flag a channel declares must be one the store writer defines."""
    from shepherd_score import screen as S
    from shepherd_score.accel._modes import CANONICAL_MODES

    known = set(S._SCHEMA_FLAGS)
    for m in CANONICAL_MODES:
        unknown = S._mode_flags(m) - known
        assert not unknown, f"{m} declares schema flags the store does not define: {sorted(unknown)}"


def test_combo_store_carries_the_with_H_block():
    """A combo-only store carries the with-H block ``_flush`` writes under ``schema["charges"]``."""
    from shepherd_score.screen import ProfileStore

    mols = [_mol(s, i) for i, s in enumerate(SMILES)]
    tmp = tempfile.mkdtemp(prefix="fss_combostore_")
    try:
        path = os.path.join(tmp, "s")
        os.makedirs(path, exist_ok=True)
        st = ProfileStore.create(path, num_surf_points=N_SURF, modes=["vol_and_surf_esp"])
        for i, m in enumerate(mols):
            st.add(m, id="m%d" % i)
        st.close()

        written = {f.split("__")[-1].replace(".npy", "")
                   for f in os.listdir(path) if f.endswith(".npy")}
        for key in ("cwh", "radii", "all_off", "nonH", "charges"):
            assert key in written, (
                f"a combo-only store is missing {key!r}; it has {sorted(written)}. "
                "_flush writes the with-H block under schema['charges'].")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
