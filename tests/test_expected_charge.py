"""Regression tests for expected-charge-first molecule reconstruction."""

import numpy as np
import pytest

from shepherd_score.evaluations.evaluate import pipelines, _pipeline_eval_single
from shepherd_score.evaluations.utils import convert_data


def test_expected_charge_is_tried_first_then_legacy_fallback(monkeypatch):
    attempted = []
    sentinel = object()

    def fake_extract(*, xyz_block, charge):
        attempted.append(charge)
        return sentinel if charge == -1 else None

    monkeypatch.setattr(convert_data, "extract_mol_from_xyz_block", fake_extract)
    mol, charge, _ = convert_data.get_mol_from_atom_pos(
        np.array([6]), np.zeros((1, 3)), expected_charge=2
    )

    assert mol is sentinel
    assert charge == -1
    assert attempted == [2, 0, 1, -1]


def test_expected_charge_deduplicates_legacy_sequence(monkeypatch):
    attempted = []

    def fake_extract(*, xyz_block, charge):
        attempted.append(charge)
        return None

    monkeypatch.setattr(convert_data, "extract_mol_from_xyz_block", fake_extract)
    convert_data.get_mol_from_atom_pos(
        np.array([6]), np.zeros((1, 3)), expected_charge=1
    )

    assert attempted == [1, 0, -1, 2, -2]


def test_default_charge_order_is_unchanged(monkeypatch):
    attempted = []

    def fake_extract(*, xyz_block, charge):
        attempted.append(charge)
        return None

    monkeypatch.setattr(convert_data, "extract_mol_from_xyz_block", fake_extract)
    convert_data.get_mol_from_atom_pos(np.array([6]), np.zeros((1, 3)))

    assert attempted == [0, 1, -1, 2, -2]


def test_pipeline_propagates_expected_charge_to_worker(monkeypatch):
    captured = {}

    def fake_dispatch(**kwargs):
        captured["inputs"] = kwargs["inputs"]

    monkeypatch.setattr(pipelines, "_dispatch_eval", fake_dispatch)
    pipeline = pipelines.ConditionalEvalPipeline.__new__(pipelines.ConditionalEvalPipeline)
    pipeline.generated_mols = [(np.array([6]), np.zeros((1, 3)))]
    pipeline.num_generated_mols = 1
    pipeline.ref_molec = object()
    pipeline.condition = "all"
    pipeline.num_surf_points = 400
    pipeline.pharm_multi_vector = False
    pipeline.solvent = None
    pipeline.priority_pharm_indices = None
    pipeline.expected_charge = 2
    pipeline.num_valid = pipeline.num_valid_post_opt = pipeline.num_consistent_graph = 0
    pipeline.graph_similarities = np.array([np.nan])
    pipeline.smiles = pipeline.smiles_post_opt = [None]

    pipeline.evaluate()

    assert captured["inputs"][0][-1] == 2


def test_consistency_pipeline_propagates_expected_charge_to_worker(monkeypatch):
    captured = {}

    def fake_dispatch(**kwargs):
        captured["inputs"] = kwargs["inputs"]

    monkeypatch.setattr(pipelines, "_dispatch_eval", fake_dispatch)
    pipeline = pipelines.ConsistencyEvalPipeline(
        generated_mols=[(np.array([6]), np.zeros((1, 3)))], expected_charge=2
    )

    assert pipeline.expected_charge == 2
    pipeline.evaluate()

    assert captured["inputs"][0][-1] == 2


def test_eval_consistency_single_passes_expected_charge(monkeypatch):
    captured = {}

    class FakeConsistencyEval:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capture")

    monkeypatch.setattr(_pipeline_eval_single, "ConsistencyEval", FakeConsistencyEval)
    res = _pipeline_eval_single._eval_consistency_single(
        0, np.array([6]), np.zeros((1, 3)), None, None, None, None, 1.2, None, 1,
        expected_charge=2,
    )

    assert captured["expected_charge"] == 2
    assert res["error"] is not None


def _record_attempts(monkeypatch):
    attempted = []

    def fake_extract(*, xyz_block, charge):
        attempted.append(charge)
        return None

    monkeypatch.setattr(convert_data, "extract_mol_from_xyz_block", fake_extract)
    return attempted


def test_expected_charge_list_replaces_default_order(monkeypatch):
    attempted = _record_attempts(monkeypatch)
    convert_data.get_mol_from_atom_pos(
        np.array([6]), np.zeros((1, 3)), expected_charge=[3, -3, 0]
    )

    assert attempted == [3, -3, 0]


@pytest.mark.parametrize(
    "expected_charge, fallback",
    [(None, 0), (2, 2), ([3, -3], 3)],
)
def test_failed_reconstruction_falls_back_to_first_charge_tried(monkeypatch, expected_charge, fallback):
    _record_attempts(monkeypatch)
    mol, charge, _ = convert_data.get_mol_from_atom_pos(
        np.array([6]), np.zeros((1, 3)), expected_charge=expected_charge
    )

    assert mol is None
    assert charge == fallback


@pytest.mark.parametrize(
    "value, expected",
    [(np.int64(3), 3), (np.array(3), 3), (3.0, 3), (np.array([4, 4, -4]), [4, -4])],
)
def test_normalize_expected_charge(value, expected):
    assert convert_data.normalize_expected_charge(value) == expected


@pytest.mark.parametrize("bad", [[], 1.5, [1, 2.5]])
def test_normalize_expected_charge_rejects_invalid(bad):
    with pytest.raises(ValueError):
        convert_data.normalize_expected_charge(bad)


def test_pipeline_normalizes_expected_charge_at_init():
    mols = [(np.array([6]), np.zeros((1, 3)))]
    pipeline = pipelines.ConsistencyEvalPipeline(generated_mols=mols, expected_charge=(3, 3, -3))
    assert pipeline.expected_charge == [3, -3]

    with pytest.raises(ValueError):
        pipelines.ConsistencyEvalPipeline(generated_mols=mols, expected_charge=[1, 2.5])
