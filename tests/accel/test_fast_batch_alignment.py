import math

import pytest
import torch


def _require_fast_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


@pytest.mark.cuda
def test_fast_esp_batch_matches_single():
    """vol_esp on one unpadded pair matches the same pair zero-padded with real counts."""
    _require_fast_cuda()

    from .._support.engine import batch, run

    torch.manual_seed(0)
    device = torch.device("cuda")

    N, M = 32, 27
    ref = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.5
    fit = torch.randn(M, 3, device=device, dtype=torch.float32) * 0.5
    ref_c = torch.randn(N, device=device, dtype=torch.float32) * 0.1
    fit_c = torch.randn(M, device=device, dtype=torch.float32) * 0.1

    kw = dict(alpha=0.81, lam=0.01, topk=3, steps_fine=5, lr=0.075, early_stop_patience=5)

    def _score(A, B, CA, CB, n_real=None, m_real=None):
        b = batch(A, B, n_real, m_real)
        chans = {"heavy": b, "charges": batch(CA, CB, b.n_real, b.m_real)}
        return run("vol_esp", chans, **kw)[0]

    score_single = _score(ref[None], fit[None], ref_c[None], fit_c[None])

    pad = lambda x, n: torch.cat([x, x.new_zeros((n,) + x.shape[1:])])[None]
    N_real = torch.tensor([N], device=device, dtype=torch.int32)
    M_real = torch.tensor([M], device=device, dtype=torch.int32)
    score_batch = _score(pad(ref, 3), pad(fit, 5), pad(ref_c, 3), pad(fit_c, 5), N_real, M_real)

    s1 = float(score_single[0])
    s2 = float(score_batch[0])
    assert math.isfinite(s1) and math.isfinite(s2)
    assert abs(s1 - s2) < 1e-4


def _combo_scores(cwH1, cwH2, c1, c2, pts1, pts2, p1, p2, pc1, pc2, r1, r2, alpha, *, lam,
                  probe_radius, esp_weight, N_real_atoms_w_H_1, M_real_atoms_w_H_2,
                  N_real_centers, M_real_centers, N_real_surf_1, M_real_surf_2, topk,
                  steps_fine, lr):
    """``vol_and_surf_esp`` scores for padded combo inputs, through the engine."""
    from .._support.engine import combo_chans, run
    chans = combo_chans(cwH1, cwH2, c1, c2, pts1, pts2, p1, p2, pc1, pc2, r1, r2,
                        N_real_centers, M_real_centers, N_real_atoms_w_H_1, M_real_atoms_w_H_2,
                        N_real_surf_1, M_real_surf_2)
    return run("vol_and_surf_esp", chans, alpha=alpha, lam=lam, probe_radius=probe_radius,
               esp_weight=esp_weight, topk=topk, steps_fine=steps_fine, lr=lr,
               early_stop_patience=5)[0]


@pytest.mark.cuda
def test_fast_esp_combo_padding_masks_stable():
    _require_fast_cuda()

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Construct two pairs with different atom counts so padding is required.
    K = 2
    n_wH = [9, 2]
    m_wH = [8, 3]
    n_surf = [40, 40]
    m_surf = [41, 41]

    n_wH_pad = max(n_wH)
    m_wH_pad = max(m_wH)
    n_surf_pad = max(n_surf)
    m_surf_pad = max(m_surf)

    centers_w_H_1 = torch.zeros(K, n_wH_pad, 3, device=device, dtype=torch.float32)
    centers_w_H_2 = torch.zeros(K, m_wH_pad, 3, device=device, dtype=torch.float32)
    partial_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
    partial_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)
    radii_1 = torch.ones(K, n_wH_pad, device=device, dtype=torch.float32) * 1.5
    radii_2 = torch.ones(K, m_wH_pad, device=device, dtype=torch.float32) * 1.5

    # Shape centers for alpha==0.81 use volumetric atoms.
    centers_1 = torch.zeros(K, n_wH_pad, 3, device=device, dtype=torch.float32)
    centers_2 = torch.zeros(K, m_wH_pad, 3, device=device, dtype=torch.float32)

    # Surface points: choose points near origin so that unmasked padded atoms at origin would matter.
    points_1 = torch.zeros(K, n_surf_pad, 3, device=device, dtype=torch.float32)
    points_2 = torch.zeros(K, m_surf_pad, 3, device=device, dtype=torch.float32)
    point_charges_1 = torch.zeros(K, n_surf_pad, device=device, dtype=torch.float32)
    point_charges_2 = torch.zeros(K, m_surf_pad, device=device, dtype=torch.float32)

    for i in range(K):
        centers_w_H_1[i, : n_wH[i]] = torch.randn(n_wH[i], 3, device=device) * 0.2
        centers_w_H_2[i, : m_wH[i]] = torch.randn(m_wH[i], 3, device=device) * 0.2
        centers_1[i, : n_wH[i]] = centers_w_H_1[i, : n_wH[i]]
        centers_2[i, : m_wH[i]] = centers_w_H_2[i, : m_wH[i]]

        points_1[i, : n_surf[i]] = torch.randn(n_surf[i], 3, device=device) * 0.3
        points_2[i, : m_surf[i]] = torch.randn(m_surf[i], 3, device=device) * 0.3
        point_charges_1[i, : n_surf[i]] = torch.randn(n_surf[i], device=device) * 0.05
        point_charges_2[i, : m_surf[i]] = torch.randn(m_surf[i], device=device) * 0.05

    N_real_atoms_w_H_1 = torch.tensor(n_wH, device=device, dtype=torch.int32)
    M_real_atoms_w_H_2 = torch.tensor(m_wH, device=device, dtype=torch.int32)
    N_real_centers = N_real_atoms_w_H_1.clone()
    M_real_centers = M_real_atoms_w_H_2.clone()
    N_real_surf_1 = torch.tensor(n_surf, device=device, dtype=torch.int32)
    M_real_surf_2 = torch.tensor(m_surf, device=device, dtype=torch.int32)

    alpha = 0.81
    scores_full = _combo_scores(
        centers_w_H_1,
        centers_w_H_2,
        centers_1,
        centers_2,
        points_1,
        points_2,
        partial_1,
        partial_2,
        point_charges_1,
        point_charges_2,
        radii_1,
        radii_2,
        alpha,
        lam=0.001,
        probe_radius=1.0,
        esp_weight=0.5,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1,
        M_real_atoms_w_H_2=M_real_atoms_w_H_2,
        N_real_centers=N_real_centers,
        M_real_centers=M_real_centers,
        N_real_surf_1=N_real_surf_1,
        M_real_surf_2=M_real_surf_2,
        topk=3,
        steps_fine=5,
        lr=0.075,
    )

    # Slice-out runs should match per-item results when padding is mask-safe.
    scores_slice0 = _combo_scores(
        centers_w_H_1[:1],
        centers_w_H_2[:1],
        centers_1[:1],
        centers_2[:1],
        points_1[:1],
        points_2[:1],
        partial_1[:1],
        partial_2[:1],
        point_charges_1[:1],
        point_charges_2[:1],
        radii_1[:1],
        radii_2[:1],
        alpha,
        lam=0.001,
        probe_radius=1.0,
        esp_weight=0.5,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1[:1],
        M_real_atoms_w_H_2=M_real_atoms_w_H_2[:1],
        N_real_centers=N_real_centers[:1],
        M_real_centers=M_real_centers[:1],
        N_real_surf_1=N_real_surf_1[:1],
        M_real_surf_2=M_real_surf_2[:1],
        topk=3,
        steps_fine=5,
        lr=0.075,
    )

    scores_slice1 = _combo_scores(
        centers_w_H_1[1:2],
        centers_w_H_2[1:2],
        centers_1[1:2],
        centers_2[1:2],
        points_1[1:2],
        points_2[1:2],
        partial_1[1:2],
        partial_2[1:2],
        point_charges_1[1:2],
        point_charges_2[1:2],
        radii_1[1:2],
        radii_2[1:2],
        alpha,
        lam=0.001,
        probe_radius=1.0,
        esp_weight=0.5,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1[1:2],
        M_real_atoms_w_H_2=M_real_atoms_w_H_2[1:2],
        N_real_centers=N_real_centers[1:2],
        M_real_centers=M_real_centers[1:2],
        N_real_surf_1=N_real_surf_1[1:2],
        M_real_surf_2=M_real_surf_2[1:2],
        topk=3,
        steps_fine=5,
        lr=0.075,
    )

    s_full = scores_full.detach().cpu().tolist()
    assert all(math.isfinite(x) for x in s_full)
    assert abs(float(scores_slice0[0]) - s_full[0]) < 1e-4
    assert abs(float(scores_slice1[0]) - s_full[1]) < 1e-4


def _esp_combo_synth_batch(device):
    """Two synthetic esp_combo pairs of different sizes on ``device``, reseeded per call."""
    torch.manual_seed(0)
    device = torch.device(device)
    K = 2
    n_wH, m_wH = [9, 2], [8, 3]
    n_surf, m_surf = [40, 40], [41, 41]
    nwp, mwp, nsp, msp = max(n_wH), max(m_wH), max(n_surf), max(m_surf)
    z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)
    cwH1, cwH2 = z(K, nwp, 3), z(K, mwp, 3)
    p1, p2 = z(K, nwp), z(K, mwp)
    r1 = torch.ones(K, nwp, device=device) * 1.5
    r2 = torch.ones(K, mwp, device=device) * 1.5
    c1, c2 = z(K, nwp, 3), z(K, mwp, 3)
    pts1, pts2 = z(K, nsp, 3), z(K, msp, 3)
    pc1, pc2 = z(K, nsp), z(K, msp)
    for i in range(K):
        cwH1[i, : n_wH[i]] = torch.randn(n_wH[i], 3) * 0.2
        cwH2[i, : m_wH[i]] = torch.randn(m_wH[i], 3) * 0.2
        c1[i, : n_wH[i]] = cwH1[i, : n_wH[i]]
        c2[i, : m_wH[i]] = cwH2[i, : m_wH[i]]
        pts1[i, : n_surf[i]] = torch.randn(n_surf[i], 3) * 0.3
        pts2[i, : m_surf[i]] = torch.randn(m_surf[i], 3) * 0.3
        pc1[i, : n_surf[i]] = torch.randn(n_surf[i]) * 0.05
        pc2[i, : m_surf[i]] = torch.randn(m_surf[i]) * 0.05
    ints = lambda lst: torch.tensor(lst, device=device, dtype=torch.int32)
    args = (cwH1, cwH2, c1, c2, pts1, pts2, p1, p2, pc1, pc2, r1, r2, 0.81)
    kw = dict(lam=0.001, probe_radius=1.0, esp_weight=0.5,
              N_real_atoms_w_H_1=ints(n_wH), M_real_atoms_w_H_2=ints(m_wH),
              N_real_centers=ints(n_wH), M_real_centers=ints(m_wH),
              N_real_surf_1=ints(n_surf), M_real_surf_2=ints(m_surf),
              topk=3, steps_fine=20, lr=0.075)
    return args, kw


def test_esp_combo_numba_cpu_and_matches_triton():
    """esp_combo on numba gives finite in-range scores and matches Triton when CUDA is present."""
    pytest.importorskip("numba")
    args, kw = _esp_combo_synth_batch("cpu")
    scores_cpu = _combo_scores(*args, **kw)
    s_cpu = scores_cpu.detach().cpu().tolist()
    assert all(math.isfinite(x) for x in s_cpu)
    assert all(-1e-4 <= x <= 1.0 + 1e-4 for x in s_cpu)

    if torch.cuda.is_available():
        try:
            import triton  # noqa: F401
        except Exception:
            return
        args_g, kw_g = _esp_combo_synth_batch("cuda")
        scores_gpu = _combo_scores(*args_g, **kw_g)
        s_gpu = scores_gpu.detach().cpu().tolist()
        for a, b in zip(s_cpu, s_gpu):
            assert abs(a - b) < 5e-3, f"numba {a} vs triton {b}"


def test_pharm_similarity_matches_legacy():
    from shepherd_score.accel.drivers.pharm_overlap import batch_pharm_similarity
    from shepherd_score.score.pharmacophore_scoring import get_overlap_pharm

    torch.manual_seed(0)

    N, M = 7, 6
    anchors_1 = torch.randn(N, 3, dtype=torch.float32)
    anchors_2 = torch.randn(M, 3, dtype=torch.float32)
    vectors_1 = torch.randn(N, 3, dtype=torch.float32)
    vectors_2 = torch.randn(M, 3, dtype=torch.float32)

    vectors_1 = vectors_1 / (vectors_1.norm(dim=1, keepdim=True) + 1e-8)
    vectors_2 = vectors_2 / (vectors_2.norm(dim=1, keepdim=True) + 1e-8)

    types_1 = torch.randint(0, 8, (N,), dtype=torch.int64)
    types_2 = torch.randint(0, 8, (M,), dtype=torch.int64)

    cases = [
        ("tanimoto", False, False),
        ("tversky", False, False),
        ("tversky_ref", False, False),
        ("tversky_fit", False, False),
        ("tanimoto", True, False),
        ("tversky", True, False),
        ("tanimoto", True, True),
        ("tversky_ref", True, True),
    ]

    for similarity, extended_points, only_extended in cases:
        legacy = get_overlap_pharm(
            ptype_1=types_1,
            ptype_2=types_2,
            anchors_1=anchors_1,
            anchors_2=anchors_2,
            vectors_1=vectors_1,
            vectors_2=vectors_2,
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
        )

        batched = batch_pharm_similarity(
            anchors_1.unsqueeze(0),
            anchors_2.unsqueeze(0),
            vectors_1.unsqueeze(0),
            vectors_2.unsqueeze(0),
            types_1.unsqueeze(0),
            types_2.unsqueeze(0),
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            N_real=torch.tensor([N], dtype=torch.int32),
            M_real=torch.tensor([M], dtype=torch.int32),
        )[0]

        assert torch.allclose(legacy, batched, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
def test_fast_pharm_batch_smoke_flags():
    _require_fast_cuda()

    from shepherd_score.accel.drivers.pharm import fast_optimize_pharm_overlay_batch

    torch.manual_seed(0)
    device = torch.device("cuda")

    B = 2
    N = 6
    M = 5

    anchors_1 = torch.randn(B, N, 3, device=device, dtype=torch.float32) * 0.5
    anchors_2 = torch.randn(B, M, 3, device=device, dtype=torch.float32) * 0.5
    vectors_1 = torch.randn(B, N, 3, device=device, dtype=torch.float32)
    vectors_2 = torch.randn(B, M, 3, device=device, dtype=torch.float32)
    types_1 = torch.randint(0, 8, (B, N), device=device, dtype=torch.int64)
    types_2 = torch.randint(0, 8, (B, M), device=device, dtype=torch.int64)

    N_real = torch.tensor([N, N], device=device, dtype=torch.int32)
    M_real = torch.tensor([M, M], device=device, dtype=torch.int32)

    for similarity, extended_points, only_extended in [
        ("tversky_ref", True, False),
        ("tversky_fit", True, True),
    ]:
        _, _, _, _, scores = fast_optimize_pharm_overlay_batch(
            types_1,
            types_2,
            anchors_1,
            anchors_2,
            vectors_1,
            vectors_2,
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            num_repeats=1,
            N_real=N_real,
            M_real=M_real,
            topk=2,
            steps_fine=2,
            lr=0.075,
        )
        out = scores.detach().cpu().tolist()
        assert all(math.isfinite(x) for x in out)

