"""Alignment mode specifications, channels, and optimizer defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# Objective terms and mode specs

@dataclass(frozen=True)
class Term:
    """One contribution to a mode's objective, computed by one kernel launch per fine step.

    kernel : which value+gradient kernel evaluates it. One of
        ``"shape"``   Gaussian volume overlap of two point clouds
                       (``overlap_score_grad_se3_batch``; channels: ref points, fit points);
        ``"esp"``     shape overlap weighted by a per-point scalar field
                       (``overlap_score_grad_esp_se3_batch``; points + scalar per side);
        ``"color"``   directionless typed-anchor overlap
                       (``pharm_color_score_grad_se3_batch``; anchors + labels per side);
        ``"pharm"``   directional pharmacophore overlap
                       (``pharm_grad_dq_se3_batch``; anchors + vectors + labels per side);
        ``"avoid"``   linear hard-sphere excluded-volume penalty
                       (``overlap_score_grad_avoid_se3_batch``; a fixed avoid cloud on the ref
                       slot, fit points on the fit slot; the engine keeps the cloud per pair,
                       Z-ordered on the way in, and the kernels skip points beyond reach of the fit);
        ``"esp_cmp"`` the ShaEP surface-ESP agreement
                       (``esp_agreement_grad_se3_batch``; with-H atoms, charges, radii, surface
                       points, surface ESP per side).
    ref / fit : channel names, in the kernel's argument order, for each side.
    params : call-kwarg names forwarded to the kernel (``"alpha"``, ``"lam"``, ``"avoid_min_dist"``).
    reduction : how the raw overlap becomes a similarity:
        ``"tanimoto"``  V / (VAA + VBB - V)            scale = (VAA+VBB) / denom^2
        ``"tversky"``   V / (k V + C), C = ta VAA + tb VBB, k = 1 - ta - tb; scale = C / denom^2
        ``"raw"``       the value itself (a penalty); scale = 1
        ``"agreement"`` the esp_cmp average in [0, 1]; scale = 1
        ``"pharm_sim"`` the pharmacophore family's own reduction, selected by the call's
                        ``similarity`` keyword: a guarded Tanimoto, or a guarded Tversky
                        ``V / (sigma VAA + (1-sigma) VBB)`` clamped to 1 with sigma from
                        ``{tversky: 0.95, tversky_ref: 1.0, tversky_fit: 0.05}``. Distinct
                        from the ``tversky`` reduction above: it takes no ta/tb and its
                        gradient is the hinged ``-1(V < D)/D``.
    weight : the blend weight: a call-kwarg name, a float, or ``None`` for the complement of
        the other term's named weight (``1 - w``). Negative weights subtract (penalties).
    grad : whether the term's kernel gradient steers the pose. Every registered mode sets it,
        so each optimiser follows the gradient of the score it reports; ``False`` makes the
        term enter only the tracked score.
    guard : mask the term to zero for pairs where either side has no real points (the lipo
        family), so an empty channel contributes neither score nor gradient.
    tables : lookup-table set for the typed kernels: ``"color"`` (directionless pharmacophore),
        ``"pharm"`` (directional pharmacophore) or ``"element"`` (atomic numbers).
    """
    kernel: str
    ref: Tuple[str, ...]
    fit: Tuple[str, ...]
    params: Tuple[str, ...] = ()
    reduction: str = "tanimoto"
    weight: object = 1.0
    grad: bool = True
    guard: bool = False
    tables: Optional[str] = None


@dataclass(frozen=True)
class ModeSpec:
    """Everything the library needs to know about one alignment mode, as data.

    name : canonical mode id (also the ``align_with_<name>`` / ``_align_batch_<name>`` suffix).
    attrs : ``(transform_attr, score_attr)`` written on a ``MoleculePair``.
    seeds / steps / patience : the optimiser defaults (SO(3) multi-starts, fine steps,
        early-stop patience in 5-step checks).
    seed_channel : the point channel whose principal frames generate the seeds. Modes whose seed
        channel is one a canonical store rotates into its principal frame get the store's
        constant seed set on the screen path (see ``CONST_SEED_MODES``).
    channels : every channel the objective reads (per side; ``avoid`` is pair-side).
    bucket : the channels whose real counts are the cost-driving pad dims. One channel means
        ``PadSpec(merge={ref, fit})``; several means one merge dim per side per channel, with
        ``work`` naming the cost model.
    work : ``"product"`` (default N_pad*M_pad over the single bucket channel) or ``"combo"``
        (the sum of the three channel products the ShaEP combo evaluates).
    terms : the objective terms, in gradient-accumulation order.
    params : call keyword -> default. ``None`` marks a required keyword.
    center_clouds : center both seed-channel clouds on their own real-point centroids before
        seeding and fold the shift back into the returned translation (the pharm family).
    pharm_style : the pharm family's optimiser tail: unit-normalise ``q`` before the kernel,
        apply the normalisation Jacobian and the guarded/clamped pharm similarity, and use the
        un-projected fused Adam.
    graph_budget : per-row work budget for ``drivers/_graphed.graph_cap``; ``None`` disables the
        CUDA-graph fine loop for the mode.
    graph_full_steps : replay the graph for the full step count instead of the blocked
        early-stop (the combo modes, whose ESP landscape converges slowly).
    cpu_fused : whether the fused numba fine loop is used on CPU. False for the pharmacophore
        family, whose multi-basin objective lets the float32 tail's rounding change which seed
        wins.
    cpu_fused_max_pad : the fused loop is used only when every padded width is at most this
        (``vol_esp`` only).
    fused_pair : the two gradient terms can be evaluated by one fused kernel (the shape+color
        single launch in ``kernels/vol_color_triton.py``), collapsing two launches per fine step
        into one. CUDA-only and single-tile, so the engine falls back to the two separate
        kernels on CPU tensors or past ``VOL_COLOR_FUSED_MAX_PAD``.
    multipose : poses per CTA for the deduplicated multi-pose shape layout (surf only).
    pose_cap : bound the fine-loop sub-batch at ``_pad._FINE_CHUNK_POSES`` poses (vol only).
    lam_scaling : multiply ``lam`` by ``score.constants.LAM_SCALING`` (the surface convention).
    channel_switch : ``{virtual channel: (kwarg, value, if_equal, else)}``; the combo modes
        score volumetric shape when ``alpha == 0.81`` and surface shape otherwise.
    screen_lr : the fine-loop learning rate ``screen`` and ``MoleculePairBatch`` use when the
        caller passes none (0.1 or 0.075 per mode, the values both used on ``main``).
    coarse_channel : the cloud the legacy ``trans_init`` coarse grid is built from when that is
        not the seed channel (the combo modes build it from the surface clouds while seeding
        from the volume centers).
    """
    name: str
    attrs: Tuple[str, str]
    seeds: int
    steps: int
    patience: int
    seed_channel: str
    channels: Tuple[str, ...]
    bucket: Tuple[str, ...]
    terms: Tuple[Term, ...]
    params: dict = field(default_factory=dict)
    work: str = "product"
    center_clouds: bool = False
    pharm_style: bool = False
    graph_budget: Optional[int] = 300_000_000
    graph_full_steps: bool = False
    cpu_fused: bool = True
    cpu_fused_max_pad: Optional[int] = None
    fused_pair: bool = False
    multipose: int = 1
    pose_cap: bool = False
    lam_scaling: bool = False
    channel_switch: dict = field(default_factory=dict)
    coarse_channel: Optional[str] = None
    screen_lr: float = 0.075

    @property
    def transform_attr(self) -> str:
        return self.attrs[0]

    @property
    def score_attr(self) -> str:
        return self.attrs[1]

    def resolve_channel(self, name: str, kwargs: dict) -> str:
        """Concrete channel for ``name`` under this call's keywords (``channel_switch``)."""
        sw = self.channel_switch.get(name)
        if sw is None:
            return name
        kw, value, if_eq, else_ = sw
        return if_eq if kwargs.get(kw, self.params.get(kw)) == value else else_

    def all_channels(self) -> Tuple[str, ...]:
        """Every concrete channel this mode can read, switches expanded."""
        out = []
        for c in self.channels:
            sw = self.channel_switch.get(c)
            names = (sw[2], sw[3]) if sw else (c,)
            for n in names:
                if n not in out:
                    out.append(n)
        return tuple(out)


# ---- shared building blocks ------------------------------------------------------------------
_SHAPE = Term("shape", ("atoms",), ("atoms",), params=("alpha",))
_SHAPE_T = Term("shape", ("atoms",), ("atoms",), params=("alpha",), reduction="tversky")
_SURF = Term("shape", ("surf",), ("surf",), params=("alpha",))
_SURF_T = Term("shape", ("surf",), ("surf",), params=("alpha",), reduction="tversky")
_HEAVY_ESP = Term("esp", ("heavy", "charges"), ("heavy", "charges"), params=("alpha", "lam"))
_HEAVY_ESP_T = Term("esp", ("heavy", "charges"), ("heavy", "charges"), params=("alpha", "lam"),
                    reduction="tversky")
_SURF_ESP = Term("esp", ("surf", "surf_esp"), ("surf", "surf_esp"), params=("alpha", "lam"))
_SURF_ESP_T = Term("esp", ("surf", "surf_esp"), ("surf", "surf_esp"), params=("alpha", "lam"),
                   reduction="tversky")


def _field_blend(field_pos, field_val, weight, reduction="tanimoto"):
    """shape + a per-atom scalar field carried by the ESP kernel (lipo / mr / fukui)."""
    return (Term("shape", ("atoms",), ("atoms",), params=("alpha",), reduction=reduction,
                 weight=None),
            Term("esp", (field_pos, field_val), (field_pos, field_val), params=("alpha", "lam"),
                 reduction=reduction, weight=weight, guard=True))


def _color_blend(reduction="tanimoto"):
    return (Term("shape", ("atoms",), ("atoms",), params=("alpha",), reduction=reduction,
                 weight=None),
            Term("color", ("pharm_ancs", "pharm_types"), ("pharm_ancs", "pharm_types"),
                 reduction=reduction, weight="color_weight", tables="color"))


_COMBO_CH = ("cwh", "partial", "radii", "surf", "surf_esp", "centers")
_COMBO_SWITCH = {"centers": ("alpha", 0.81, "atoms", "surf")}


def _combo_terms(reduction="tanimoto"):
    return (Term("shape", ("centers",), ("centers",), params=("alpha",), reduction=reduction,
                 weight=None),
            Term("esp_cmp", ("cwh", "partial", "radii", "surf", "surf_esp"),
                 ("cwh", "partial", "radii", "surf", "surf_esp"),
                 params=("lam", "probe_radius"), reduction="agreement", weight="esp_weight"))


_TV = {"tversky_alpha": 0.95, "tversky_beta": 0.05}
_LR = {"lr": 0.075}

# ---- the 21 canonical modes, in public order ---------------------------------------------------
SPECS = {}


def _reg(spec: ModeSpec) -> ModeSpec:
    SPECS[spec.name] = spec
    return spec


_reg(ModeSpec("vol", ("transform_vol_noH", "sim_aligned_vol_noH"), 10, 30, 2,
              seed_channel="atoms", channels=("atoms",), bucket=("atoms",), terms=(_SHAPE,),
              params={"alpha": 0.81, **_LR}, pose_cap=True))
_reg(ModeSpec("vol_esp", ("transform_vol_esp_noH", "sim_aligned_vol_esp_noH"), 16, 50, 5,
              seed_channel="heavy", channels=("heavy", "charges"), bucket=("heavy",),
              terms=(_HEAVY_ESP,), params={"alpha": 0.81, "lam": 0.1, **_LR},
              cpu_fused_max_pad=100, screen_lr=0.1))
_reg(ModeSpec("surf", ("transform_surf", "sim_aligned_surf"), 8, 40, 2,
              seed_channel="surf", channels=("surf",), bucket=("surf",), terms=(_SURF,),
              params={"alpha": 0.81, **_LR}, multipose=8))
_reg(ModeSpec("surf_esp", ("transform_surf_esp", "sim_aligned_surf_esp"), 8, 40, 5,
              seed_channel="surf", channels=("surf", "surf_esp"), bucket=("surf",),
              terms=(_SURF_ESP,), params={"alpha": 0.81, "lam": 0.3, **_LR},
              lam_scaling=True, screen_lr=0.1))
_reg(ModeSpec("vol_and_surf_esp", ("transform_vol_and_surf_esp", "sim_aligned_vol_and_surf_esp"),
              8, 60, 5, seed_channel="centers", channels=_COMBO_CH, bucket=("cwh", "surf", "centers"),
              work="combo", terms=_combo_terms(),
              params={"alpha": 0.81, "lam": 0.001, "probe_radius": 1.0, "esp_weight": 0.5, **_LR},
              graph_budget=8_000_000, graph_full_steps=True, channel_switch=_COMBO_SWITCH,
              coarse_channel="surf", screen_lr=0.1))
_reg(ModeSpec("pharm", ("transform_pharm", "sim_aligned_pharm"), 32, 50, 5,
              seed_channel="pharm_ancs", channels=("pharm_ancs", "pharm_vecs", "pharm_types"),
              bucket=("pharm_ancs",),
              terms=(Term("pharm", ("pharm_ancs", "pharm_vecs", "pharm_types"),
                          ("pharm_ancs", "pharm_vecs", "pharm_types"), tables="pharm",
                          reduction="pharm_sim"),),
              params={"similarity": "tanimoto", "extended_points": False, "only_extended": False,
                      **_LR},
              center_clouds=True, pharm_style=True, graph_budget=10_000_000,
              cpu_fused=False, screen_lr=0.1))
_reg(ModeSpec("vol_color", ("transform_vol_color", "sim_aligned_vol_color"), 16, 40, 2,
              seed_channel="atoms", channels=("atoms", "pharm_ancs", "pharm_types"), bucket=("atoms",),
              terms=_color_blend(), params={"alpha": 0.81, "color_weight": 0.5, **_LR},
              graph_budget=30_000_000, fused_pair=True, screen_lr=0.1))
_reg(ModeSpec("vol_tversky", ("transform_vol_tversky", "sim_aligned_vol_tversky"), 10, 40, 2,
              seed_channel="atoms", channels=("atoms",), bucket=("atoms",), terms=(_SHAPE_T,),
              params={"alpha": 0.81, **_TV, **_LR}))
_reg(ModeSpec("vol_lipo", ("transform_vol_lipo", "sim_aligned_vol_lipo"), 16, 50, 2,
              seed_channel="atoms", channels=("atoms", "lipo_pos", "lipo"), bucket=("atoms",),
              terms=_field_blend("lipo_pos", "lipo", "lipo_weight"),
              params={"alpha": 0.81, "lam": 0.1, "lipo_weight": 0.5, **_LR},
              graph_budget=30_000_000, screen_lr=0.1))
_reg(ModeSpec("vol_esp_tversky", ("transform_vol_esp_tversky", "sim_aligned_vol_esp_tversky"),
              16, 50, 5, seed_channel="heavy", channels=("heavy", "charges"), bucket=("heavy",),
              terms=(_HEAVY_ESP_T,), params={"alpha": 0.81, "lam": 0.1, **_TV, **_LR}))
# experimental modes
_reg(ModeSpec("vol_mr", ("transform_vol_mr", "sim_aligned_vol_mr"), 16, 50, 2,
              seed_channel="atoms", channels=("atoms", "mr_pos", "mr"), bucket=("atoms",),
              terms=_field_blend("mr_pos", "mr", "mr_weight"),
              params={"alpha": 0.81, "lam": 0.1, "mr_weight": 0.5, **_LR},
              graph_budget=30_000_000, screen_lr=0.1))
_reg(ModeSpec("surf_tversky", ("transform_surf_tversky", "sim_aligned_surf_tversky"), 8, 40, 2,
              seed_channel="surf", channels=("surf",), bucket=("surf",), terms=(_SURF_T,),
              params={"alpha": 0.81, **_TV, **_LR}))
_reg(ModeSpec("surf_esp_tversky", ("transform_surf_esp_tversky", "sim_aligned_surf_esp_tversky"),
              8, 40, 5, seed_channel="surf", channels=("surf", "surf_esp"), bucket=("surf",),
              terms=(_SURF_ESP_T,), params={"alpha": 0.81, "lam": 0.3, **_TV, **_LR},
              lam_scaling=True))
_reg(ModeSpec("vol_lipo_tversky", ("transform_vol_lipo_tversky", "sim_aligned_vol_lipo_tversky"),
              16, 50, 2, seed_channel="atoms", channels=("atoms", "lipo_pos", "lipo"), bucket=("atoms",),
              terms=_field_blend("lipo_pos", "lipo", "lipo_weight", reduction="tversky"),
              params={"alpha": 0.81, "lam": 0.1, "lipo_weight": 0.5, **_TV, **_LR},
              graph_budget=30_000_000, screen_lr=0.1))
_reg(ModeSpec("vol_color_tversky", ("transform_vol_color_tversky", "sim_aligned_vol_color_tversky"),
              16, 40, 2, seed_channel="atoms", channels=("atoms", "pharm_ancs", "pharm_types"),
              bucket=("atoms",), terms=_color_blend(reduction="tversky"),
              params={"alpha": 0.81, "color_weight": 0.5, **_TV, **_LR},
              graph_budget=30_000_000, fused_pair=True, screen_lr=0.1))
_reg(ModeSpec("vol_atomtype", ("transform_vol_atomtype", "sim_aligned_vol_atomtype"), 16, 40, 2,
              seed_channel="atoms", channels=("atoms", "type_pos", "atomlabels"), bucket=("atoms",),
              terms=(Term("shape", ("atoms",), ("atoms",), params=("alpha",), weight=None),
                     Term("color", ("type_pos", "atomlabels"), ("type_pos", "atomlabels"),
                          weight="atomtype_weight", tables="element", params=("alpha",))),
              params={"alpha": 0.81, "atomtype_weight": 0.5, **_LR},
              graph_budget=30_000_000, screen_lr=0.1))
_reg(ModeSpec("vol_pharm", ("transform_vol_pharm", "sim_aligned_vol_pharm"), 32, 50, 2,
              seed_channel="atoms", channels=("atoms", "pharm_ancs", "pharm_vecs", "pharm_types"),
              bucket=("atoms",),
              terms=(Term("shape", ("atoms",), ("atoms",), params=("alpha",), weight=None),
                     Term("pharm", ("pharm_ancs", "pharm_vecs", "pharm_types"),
                          ("pharm_ancs", "pharm_vecs", "pharm_types"),
                          weight="color_weight", tables="pharm")),
              params={"alpha": 0.81, "color_weight": 0.5, **_LR},
              graph_budget=10_000_000, screen_lr=0.1))
_reg(ModeSpec("pharm_tversky", ("transform_pharm_tversky", "sim_aligned_pharm_tversky"), 32, 50, 5,
              seed_channel="pharm_ancs", channels=("pharm_ancs", "pharm_vecs", "pharm_types"),
              bucket=("pharm_ancs",),
              terms=(Term("pharm", ("pharm_ancs", "pharm_vecs", "pharm_types"),
                          ("pharm_ancs", "pharm_vecs", "pharm_types"), tables="pharm",
                          reduction="pharm_sim"),),
              params={"similarity": "tversky", "extended_points": False, "only_extended": False,
                      **_LR},
              center_clouds=True, pharm_style=True, graph_budget=10_000_000,
              cpu_fused=False, screen_lr=0.1))
_reg(ModeSpec("vol_and_surf_esp_tversky",
              ("transform_vol_and_surf_esp_tversky", "sim_aligned_vol_and_surf_esp_tversky"),
              8, 60, 5, seed_channel="centers", channels=_COMBO_CH, bucket=("cwh", "surf", "centers"),
              work="combo", terms=_combo_terms(reduction="tversky"),
              params={"alpha": 0.81, "lam": 0.001, "probe_radius": 1.0, "esp_weight": 0.5,
                      **_TV, **_LR},
              graph_budget=8_000_000, graph_full_steps=True, channel_switch=_COMBO_SWITCH,
              coarse_channel="surf", screen_lr=0.1))
_reg(ModeSpec("vol_fukui", ("transform_vol_fukui", "sim_aligned_vol_fukui"), 16, 50, 2,
              seed_channel="atoms", channels=("atoms", "fukui_pos", "fukui"), bucket=("atoms",),
              terms=_field_blend("fukui_pos", "fukui", "fukui_weight"),
              params={"alpha": 0.81, "lam": 0.1, "fukui_weight": 0.5, **_LR},
              graph_budget=30_000_000, screen_lr=0.1))
# shape Tanimoto minus a linear hard-sphere excluded-volume penalty against a fixed avoid cloud,
# a query-side (pair-level) third input carried by the ``avoid`` channel.
_reg(ModeSpec("vol_avoid", ("transform_vol_avoid", "sim_aligned_vol_avoid"), 16, 50, 2,
              seed_channel="atoms", channels=("atoms", "avoid"), bucket=("atoms",),
              terms=(Term("shape", ("atoms",), ("atoms",), params=("alpha",), weight=1.0),
                     Term("avoid", ("avoid",), ("atoms",), params=("avoid_min_dist",),
                          reduction="raw", weight="-avoid_weight", guard=True)),
              params={"alpha": 0.81, "avoid_min_dist": 2.0, "avoid_weight": 1.0, **_LR},
              graph_budget=None, screen_lr=0.1))

# Derived flat tables (the surface every consumer reads)

#: Canonical mode id -> (transform_attr, score_attr) written in-place on a MoleculePair by
#: ``MoleculePairBatch.align_with_<mode>``. Derived from SPECS, in public order.
MODE_ATTRS = {m: s.attrs for m, s in SPECS.items()}

#: The 21 canonical mode ids, in public order.
CANONICAL_MODES = tuple(MODE_ATTRS)

#: Legacy mode names -> canonical. The public entry points (``align_with_esp`` /
#: ``align_with_esp_combo``, the screen ``mode=`` arg, the pool mode strings, old pickles)
#: normalise through ``canonical()``.
LEGACY_MODE_ALIASES = {"esp": "surf_esp", "esp_combo": "vol_and_surf_esp"}


def canonical(mode: str) -> str:
    """Resolve a (possibly legacy) mode name to its canonical form; unknown names pass through."""
    return LEGACY_MODE_ALIASES.get(mode, mode)


def spec_of(mode: str) -> ModeSpec:
    """The :class:`ModeSpec` for a (possibly legacy) mode name."""
    return SPECS[canonical(mode)]


#: Modes with a process-per-GPU (multi_gpu) and CPU-pool (cpu_pool) tensor spec; every mode
#: declares its tensors through its channels, so this is the whole registry.
PROCESS_MODES = tuple(SPECS)

#: Per-mode (SO(3) seed count, fine step count) defaults, chosen at the accuracy/throughput
#: knee; a caller's ``max_num_steps`` / ``num_repeats`` override them.
MODE_SEEDS = {m: s.seeds for m, s in SPECS.items()}
MODE_STEPS = {m: s.steps for m, s in SPECS.items()}

#: Channels a canonical ProfileStore rotates into the principal frame; a mode seeding from one of
#: them takes the store's constant seed set on the screen path (``_common.canonical_seed_quats``).
CANONICAL_SEED_CHANNELS = ("atoms", "heavy")


def _const_seed_capable(s: ModeSpec) -> bool:
    sw = s.channel_switch.get(s.seed_channel)
    seed_ch = sw[2] if sw else s.seed_channel
    return seed_ch in CANONICAL_SEED_CHANNELS


CONST_SEED_MODES = tuple(m for m, s in SPECS.items() if _const_seed_capable(s))
