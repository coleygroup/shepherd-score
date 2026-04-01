"""Backward compatibility shim — import from shepherd_score.alignment._jax instead."""
from shepherd_score.alignment._jax import *  # noqa: F401, F403
from shepherd_score.alignment._jax import (   # explicit for IDE/type checkers
    optimize_ROCS_overlay_jax,
    optimize_ROCS_overlay_jax_mask,
    optimize_ROCS_esp_overlay_jax,
    optimize_esp_combo_score_overlay_jax,
    optimize_pharm_overlay_jax,
    optimize_pharm_overlay_jax_vectorized,
    optimize_pharm_overlay_jax_vectorized_mask,
    convert_to_jnp_array,
    _make_jit_val_grad_pharm_vectorized,
    _make_jit_val_grad_pharm_vectorized_mask,
)
from shepherd_score.alignment._jax_parallel import (  # parallel (shard_map) functions
    optimize_ROCS_overlay_jax_vol_shmap,
    optimize_ROCS_esp_overlay_jax_vol_esp_shmap,
    optimize_ROCS_overlay_jax_surf_shmap,
    optimize_ROCS_esp_overlay_jax_surf_esp_shmap,
    optimize_pharm_overlay_jax_pharm_shmap,
)
