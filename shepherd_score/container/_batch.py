"""MoleculePairBatch: batch of MoleculePair objects for fast sequential JAX alignment."""
from typing import List, Tuple

import numpy as np

from shepherd_score.container._core import MoleculePair


def _pad_arrays(arrays, pad_value=0.0, dtype=np.float32):
    """Pad a list of arrays to the same max length along axis 0.

    Parameters
    ----------
    arrays : list of np.ndarray
        Arrays to pad; each shape (N,) or (N, k).
    pad_value : float
        Fill value for padding regions.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    padded : list of np.ndarray
        Padded arrays, all shape (max_len,) or (max_len, k).
    masks : list of np.ndarray
        Binary float32 masks, each shape (max_len,).
    orig_lens : list of int
        Original lengths before padding.
    max_len : int
    """
    max_len = max(a.shape[0] for a in arrays)
    is_2d = arrays[0].ndim == 2
    cols = arrays[0].shape[1] if is_2d else None

    padded, masks, orig_lens = [], [], []
    for a in arrays:
        orig_len = a.shape[0]
        orig_lens.append(orig_len)
        if is_2d:
            p = np.full((max_len, cols), pad_value, dtype=dtype)
        else:
            p = np.full(max_len, pad_value, dtype=dtype)
        p[:orig_len] = a.astype(dtype)
        padded.append(p)

        mask = np.zeros(max_len, dtype=np.float32)
        mask[:orig_len] = 1.0
        masks.append(mask)

    return padded, masks, orig_lens, max_len


class MoleculePairBatch:
    """Batch of MoleculePair objects for fast sequential JAX alignment.

    Pads all atom coordinate arrays to common max shapes so JAX's XLA compiler
    reuses the same compiled function for every pair, avoiding recompilation.
    This modifies each MoleculePair in-place (stores results on the pair).

    This is currently optimized for CPU. A GPU-optimized version would
    benefit from optimizing batches of pairs and using a GPU-optimized alignment.
    """

    def __init__(self, pairs: List[MoleculePair]):
        self.pairs = pairs

    def _pad_and_mask_vol(self, no_H: bool = True, include_charges: bool = False):
        """Extract, pad, and create masks for volumetric (and optionally ESP) alignment.

        Does NOT modify the pair objects. Returns padded arrays and masks.

        Parameters
        ----------
        no_H : bool
            If True, use heavy-atom positions (atom_pos). If False, use all atoms.
        include_charges : bool
            If True, also extract and pad partial charge arrays. The returned tuple
            per entry gains two extra elements: ``(ref_pos_pad, fit_pos_pad,
            ref_ch_pad, fit_ch_pad, mask_ref, mask_fit, orig_ref, orig_fit)``.
            If False, each entry is ``(ref_padded, fit_padded, mask_ref, mask_fit,
            orig_ref, orig_fit)``.

        Returns
        -------
        entries : list of tuples
        max_ref_len : int
        max_fit_len : int
        """
        if no_H:
            ref_pos_arrays = [p.ref_molec.atom_pos for p in self.pairs]
            fit_pos_arrays = [p.fit_molec.atom_pos for p in self.pairs]
            if include_charges:
                ref_ch_arrays = [p.ref_molec.partial_charges[p.ref_molec._nonH_atoms_idx]
                                 for p in self.pairs]
                fit_ch_arrays = [p.fit_molec.partial_charges[p.fit_molec._nonH_atoms_idx]
                                 for p in self.pairs]
        else:
            ref_pos_arrays = [p.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)
                              for p in self.pairs]
            fit_pos_arrays = [p.fit_molec.mol.GetConformer().GetPositions().astype(np.float32)
                              for p in self.pairs]
            if include_charges:
                ref_ch_arrays = [p.ref_molec.partial_charges for p in self.pairs]
                fit_ch_arrays = [p.fit_molec.partial_charges for p in self.pairs]

        ref_padded, masks_ref, orig_refs, max_ref_len = _pad_arrays(ref_pos_arrays)
        fit_padded, masks_fit, orig_fits, max_fit_len = _pad_arrays(fit_pos_arrays)

        if include_charges:
            ref_ch_padded, _, _, _ = _pad_arrays(ref_ch_arrays)
            fit_ch_padded, _, _, _ = _pad_arrays(fit_ch_arrays)
            entries = [
                (rp, fp, rc, fc, mr, mf, ori_r, ori_f)
                for rp, fp, rc, fc, mr, mf, ori_r, ori_f in zip(
                    ref_padded, fit_padded, ref_ch_padded, fit_ch_padded,
                    masks_ref, masks_fit, orig_refs, orig_fits
                )
            ]
        else:
            entries = [
                (rp, fp, mr, mf, ori_r, ori_f)
                for rp, fp, mr, mf, ori_r, ori_f in zip(
                    ref_padded, fit_padded, masks_ref, masks_fit, orig_refs, orig_fits
                )
            ]

        return entries, max_ref_len, max_fit_len

    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       verbose: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using padded masked volumetric similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_noH`` / ``pair.sim_aligned_vol_noH`` (when ``no_H=True``)
        - ``pair.transform_vol``     / ``pair.sim_aligned_vol``      (when ``no_H=False``)

        Parameters
        ----------
        no_H : bool
            Whether to exclude hydrogens. Default is True.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            If True, initialize translations to each ref atom position. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        """
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                'JAX is required for MoleculePairBatch.align_with_vol. '
                'Install it with: pip install "shepherd-score[jax]"'
            ) from exc

        from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax_mask

        entries, _, _ = self._pad_and_mask_vol(no_H=no_H)
        aligned_list = []
        scores = np.zeros((len(self.pairs),))

        for i, (pair, entry) in enumerate(zip(self.pairs, entries)):
            ref_padded, fit_padded, mask_ref, mask_fit, orig_ref, orig_fit = entry

            trans_centers = None
            if trans_init:
                if no_H:
                    trans_centers = pair.ref_molec.atom_pos
                else:
                    trans_centers = pair.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)

            aligned_pts, se3_transform, score = optimize_ROCS_overlay_jax_mask(
                ref_points=jnp.array(ref_padded),
                fit_points=jnp.array(fit_padded),
                mask_ref=jnp.array(mask_ref),
                mask_fit=jnp.array(mask_fit),
                alpha=0.81,
                num_repeats=num_repeats,
                trans_centers=trans_centers,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )

            se3_transform = np.array(se3_transform)
            score = np.array(score)
            aligned_pts = np.array(aligned_pts)[:orig_fit]
            scores[i] = score

            if no_H:
                pair.transform_vol_noH = se3_transform
                pair.sim_aligned_vol_noH = score
            else:
                pair.transform_vol = se3_transform
                pair.sim_aligned_vol = score

            aligned_list.append(aligned_pts)

        return scores, aligned_list

    def align_with_vol_esp(self,
                           lam: float,
                           no_H: bool = True,
                           num_repeats: int = 50,
                           trans_init: bool = False,
                           lr: float = 0.1,
                           max_num_steps: int = 200,
                           verbose: bool = False
                           ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using padded masked volumetric ESP similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_esp_noH`` / ``pair.sim_aligned_vol_esp_noH`` (when ``no_H=True``)
        - ``pair.transform_vol_esp``     / ``pair.sim_aligned_vol_esp``      (when ``no_H=False``)

        Parameters
        ----------
        lam : float
            Partial charge weighting parameter. Typically 0.1 for volumetric.
        no_H : bool
            Whether to exclude hydrogens. Default is True.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            If True, initialize translations to each ref atom position. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        """
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                'JAX is required for MoleculePairBatch.align_with_vol_esp. '
                'Install it with: pip install "shepherd-score[jax]"'
            ) from exc

        from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax_mask

        entries, _, _ = self._pad_and_mask_vol(no_H=no_H, include_charges=True)
        aligned_list = []
        scores = np.zeros((len(self.pairs),))

        for i, (pair, entry) in enumerate(zip(self.pairs, entries)):
            (ref_pos_pad, fit_pos_pad, ref_ch_pad, fit_ch_pad,
             mask_ref, mask_fit, orig_ref, orig_fit) = entry

            trans_centers = None
            if trans_init:
                if no_H:
                    trans_centers = pair.ref_molec.atom_pos
                else:
                    trans_centers = pair.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)

            aligned_pts, se3_transform, score = optimize_ROCS_esp_overlay_jax_mask(
                ref_points=jnp.array(ref_pos_pad),
                fit_points=jnp.array(fit_pos_pad),
                ref_charges=jnp.array(ref_ch_pad),
                fit_charges=jnp.array(fit_ch_pad),
                mask_ref=jnp.array(mask_ref),
                mask_fit=jnp.array(mask_fit),
                alpha=0.81,
                lam=lam,
                num_repeats=num_repeats,
                trans_centers=trans_centers,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )

            se3_transform = np.array(se3_transform)
            score = np.array(score)
            aligned_pts = np.array(aligned_pts)[:orig_fit]
            scores[i] = score

            if no_H:
                pair.transform_vol_esp_noH = se3_transform
                pair.sim_aligned_vol_esp_noH = score
            else:
                pair.transform_vol_esp = se3_transform
                pair.sim_aligned_vol_esp = score

            aligned_list.append(aligned_pts)

        return scores, aligned_list

    def _delegate_alignment(self, method_name: str, score_attr: str, **kwargs):
        """Delegate alignment to each MoleculePair's method and collect results.

        Parameters
        ----------
        method_name : str
            Name of the MoleculePair method to call (e.g. 'align_with_surf').
        score_attr : str
            Name of the attribute on MoleculePair where the score is stored after alignment.
        **kwargs
            Forwarded to each pair's method.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit coordinates for each pair.
        """
        aligned_list = []
        scores = np.zeros(len(self.pairs))
        for i, pair in enumerate(self.pairs):
            aligned_pts = getattr(pair, method_name)(**kwargs)
            scores[i] = float(getattr(pair, score_attr))
            aligned_list.append(aligned_pts)
        return scores, aligned_list

    def align_with_surf(self,
                        alpha: float,
                        num_repeats: int = 50,
                        trans_init: bool = False,
                        lr: float = 0.1,
                        max_num_steps: int = 200,
                        use_jax: bool = True,
                        use_analytical: bool = True,
                        verbose: bool = False
                        ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using surface similarity by delegating to MoleculePair.

        Surface arrays are same-sized across all pairs so no padding is needed.
        Results are stored in-place on each MoleculePair:
        - ``pair.transform_surf`` and ``pair.sim_aligned_surf``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            Apply translation initialization for alignment. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        use_jax : bool
            Whether to use JAX backend. Default is True.
        use_analytical : bool
            Whether to use analytical gradients (PyTorch only). Default is True.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        """
        return self._delegate_alignment(
            'align_with_surf', 'sim_aligned_surf',
            alpha=alpha,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            use_jax=use_jax,
            use_analytical=use_analytical,
            verbose=verbose,
        )

    def align_with_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = True,
                       use_analytical: bool = True,
                       verbose: bool = False
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using ESP+surface similarity by delegating to MoleculePair.

        Surface arrays are same-sized across all pairs so no padding is needed.
        Results are stored in-place on each MoleculePair:
        - ``pair.transform_esp`` and ``pair.sim_aligned_esp``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float
            Weighting factor for ESP scoring. Scaled internally. Default is 0.3.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            Apply translation initialization for alignment. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        use_jax : bool
            Whether to use JAX backend. Default is True.
        use_analytical : bool
            Whether to use analytical gradients (PyTorch only). Default is True.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        """
        return self._delegate_alignment(
            'align_with_esp', 'sim_aligned_esp',
            alpha=alpha,
            lam=lam,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            use_jax=use_jax,
            use_analytical=use_analytical,
            verbose=verbose,
        )

    def _pad_and_mask_pharm(self):
        """Extract, pad, and create masks for pharmacophore alignment.

        Validates that all pairs have pharmacophore data. Does NOT modify the
        pair objects. Returns padded arrays and masks.

        Returns
        -------
        entries : list of tuples
            Each tuple is (ref_ptypes, fit_ptypes,
                           ref_ancs_pad, fit_ancs_pad,
                           ref_vecs_pad, fit_vecs_pad,
                           mask_ref, mask_fit,
                           orig_ref_len, orig_fit_len).
        max_ref_len : int
        max_fit_len : int
        """
        for i, pair in enumerate(self.pairs):
            if (pair.ref_molec.pharm_types is None or
                    pair.fit_molec.pharm_types is None):
                raise ValueError(
                    f'Pair {i} is missing pharmacophore data. '
                    'Create Molecule objects with pharm_multi_vector set to True or False.'
                )

        DUMMY_TYPE = 8  # index of 'Dummy' in P_TYPES

        ref_types_list = [p.ref_molec.pharm_types for p in self.pairs]
        fit_types_list = [p.fit_molec.pharm_types for p in self.pairs]

        max_ref_len = max(t.shape[0] for t in ref_types_list)
        max_fit_len = max(t.shape[0] for t in fit_types_list)

        ref_ancs_padded, masks_ref, orig_refs, _ = _pad_arrays([p.ref_molec.pharm_ancs for p in self.pairs])
        fit_ancs_padded, masks_fit, orig_fits, _ = _pad_arrays([p.fit_molec.pharm_ancs for p in self.pairs])
        ref_vecs_padded, _, _, _ = _pad_arrays([p.ref_molec.pharm_vecs for p in self.pairs])
        fit_vecs_padded, _, _, _ = _pad_arrays([p.fit_molec.pharm_vecs for p in self.pairs])

        entries = []
        for (ref_types, fit_types,
             ref_ancs_pad, fit_ancs_pad,
             ref_vecs_pad, fit_vecs_pad,
             mask_ref, mask_fit,
             orig_ref, orig_fit) in zip(
                ref_types_list, fit_types_list,
                ref_ancs_padded, fit_ancs_padded,
                ref_vecs_padded, fit_vecs_padded,
                masks_ref, masks_fit,
                orig_refs, orig_fits
        ):
            ref_types_pad = np.full(max_ref_len, DUMMY_TYPE, dtype=np.int32)
            ref_types_pad[:orig_ref] = ref_types
            fit_types_pad = np.full(max_fit_len, DUMMY_TYPE, dtype=np.int32)
            fit_types_pad[:orig_fit] = fit_types

            entries.append((ref_types_pad, fit_types_pad,
                            ref_ancs_pad, fit_ancs_pad,
                            ref_vecs_pad, fit_vecs_pad,
                            mask_ref, mask_fit,
                            orig_ref, orig_fit))

        return entries, max_ref_len, max_fit_len

    def align_with_pharm(self,
                         similarity: str = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = 50,
                         trans_init: bool = False,
                         lr: float = 0.1,
                         max_num_steps: int = 200,
                         verbose: bool = False
                         ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Align all pairs using padded masked pharmacophore similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_pharm`` and ``pair.sim_aligned_pharm``

        Parameters
        ----------
        similarity : str
            One of ``'tanimoto'``, ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'``.
        extended_points : bool
            Score HBA/HBD with extended-point Gaussians.
        only_extended : bool
            When ``extended_points`` is True, ignore anchor overlaps.
        num_repeats : int
            Number of SE(3) initializations per pair.
        trans_init : bool
            If True, initialize translations to each ref pharmacophore anchor.
        lr : float
            Optimizer learning rate.
        max_num_steps : int
            Maximum optimization steps.
        verbose : bool
            Print scores per pair.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_anchors_list : list of np.ndarray
            Aligned fit pharmacophore anchors (unpadded) for each pair.
        aligned_vectors_list : list of np.ndarray
            Aligned fit pharmacophore vectors (unpadded) for each pair.
        """
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                'JAX is required for MoleculePairBatch.align_with_pharm. '
                'Install it with: pip install "shepherd-score[jax]"'
            ) from exc

        from shepherd_score.alignment_jax import optimize_pharm_overlay_jax_vectorized_mask

        entries, _, _ = self._pad_and_mask_pharm()
        aligned_anchors_list = []
        aligned_vectors_list = []
        scores = np.zeros((len(self.pairs),))

        for i, (pair, entry) in enumerate(zip(self.pairs, entries)):
            (ref_types_pad, fit_types_pad,
             ref_ancs_pad, fit_ancs_pad,
             ref_vecs_pad, fit_vecs_pad,
             mask_ref, mask_fit,
             orig_ref, orig_fit) = entry

            trans_centers = None
            if trans_init:
                trans_centers = pair.ref_molec.pharm_ancs

            aligned_ancs, aligned_vecs, se3_transform, score = optimize_pharm_overlay_jax_vectorized_mask(
                ref_pharms=jnp.array(ref_types_pad),
                fit_pharms=jnp.array(fit_types_pad),
                ref_anchors=jnp.array(ref_ancs_pad),
                fit_anchors=jnp.array(fit_ancs_pad),
                ref_vectors=jnp.array(ref_vecs_pad),
                fit_vectors=jnp.array(fit_vecs_pad),
                mask_ref=jnp.array(mask_ref),
                mask_fit=jnp.array(mask_fit),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended,
                num_repeats=num_repeats,
                trans_centers=trans_centers,
                init_ref_anchors=pair.ref_molec.pharm_ancs,
                init_fit_anchors=pair.fit_molec.pharm_ancs,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )

            se3_transform = np.array(se3_transform)
            score = float(np.array(score))
            aligned_ancs = np.array(aligned_ancs)[:orig_fit]
            aligned_vecs = np.array(aligned_vecs)[:orig_fit]

            scores[i] = score
            pair.transform_pharm = se3_transform
            pair.sim_aligned_pharm = score

            aligned_anchors_list.append(aligned_ancs)
            aligned_vectors_list.append(aligned_vecs)

        return scores, aligned_anchors_list, aligned_vectors_list
