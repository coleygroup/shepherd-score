"""
Mode-generic SE(3) optimizer: multi-start Adam over the fit pose, maximizing ``mode.score``.
"""
from typing import Any, Optional, Tuple

import torch
from torch import optim

from shepherd_score.alignment._torch import _initialize_se3_params, _initialize_se3_params_with_translations
from shepherd_score.alignment.utils.se3 import get_SE3_transform


def tree_map(fn, tree, _results_by_id=None):
    """Apply ``fn`` to every array in a dict / tuple nest (the shape of ``Mode.inputs``).

    An array object that appears more than once (``Combined`` terms sharing coordinates) is
    passed to ``fn`` once, and every occurrence gets that one result, so the sharing survives
    tensor conversion and repetition. ``_results_by_id`` maps ``id(array)`` to its result and
    is only passed internally, by the recursion.
    """
    results_by_id = {} if _results_by_id is None else _results_by_id
    if isinstance(tree, dict):
        return {k: tree_map(fn, v, results_by_id) for k, v in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(tree_map(fn, v, results_by_id) for v in tree)
    if id(tree) not in results_by_id:
        results_by_id[id(tree)] = fn(tree)
    return results_by_id[id(tree)]


def optimize_se3(mode,
                 ref: Any,
                 fit: Any,
                 *,
                 num_repeats: int = 50,
                 trans_centers: Optional[torch.Tensor] = None,
                 lr: float = 0.1,
                 max_num_steps: int = 200,
                 verbose: bool = False) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Align ``fit`` to ``ref`` by maximizing ``mode.score`` over SE(3) with autograd.

    Seeds are the identity, the principal-axis alignments and Fibonacci rotations of the
    mode's seed points (or 10 rotations at each of ``trans_centers``). Each seed is optimized
    with Adam until the mean loss changes by less than 1e-5 for 10 consecutive steps.

    Parameters
    ----------
    mode : shepherd_score.modes.Mode
    ref, fit : nest of torch.Tensor
        ``mode.inputs`` of the reference and fit molecules, as tensors.
    num_repeats : int
        Number of seeds when ``trans_centers`` is None.
    trans_centers : torch.Tensor (P, 3), optional
        Seed translations for the fit's center of mass.
    lr : float
        Adam learning rate.
    max_num_steps : int
        Maximum optimization steps.
    verbose : bool
        Print the initial, intermediate and final scores.

    Returns
    -------
    aligned : nest of torch.Tensor
        ``fit`` under the best pose.
    se3_transform : torch.Tensor (4, 4)
    score : torch.Tensor ()
    """
    ref_seed, fit_seed = mode.seed_points(ref), mode.seed_points(fit)
    if trans_centers is None:
        se3_params = _initialize_se3_params(ref_seed, fit_seed, num_repeats=num_repeats)
    else:
        se3_params = _initialize_se3_params_with_translations(ref_seed, fit_seed, trans_centers,
                                                              num_repeats_per_trans=10)
    batched = se3_params.dim() == 2    # a single seed (num_repeats=1) stays unbatched
    if batched:
        n_poses = se3_params.shape[0]
        fit_rep = tree_map(lambda x: x.repeat((n_poses,) + (1,) * x.dim()), fit)
    else:
        fit_rep = fit

    with torch.no_grad():
        cache = mode.prepare(ref, fit)
        if verbose:
            print(f'Initial score: {mode.score(ref, fit, cache).item():.3f}')

    optimizer = optim.Adam([se3_params], lr=lr)
    last_loss = float('inf')
    counter = 0
    for step in range(max_num_steps):
        se3 = get_SE3_transform(se3_params)
        loss = 1 - mode.score(ref, mode.pose(fit_rep, se3), cache).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f'Step {step}, Score: {1 - loss.item():.3f}')
        counter = counter + 1 if abs(loss.item() - last_loss) <= 1e-5 else 0
        last_loss = loss.item()
        if counter > 10:
            break

    with torch.no_grad():
        se3 = get_SE3_transform(se3_params.detach())
        aligned = mode.pose(fit_rep, se3)
        scores = mode.score(ref, aligned, cache)
    if verbose:
        print(f'Optimized score -- max: {scores.max():.3f} | mean: {scores.mean():.3f} | min: {scores.min():.3f}')
    if not batched:
        return tree_map(lambda x: x.cpu(), aligned), se3.cpu(), scores.cpu()
    best = torch.argmax(scores.cpu())
    return tree_map(lambda x: x[best].cpu(), aligned), se3[best].cpu(), scores[best].cpu()
