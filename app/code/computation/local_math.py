"""Site-side math for dSNE-MS (ported from ``local.py``)."""

from typing import Optional

import numpy as np
from framework import cache_file, with_state

from .inputs import SITE_DATA_CACHE
from .tsne import (
    initialize_site_embedding,
    joint_probabilities,
    make_rng,
    normalize_rows,
    site_gradient_step,
)
from .types import (
    PHASE_INITIALIZE,
    PHASE_OPTIMIZE,
    PHASE_SETUP,
    GlobalMessage,
    LocalState,
    SiteUpdate,
)

P_CACHE = "local_P.npy"
OPTIMIZER_CACHE = "local_optimizer.npz"


def compute_site_round(
    message: GlobalMessage,
    state: LocalState,
    *,
    cache_dir: str,
    logger,
    initial_dims: int = 30,
    perplexity: float = 30,
    max_iterations: int = 1000,
    random_seed: Optional[int] = None,
):
    """Dispatch one site round according to the central message's phase."""
    if message.phase == PHASE_SETUP:
        return SiteUpdate(
            phase=PHASE_SETUP,
            n_samples=state.n_samples,
            n_features=state.n_features,
            token=state.token,
        )
    if message.phase == PHASE_INITIALIZE:
        return _initialize(
            message, state, cache_dir, logger, initial_dims, perplexity, random_seed
        )
    if message.phase == PHASE_OPTIMIZE:
        return _optimize(message, state, cache_dir, logger, max_iterations)
    raise ValueError(f"Unexpected workflow phase at site: {message.phase!r}")


def _initialize(
    message, state, cache_dir, logger, initial_dims, perplexity, random_seed
):
    """Build the combined affinity matrix and initial positions (``local_1``)."""
    shared_X = np.load(message.shared_X.path, allow_pickle=False)
    shared_Y = np.asarray(message.shared_Y, dtype=float)
    site_X = np.load(cache_file(SITE_DATA_CACHE, cache_dir=cache_dir))
    shared_rows = shared_X.shape[0]

    combined_X = normalize_rows(np.concatenate((shared_X, site_X), axis=0))
    rng = make_rng(random_seed)
    Y = initialize_site_embedding(shared_Y, combined_X.shape[0], rng)
    logger.info(
        "Computing affinities for %d reference + %d site points",
        shared_rows,
        site_X.shape[0],
    )
    P = joint_probabilities(combined_X, int(initial_dims), float(perplexity))

    np.save(cache_file(P_CACHE, cache_dir=cache_dir), P)
    np.savez(
        cache_file(OPTIMIZER_CACHE, cache_dir=cache_dir),
        Y=Y,
        iY=np.zeros_like(Y),
        gains=np.ones_like(Y),
    )

    state.shared_rows = shared_rows
    return with_state(
        SiteUpdate(
            phase=PHASE_INITIALIZE,
            n_samples=state.n_samples,
            n_features=state.n_features,
        ),
        state,
    )


def _optimize(message, state, cache_dir, logger, max_iterations):
    """Run one gradient step on combined data (``local_2``)."""
    iteration = message.iteration
    shared_rows = state.shared_rows
    p_path = cache_file(P_CACHE, cache_dir=cache_dir)
    optimizer_path = cache_file(OPTIMIZER_CACHE, cache_dir=cache_dir)

    P = np.load(p_path)
    with np.load(optimizer_path) as cached:
        Y = cached["Y"]
        iY = cached["iY"]
        gains = cached["gains"]

    # Use the averaged shared embedding, then remove the global mean.
    Y[:shared_rows, :] = message.shared_Y
    Y = Y - np.asarray(message.average, dtype=float)

    _, iY, gains, new_P, error = site_gradient_step(
        Y, iY, gains, P, iteration, message.error
    )
    # Sites move only their own points; shared points are updated centrally
    # from the averaged site updates.
    Y[shared_rows:, :] = Y[shared_rows:, :] + iY[shared_rows:, :]

    if new_P is not P:
        np.save(p_path, new_P)
    np.savez(optimizer_path, Y=Y, iY=iY, gains=gains)

    if (iteration + 1) % 100 == 0:
        logger.info("Site iteration %d: error %.6f", iteration + 1, error)

    update = SiteUpdate(
        phase=PHASE_OPTIMIZE,
        n_samples=state.n_samples,
        n_features=state.n_features,
        iteration=iteration,
        shared_Y=Y[:shared_rows, :],
        shared_iY=iY[:shared_rows, :],
        mean=np.mean(Y, 0).tolist(),
        error=error,
    )
    if iteration == int(max_iterations) - 1:
        update.final_embedding = Y[shared_rows:, :]
        update.final_labels = state.labels
    return update
