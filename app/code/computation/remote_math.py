"""Central aggregation math for dSNE-MS (ported from ``remote.py``)."""

import os
from typing import Dict, Optional

import numpy as np
from framework import artifact, with_state

from .inputs import load_reference_data
from .tsne import embed_reference, make_rng, normalize_rows
from .types import (
    MAX_SUPPORTED_ITERATIONS,
    PHASE_DONE,
    PHASE_INITIALIZE,
    PHASE_OPTIMIZE,
    PHASE_SETUP,
    REFERENCE_SOURCE,
    SUPPORTED_NO_DIMS,
    GlobalMessage,
    RemoteState,
    SiteUpdate,
)

SHARED_X_ARTIFACT = "shared_X.npy"


def validate_parameters(no_dims, initial_dims, perplexity, max_iterations, n_ref):
    """Reject parameter values the algorithm cannot run with."""
    if no_dims != SUPPORTED_NO_DIMS:
        raise ValueError(
            f"no_dims must be {SUPPORTED_NO_DIMS}; this algorithm only supports "
            "two-dimensional embeddings"
        )
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or not 1 <= max_iterations <= MAX_SUPPORTED_ITERATIONS
    ):
        raise ValueError(
            f"max_iterations must be an integer from 1 to {MAX_SUPPORTED_ITERATIONS}"
        )
    if isinstance(initial_dims, bool) or not isinstance(initial_dims, int):
        raise ValueError("initial_dims must be an integer")
    if initial_dims < no_dims:
        raise ValueError("initial_dims must be at least no_dims")
    if not isinstance(perplexity, (int, float)) or perplexity <= 0:
        raise ValueError("perplexity must be a positive number")
    if perplexity >= n_ref:
        raise ValueError(
            f"perplexity ({perplexity}) must be smaller than the number of "
            f"reference samples ({n_ref})"
        )


def aggregate_site_round(
    site_updates: Dict[str, SiteUpdate],
    state: Optional[RemoteState] = None,
    *,
    artifact_dir: str,
    logger,
    no_dims: int = 2,
    initial_dims: int = 30,
    perplexity: float = 30,
    max_iterations: int = 1000,
    random_seed: Optional[int] = None,
):
    """Advance the central side of the workflow by one round."""
    phases = {update.phase for update in site_updates.values()}
    if len(phases) != 1:
        raise ValueError(f"Sites reported different workflow phases: {sorted(phases)}")
    phase = phases.pop()

    if phase == PHASE_SETUP:
        return _embed_reference(
            site_updates,
            artifact_dir,
            logger,
            no_dims,
            initial_dims,
            perplexity,
            max_iterations,
            random_seed,
        )

    if state is None:
        raise ValueError("Central state is missing after the setup round")

    if phase == PHASE_INITIALIZE:
        # Sites finished initialization: start optimization (``remote_2``).
        return GlobalMessage(
            phase=PHASE_OPTIMIZE,
            iteration=0,
            shared_Y=state.shared_Y,
            average=[0.0, 0.0],
            error=0.0,
        )

    if phase == PHASE_OPTIMIZE:
        return _average_site_updates(site_updates, state, logger)
    raise ValueError(f"Unexpected workflow phase from sites: {phase!r}")


def _embed_reference(
    site_updates,
    artifact_dir,
    logger,
    no_dims,
    initial_dims,
    perplexity,
    max_iterations,
    random_seed,
):
    """Embed the reference data centrally and share it (``remote_1``)."""
    shared_X, shared_labels = load_reference_data()
    validate_parameters(
        no_dims, initial_dims, perplexity, max_iterations, shared_X.shape[0]
    )
    if initial_dims > shared_X.shape[1]:
        raise ValueError(
            f"initial_dims ({initial_dims}) exceeds the number of features "
            f"({shared_X.shape[1]})"
        )
    for site_name, update in site_updates.items():
        if update.n_features != shared_X.shape[1]:
            raise ValueError(
                f"Site '{site_name}' data has {update.n_features} features but "
                f"the reference data has {shared_X.shape[1]}"
            )

    shared_X = normalize_rows(shared_X)
    rng = make_rng(random_seed)
    init_Y = rng.randn(shared_X.shape[0], no_dims)
    logger.info("Embedding %d reference samples", shared_X.shape[0])
    shared_Y = embed_reference(shared_X, init_Y, initial_dims, float(perplexity))

    shared_X_path = os.path.join(artifact_dir, SHARED_X_ARTIFACT)
    np.save(shared_X_path, shared_X)
    return with_state(
        GlobalMessage(
            phase=PHASE_INITIALIZE,
            shared_X=artifact(
                SHARED_X_ARTIFACT,
                shared_X_path,
                "application/x-neuroflame-numpy-array",
            ),
            shared_Y=shared_Y,
        ),
        RemoteState(shared_labels=shared_labels, shared_Y=shared_Y),
    )


def _average_site_updates(site_updates, state, logger):
    """Average site updates into the next shared embedding (``remote_3``)."""
    updates = list(site_updates.values())
    iterations = {update.iteration for update in updates}
    if len(iterations) != 1:
        raise ValueError(f"Sites reported different iterations: {sorted(iterations)}")
    iteration = iterations.pop()

    average = np.mean([update.mean for update in updates], axis=0)
    error = float(np.mean([update.error for update in updates]))
    mean_Y = np.mean([update.shared_Y for update in updates], axis=0)
    mean_iY = np.mean([update.shared_iY for update in updates], axis=0)
    shared_Y = mean_Y + mean_iY

    finished = [update.final_embedding is not None for update in updates]
    if any(finished) and not all(finished):
        raise ValueError("Sites disagree about the final iteration")

    if not all(finished):
        return GlobalMessage(
            phase=PHASE_OPTIMIZE,
            iteration=iteration + 1,
            shared_Y=shared_Y,
            average=average.tolist(),
            error=error,
        )

    logger.info("Optimization finished: error %.6f", error)
    embedding = [shared_Y]
    labels = [state.shared_labels]
    sources = [REFERENCE_SOURCE] * shared_Y.shape[0]
    for site_name, update in site_updates.items():
        embedding.append(np.asarray(update.final_embedding, dtype=float))
        labels.append(np.asarray(update.final_labels, dtype=float))
        sources.extend([site_name] * update.final_embedding.shape[0])

    return GlobalMessage(
        phase=PHASE_DONE,
        shared_Y=shared_Y,
        average=average.tolist(),
        error=error,
        embedding=np.vstack(embedding),
        embedding_labels=np.concatenate(labels),
        embedding_sources=sources,
    )


def is_complete(message: GlobalMessage) -> bool:
    """Stop the iterative workflow once the final embedding is assembled."""
    return message.phase == PHASE_DONE
