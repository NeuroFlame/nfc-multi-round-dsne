"""t-SNE math ported from coinstac-dsne-multishot ``tsneFunctions.py``.

The numerical behavior intentionally matches the original COINSTAC
implementation. The only change is that the per-point Python gradient loop
is replaced by an equivalent matrix expression, which is substantially faster
for the thousands of rounds this computation runs.
"""

from typing import Optional, Tuple

import numpy as np
from threadpoolctl import threadpool_limits

# NVFlare runs computation code in processes forked from a multithreaded
# parent. A multithreaded OpenBLAS pool inherited across that fork can
# deadlock on the first large LAPACK call (observed in ``np.linalg.eig``
# during the reference PCA). Pin BLAS to one thread for this process; the
# matrices here are small enough that this costs little.
threadpool_limits(limits=1, user_api="blas")

INITIAL_MOMENTUM = 0.5
FINAL_MOMENTUM = 0.9
MOMENTUM_SWITCH_ITERATION = 20
LEARNING_RATE = 500
MIN_GAIN = 0.01
EARLY_EXAGGERATION = 8
STOP_EXAGGERATION_ITERATION = 100
EXAGGERATION_REDUCTION = 4
REFERENCE_ITERATIONS = 1000
COST_INTERVAL = 10


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Scale each row by its maximum absolute value, in place.

    This is ``normalize_columns`` from the original code, which despite its
    name normalizes rows.
    """
    for row in range(X.shape[0]):
        peak = np.abs(X[row, :]).max()
        if peak != 0:
            X[row, :] = X[row, :] / peak
    return X


def _hbeta(D: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    """Return the entropy and conditional probabilities for one precision."""
    P = np.exp(-D.copy() * beta)
    sum_P = np.sum(P)
    H = np.log(sum_P) + beta * np.sum(D * P) / sum_P
    return H, P / sum_P


def x2p(X: np.ndarray, tol: float = 1e-5, perplexity: float = 30.0) -> np.ndarray:
    """Binary-search Gaussian precisions so each row matches the perplexity."""
    n = X.shape[0]
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    log_U = np.log(perplexity)

    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf
        others = np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))
        Di = D[i, others]
        H, this_P = _hbeta(Di, beta[i])

        H_diff = H - log_U
        tries = 0
        while np.abs(H_diff) > tol and tries < 50:
            if H_diff > 0:
                beta_min = beta[i].copy()
                if np.isinf(beta_max):
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + beta_max) / 2
            else:
                beta_max = beta[i].copy()
                if np.isinf(beta_min):
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + beta_min) / 2
            H, this_P = _hbeta(Di, beta[i])
            H_diff = H - log_U
            tries += 1

        P[i, others] = this_P

    return P


def pca(X: np.ndarray, no_dims: int) -> np.ndarray:
    """Project X onto ``no_dims`` eigenvectors of its scatter matrix."""
    n = X.shape[0]
    X = X - np.tile(np.mean(X, 0), (n, 1))
    _, M = np.linalg.eig(np.dot(X.T, X))
    return np.dot(X, M[:, 0:no_dims])


def joint_probabilities(
    X: np.ndarray, initial_dims: int, perplexity: float
) -> np.ndarray:
    """Return the exaggerated symmetric affinity matrix used by both sides."""
    X = pca(X, initial_dims).real
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * EARLY_EXAGGERATION
    return np.maximum(P, 1e-12)


def _student_t(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return unnormalized Student-t kernel values and normalized Q."""
    n = Y.shape[0]
    sum_Y = np.sum(np.square(Y), 1)
    num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0
    Q = np.maximum(num / np.sum(num), 1e-12)
    return num, Q


def _gradient(P: np.ndarray, Q: np.ndarray, num: np.ndarray, Y: np.ndarray):
    """Vectorized form of the original per-row gradient loop.

    Original: ``dY[i] = sum_j PQ[j, i] * num[j, i] * (Y[i] - Y[j])``.
    """
    W = (P - Q) * num
    return W.sum(axis=0)[:, None] * Y - W.T @ Y


def _momentum(iteration: int) -> float:
    if iteration < MOMENTUM_SWITCH_ITERATION:
        return INITIAL_MOMENTUM
    return FINAL_MOMENTUM


def _update_gains(gains: np.ndarray, dY: np.ndarray, iY: np.ndarray) -> np.ndarray:
    gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * (
        (dY > 0) == (iY > 0)
    )
    gains[gains < MIN_GAIN] = MIN_GAIN
    return gains


def embed_reference(
    X: np.ndarray,
    Y: np.ndarray,
    initial_dims: int,
    perplexity: float,
) -> np.ndarray:
    """Run full centralized t-SNE on the reference data (``remote_1``)."""
    n, no_dims = Y.shape
    P = joint_probabilities(X, initial_dims, perplexity)
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    for iteration in range(REFERENCE_ITERATIONS):
        num, Q = _student_t(Y)
        dY = _gradient(P, Q, num, Y)
        gains = _update_gains(gains, dY, iY)
        iY = _momentum(iteration) * iY - LEARNING_RATE * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        if iteration == STOP_EXAGGERATION_ITERATION:
            P = P / EXAGGERATION_REDUCTION

    return Y


def initialize_site_embedding(
    shared_Y: np.ndarray,
    n_total: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Draw initial 2-D positions uniformly within the reference embedding range.

    As in the original, every row (shared and local) is redrawn; the shared
    rows are replaced by the central embedding at the first optimize step.
    """
    x_range = np.max(shared_Y[:, 0]) - np.min(shared_Y[:, 0])
    y_range = np.max(shared_Y[:, 1]) - np.min(shared_Y[:, 1])
    Yx = rng.rand(n_total) * x_range + np.min(shared_Y[:, 0])
    Yy = rng.rand(n_total) * y_range + np.min(shared_Y[:, 1])
    return np.c_[Yx, Yy]


def site_gradient_step(
    Y: np.ndarray,
    iY: np.ndarray,
    gains: np.ndarray,
    P: np.ndarray,
    iteration: int,
    error: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One combined-data gradient step (``master_child``).

    Returns ``(dY, iY, gains, P, error)``. The cost is recomputed every
    ``COST_INTERVAL`` iterations; otherwise the incoming error passes through.
    """
    num, Q = _student_t(Y)
    dY = _gradient(P, Q, num, Y)
    gains = _update_gains(gains, dY, iY)
    iY = _momentum(iteration) * iY - LEARNING_RATE * (gains * dY)

    if (iteration + 1) % COST_INTERVAL == 0:
        error = float(np.sum(P * np.log(P / Q)))

    if iteration == STOP_EXAGGERATION_ITERATION:
        P = P / EXAGGERATION_REDUCTION

    return dY, iY, gains, P, error


def make_rng(random_seed: Optional[int]) -> np.random.RandomState:
    """Return a seeded generator, or an entropy-seeded one when seed is None."""
    return np.random.RandomState(random_seed)
