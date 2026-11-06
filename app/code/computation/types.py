"""Define payload and state values exchanged by the dSNE-MS workflow."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

# Workflow phases carried by every central-to-site message.
PHASE_SETUP = "setup"
PHASE_INITIALIZE = "initialize"
PHASE_OPTIMIZE = "optimize"
PHASE_DONE = "done"

# The original algorithm only supports a two-dimensional embedding: the site
# initialization, the gradient step, and the final plot all assume 2 columns.
SUPPORTED_NO_DIMS = 2

# Safety cap on gradient iterations. The framework's iteration limit is fixed
# when the spec is imported, so the user-facing ``max_iterations`` parameter
# is validated against this value instead.
MAX_SUPPORTED_ITERATIONS = 5000

REFERENCE_SOURCE = "reference"


@dataclass
class GlobalMessage:
    """Central-to-site payload for every round of the workflow.

    ``phase`` selects what each site does with the message:

    - ``setup``: first round; sites only report the shape of their data.
    - ``initialize``: carries the reference data and its embedding so sites can
      build their combined affinity matrix.
    - ``optimize``: carries the averaged shared embedding for gradient step
      ``iteration``.
    - ``done``: carries the final embedding of every point.
    """

    phase: str
    iteration: int = 0
    # An ``ArtifactRef`` at runtime. Left untyped because the framework's typed
    # deserializer rejects an already-materialized ArtifactRef in a typed field.
    shared_X: Any = None
    shared_Y: Optional[np.ndarray] = None
    average: List[float] = field(default_factory=lambda: [0.0, 0.0])
    error: float = 0.0
    embedding: Optional[np.ndarray] = None
    embedding_labels: Optional[np.ndarray] = None
    embedding_sources: Optional[List[str]] = None


@dataclass
class SiteUpdate:
    """Site-to-central payload for every round of the workflow.

    ``phase`` and ``iteration`` echo the message the site just processed so
    central can confirm every site is in step.
    """

    phase: str
    n_samples: int
    n_features: int
    iteration: int = 0
    shared_Y: Optional[np.ndarray] = None
    shared_iY: Optional[np.ndarray] = None
    mean: Optional[List[float]] = None
    error: float = 0.0
    final_embedding: Optional[np.ndarray] = None
    final_labels: Optional[np.ndarray] = None


@dataclass
class LocalState:
    """Small per-site values persisted across rounds.

    Large arrays (the affinity matrix and optimizer arrays) live in the
    framework ``cache_dir`` rather than in this JSON-serialized state.
    """

    n_samples: int
    n_features: int
    labels: np.ndarray
    shared_rows: int = 0


@dataclass
class RemoteState:
    """Central values persisted across rounds."""

    shared_labels: np.ndarray
    shared_Y: np.ndarray
