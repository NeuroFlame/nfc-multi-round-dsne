"""Load site-local and reference inputs for dSNE-MS."""

import os

import numpy as np
from framework import cache_file, with_state

from .types import PHASE_SETUP, GlobalMessage, LocalState

REFERENCE_DATA_DIR = os.path.join(os.path.dirname(__file__), "reference_data")
REFERENCE_DATA_FILE = os.path.join(REFERENCE_DATA_DIR, "reference_X.txt")
REFERENCE_LABELS_FILE = os.path.join(REFERENCE_DATA_DIR, "reference_labels.txt")

SITE_DATA_CACHE = "site_X.npy"


def load_matrix(path: str) -> np.ndarray:
    """Load a numeric whitespace- or comma-delimited text file as a 2-D array."""
    delimiter = "," if path.lower().endswith(".csv") else None
    return np.loadtxt(path, delimiter=delimiter, ndmin=2, dtype=float)


def load_labels(path: str) -> np.ndarray:
    """Load a numeric label file with one label per row."""
    delimiter = "," if path.lower().endswith(".csv") else None
    labels = np.loadtxt(path, delimiter=delimiter, ndmin=1, dtype=float)
    if labels.ndim != 1:
        raise ValueError(
            f"Label file '{os.path.basename(path)}' must contain exactly one "
            f"column, found shape {labels.shape}"
        )
    return labels


def _site_file(data_dir: str, filename: str, parameter_name: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Parameter '{parameter_name}' must be a filename")
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Site input '{filename}' (parameter '{parameter_name}') was not "
            "found in the site data directory"
        )
    return path


def load_site_inputs(
    data_dir: str,
    cache_dir: str,
    site_data: str = "site_data.txt",
    site_label: str = "site_label.txt",
):
    """Load site data once, cache it, and start the workflow in setup phase."""
    X = load_matrix(_site_file(data_dir, site_data, "site_data"))
    labels = load_labels(_site_file(data_dir, site_label, "site_label"))

    if X.shape[0] == 0:
        raise ValueError("Site data file contains no rows")
    if labels.shape[0] != X.shape[0]:
        raise ValueError(
            f"Site data has {X.shape[0]} rows but the label file has "
            f"{labels.shape[0]} labels"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("Site data contains missing or non-finite values")

    np.save(cache_file(SITE_DATA_CACHE, cache_dir=cache_dir), X)
    return with_state(
        GlobalMessage(phase=PHASE_SETUP),
        LocalState(n_samples=X.shape[0], n_features=X.shape[1], labels=labels),
    )


def load_reference_data():
    """Return the reference (shared) data and labels bundled with the image."""
    X = load_matrix(REFERENCE_DATA_FILE)
    labels = load_labels(REFERENCE_LABELS_FILE)
    if labels.shape[0] != X.shape[0]:
        raise ValueError("Reference data and reference labels have different lengths")
    return X, labels
