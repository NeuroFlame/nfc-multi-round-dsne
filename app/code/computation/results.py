"""Build the final dSNE-MS outputs written at every site (``remote_4``)."""

import io
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .report import build_report  # noqa: E402
from .types import PHASE_DONE, REFERENCE_SOURCE, GlobalMessage, LocalState  # noqa: E402

PLOT_FILENAME = "tsne_embedding.png"
EMBEDDING_FILENAME = "global_embedding.csv"
SUMMARY_FILENAME = "summary.json"
AXIS_NAMES = ("x", "y", "z")


def build_outputs(
    message: GlobalMessage,
    state: LocalState,
    *,
    output_dir: str,
    parameters: dict,
):
    """Write the combined embedding, this site's embedding, a plot and report."""
    if message.phase != PHASE_DONE or message.embedding is None:
        raise ValueError(
            "The workflow stopped before the final embedding was assembled; "
            "check that max_iterations does not exceed the supported limit"
        )

    embedding = np.asarray(message.embedding, dtype=float)
    labels = np.asarray(message.embedding_labels, dtype=float)
    sources = list(message.embedding_sources)
    site_names = dict(message.site_names or {})
    own_name = site_names.get(state.token)
    if own_name is None:
        raise ValueError("This site's identity was not found in the final message")
    participating = list(site_names.values())
    error_history = [list(entry) for entry in (message.error_history or [])]

    columns = {
        axis: embedding[:, index]
        for index, axis in enumerate(AXIS_NAMES[: embedding.shape[1]])
    }
    columns["label"] = _labels_for_output(labels)
    columns["source"] = sources
    global_df = pd.DataFrame(columns)
    global_df.index.name = "point"

    png_bytes = _render_plot(embedding, labels)
    with open(os.path.join(output_dir, PLOT_FILENAME), "wb") as plot_file:
        plot_file.write(png_bytes)

    summary = {
        "site": own_name,
        "final_error": message.error,
        "n_reference_samples": sources.count(REFERENCE_SOURCE),
        "n_site_samples": {
            source: sources.count(source)
            for source in dict.fromkeys(sources)
            if source != REFERENCE_SOURCE
        },
        "n_local_samples": state.n_samples,
        "kl_error_history": error_history,
        "parameters": {
            key: parameters.get(key)
            for key in (
                "no_dims",
                "initial_dims",
                "perplexity",
                "max_iterations",
                "random_seed",
            )
            if key in parameters
        },
    }

    report = build_report(
        embedding=embedding,
        labels=labels,
        sources=sources,
        own_name=own_name,
        site_names=participating,
        error_history=error_history,
        final_error=message.error,
        parameters=parameters,
        output_files={
            EMBEDDING_FILENAME: "Coordinates, label and source of every point",
            PLOT_FILENAME: (
                "Static image of the embedding colored by label"
                + (" (one fixed viewing angle)" if embedding.shape[1] == 3 else "")
            ),
            SUMMARY_FILENAME: "Run summary, KL error history and parameters",
        },
    )

    return {
        EMBEDDING_FILENAME: global_df,
        SUMMARY_FILENAME: summary,
        "index.html": report,
    }


def _labels_for_output(labels: np.ndarray):
    """Write integer-valued labels without a trailing ``.0``."""
    if np.all(np.mod(labels, 1) == 0):
        return labels.astype(int)
    return labels


def _render_plot(embedding: np.ndarray, labels: np.ndarray) -> bytes:
    """Scatter the embedding colored by label, as in the original figure.

    A 3-D embedding is drawn from one fixed viewing angle; the HTML report has
    the rotatable view.
    """
    if embedding.shape[1] == 3:
        figure = plt.figure(figsize=(8, 6))
        axes = figure.add_subplot(projection="3d")
        scatter = axes.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, c=labels
        )
        axes.view_init(elev=25, azim=35)
    else:
        figure, axes = plt.subplots(figsize=(8, 6))
        scatter = axes.scatter(embedding[:, 0], embedding[:, 1], 5, labels)
    # num=None gives one legend entry per distinct label value.
    handles, legend_labels = scatter.legend_elements(num=None)
    axes.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.15, 0.75),
        title="Label",
    )
    axes.set_title("Decentralized multi-shot t-SNE embedding")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()
