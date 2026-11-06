"""Build the final dSNE-MS outputs written at every site (``remote_4``)."""

import base64
import html
import io
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .types import PHASE_DONE, REFERENCE_SOURCE, GlobalMessage, LocalState  # noqa: E402

PLOT_FILENAME = "tsne_embedding.png"


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

    global_df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": _labels_for_output(labels),
            "source": sources,
        }
    )
    global_df.index.name = "point"

    png_bytes = _render_plot(embedding, labels)
    with open(os.path.join(output_dir, PLOT_FILENAME), "wb") as plot_file:
        plot_file.write(png_bytes)

    summary = {
        "final_error": message.error,
        "n_reference_samples": sources.count(REFERENCE_SOURCE),
        "n_site_samples": {
            source: sources.count(source)
            for source in dict.fromkeys(sources)
            if source != REFERENCE_SOURCE
        },
        "n_local_samples": state.n_samples,
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

    return {
        "global_embedding.csv": global_df,
        "summary.json": summary,
        "index.html": _render_report(summary, png_bytes),
    }


def _labels_for_output(labels: np.ndarray):
    """Write integer-valued labels without a trailing ``.0``."""
    if np.all(np.mod(labels, 1) == 0):
        return labels.astype(int)
    return labels


def _render_plot(embedding: np.ndarray, labels: np.ndarray) -> bytes:
    """Scatter the embedding colored by label, as in the original figure."""
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


def _render_report(summary: dict, png_bytes: bytes) -> str:
    """Return a self-contained HTML report with the embedded plot."""
    image = base64.b64encode(png_bytes).decode("ascii")
    site_rows = "".join(
        f"<tr><td>{html.escape(str(site))}</td><td>{count}</td></tr>"
        for site, count in summary["n_site_samples"].items()
    )
    parameter_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary["parameters"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>dSNE-MS results</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
td, th {{ border: 1px solid #ccc; padding: 0.3rem 0.8rem; text-align: left; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>Decentralized multi-shot t-SNE</h1>
<img src="data:image/png;base64,{image}" alt="t-SNE embedding scatter plot">
<h2>Summary</h2>
<table>
<tr><th>Final KL error (site average)</th><td>{summary["final_error"]:.6f}</td></tr>
<tr><th>Reference samples</th><td>{summary["n_reference_samples"]}</td></tr>
</table>
<h2>Samples per site</h2>
<table><tr><th>Site</th><th>Samples</th></tr>{site_rows}</table>
<h2>Parameters</h2>
<table><tr><th>Name</th><th>Value</th></tr>{parameter_rows}</table>
</body>
</html>
"""
