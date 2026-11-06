"""Build each site's self-contained HTML results page.

The page follows the shared NeuroFLAME computation report design (header with
chips and site pills, section navigation, cards, stat tables, theme toggle and
inline SVG charts) used by the sibling computations.
"""

import json
import math
from html import escape
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .tsne import STOP_EXAGGERATION_ITERATION
from .types import REFERENCE_SOURCE

# The categorical palette validates all-pairs for three slots only, so the
# label view colors labels directly only up to this many; beyond it, labels
# are shown as small multiples instead.
MAX_COLORED_LABELS = 3
MAX_FACET_LABELS = 12
MAX_FACET_POINTS = 60000
NEIGHBORS = 5
MAX_CATEGORICAL_LABELS = 20

# Default 3-D viewing angle (radians), shared by Python and the page script.
DEFAULT_YAW = 0.6
DEFAULT_PITCH = 0.45

GROUP_THIS_SITE = 1
GROUP_REFERENCE = 2
GROUP_OTHER_SITES = 3


def build_report(
    *,
    embedding: np.ndarray,
    labels: np.ndarray,
    sources: Sequence[str],
    own_name: str,
    site_names: List[str],
    error_history: Sequence[Sequence[float]],
    final_error: float,
    parameters: Dict[str, Any],
    output_files: Dict[str, str],
) -> str:
    """Return the HTML results page for one site.

    Args:
        embedding: Final 2-D coordinates of every point (reference then sites).
        labels: Label of every point.
        sources: ``reference`` or the site display name for every point.
        own_name: This site's display name.
        site_names: Display names of all participating sites.
        error_history: ``[iteration, KL error]`` pairs recorded during the run.
        final_error: Site-averaged KL error at the last iteration.
        parameters: Computation parameters.
        output_files: Output filename -> description, for download links.

    Returns:
        A self-contained HTML document.
    """
    sources = list(sources)
    categorical = _is_categorical(labels)
    label_values = _ordered_labels(labels) if categorical else []
    n_reference = sources.count(REFERENCE_SOURCE)
    n_own = sources.count(own_name)
    agreement = _neighbor_agreement(embedding, labels) if categorical else None

    header = _header(
        own_name=own_name,
        site_names=site_names,
        chips=[
            ("Sites", str(len(site_names))),
            ("Points", str(len(sources))),
            ("This site", str(n_own)),
            ("Reference", str(n_reference)),
            ("Dimensions", f"{embedding.shape[1]}-D"),
            ("Iterations", str(parameters.get("max_iterations", "—"))),
            ("Perplexity", _fmt(parameters.get("perplexity"), 4)),
            ("Final KL", _fmt(final_error, 3)),
        ],
    )
    sections = [
        (
            "embedding",
            "Embedding",
            _embedding_section(
                embedding, labels, sources, own_name, label_values, categorical
            ),
        ),
    ]
    if categorical and len(label_values) > MAX_COLORED_LABELS:
        sections.append(
            (
                "labels",
                "Embedding by label",
                _label_facets(embedding, labels, label_values),
            )
        )
    sections += [
        ("convergence", "Convergence", _convergence_section(error_history)),
        (
            "agreement",
            "Neighbor label agreement",
            _agreement_section(sources, own_name, site_names, agreement),
        ),
        ("settings", "Run settings", _settings(parameters)),
        ("output", "Output", _output(output_files)),
    ]
    nav = "".join(
        f'<a class="nav-link" href="#sec-{slug}">{escape(title)}</a>'
        for slug, title, _ in sections
    )
    body = "".join(
        f'<section class="section" id="sec-{slug}">'
        f'<h2 class="section-title">{escape(title)}</h2>{content}</section>'
        for slug, title, content in sections
    )
    return _page(
        f'{header}<nav class="nav">{nav}</nav><main class="container">{body}</main>'
        '<div class="tooltip" id="tooltip" role="status" hidden></div>'
    )


# ── Sections ────────────────────────────────────────────────────────────────


def _header(own_name: str, site_names: List[str], chips) -> str:
    chip_html = "".join(
        f'<div class="chip">{escape(label)} <b>{escape(value)}</b></div>'
        for label, value in chips
    )
    pills = "".join(
        f'<span class="pill{" pill-own" if name == own_name else ""}">'
        f"{escape(name)}</span>"
        for name in site_names
    )
    return (
        '<header class="page-header">'
        '<button class="theme-toggle" type="button" onclick="toggleTheme()">'
        "Toggle theme</button>"
        "<h1>Decentralized Multi-Shot t-SNE</h1>"
        f"<p>Results for site <b>{escape(own_name)}</b></p>"
        f'<div class="chips">{chip_html}</div>'
        f'<div class="pills">{pills}</div>'
        "</header>"
    )


def _embedding_section(embedding, labels, sources, own_name, label_values, categorical):
    color_labels = categorical and len(label_values) <= MAX_COLORED_LABELS
    label_slot = {value: index + 1 for index, value in enumerate(label_values)}
    groups = [_group(source, own_name) for source in sources]

    is_3d = embedding.shape[1] == 3
    if is_3d:
        unit = _unit_ball(embedding)
        _, _, depth = _rotate(unit, DEFAULT_YAW, DEFAULT_PITCH)
        order = list(np.argsort(depth))  # far points first
    else:
        # Draw reference first and this site last so this site stays on top.
        draw_rank = {GROUP_REFERENCE: 0, GROUP_OTHER_SITES: 1, GROUP_THIS_SITE: 2}
        order = sorted(range(len(sources)), key=lambda i: draw_rank[groups[i]])
    points = []
    for i in order:
        points.append(
            {
                "x": float(embedding[i, 0]),
                "y": float(embedding[i, 1]),
                "u": [round(float(v), 4) for v in unit[i]] if is_3d else None,
                "group": groups[i],
                "slot": label_slot.get(labels[i], 0) if color_labels else 0,
                "source": sources[i],
                "label": _label_text(labels[i]),
            }
        )

    mode = "label" if color_labels else "source"
    controls = ""
    if color_labels:
        controls = (
            '<div class="segmented" role="group" aria-label="Color points by">'
            '<span class="segmented-label">Color by</span>'
            '<button type="button" class="seg-btn active" data-mode="label" '
            'aria-pressed="true">Label</button>'
            '<button type="button" class="seg-btn" data-mode="source" '
            'aria-pressed="false">Source</button></div>'
        )
    if is_3d:
        controls += (
            '<div class="segmented view-controls">'
            '<span class="segmented-label">Drag to rotate, or focus the plot and '
            "use the arrow keys</span>"
            '<button type="button" class="seg-btn reset-view">Reset view</button>'
            "</div>"
        )
    label_legend = "".join(
        f'<span class="legend-item"><span class="swatch s{label_slot[value]}">'
        f"</span>Label {escape(_label_text(value))}</span>"
        for value in label_values
    )
    source_legend = (
        f'<span class="legend-item"><span class="swatch s{GROUP_THIS_SITE}"></span>'
        f"This site ({escape(own_name)})</span>"
        f'<span class="legend-item"><span class="swatch s{GROUP_REFERENCE}"></span>'
        "Reference</span>"
        f'<span class="legend-item"><span class="swatch s{GROUP_OTHER_SITES}"></span>'
        "Other sites</span>"
    )
    legends = (
        f'<div class="legend" data-legend="label"'
        f"{'' if mode == 'label' else ' hidden'}>{label_legend}</div>"
        if color_labels
        else ""
    ) + (
        f'<div class="legend" data-legend="source"'
        f"{'' if mode == 'source' else ' hidden'}>{source_legend}</div>"
    )
    note = (
        '<p class="muted">Each point is one sample. Nearby points have similar '
        "features; distances between clusters and the axis values carry no "
        "meaning in t-SNE. Hover a point for its source and label.</p>"
    )
    if is_3d:
        note += (
            '<p class="muted">This is a three-dimensional embedding. Points that '
            "look close may be far apart in depth, so rotate the view before "
            "reading proximity. Nearer points are drawn slightly larger.</p>"
        )
    if not categorical:
        note += (
            '<p class="note">Labels are not categorical (more than '
            f"{MAX_CATEGORICAL_LABELS} distinct or non-integer values), so points "
            "are colored by source only.</p>"
        )
    elif not color_labels:
        note += (
            f'<p class="note">There are {len(label_values)} labels, more than can '
            "be told apart by color, so points are colored by source here and "
            'each label is shown separately under <a href="#sec-labels">'
            "Embedding by label</a>.</p>"
        )
    chart = _scatter_3d(points, mode) if is_3d else _scatter(points, mode)
    return f'<div class="card">{controls}{legends}{chart}</div>{note}'


def _label_facets(embedding, labels, label_values) -> str:
    shown = label_values[:MAX_FACET_LABELS]
    if len(shown) * len(labels) > MAX_FACET_POINTS:
        return (
            '<p class="note">Too many points to draw one panel per label; use '
            "global_embedding.csv to explore labels individually.</p>"
        )
    view_note = ""
    if embedding.shape[1] == 3:
        sx, sy, _ = _rotate(_unit_ball(embedding), DEFAULT_YAW, DEFAULT_PITCH)
        embedding = np.column_stack([sx, sy])
        view_note = " Panels show the embedding from the default viewing angle."
    bounds = _bounds(embedding)
    panels = []
    for value in shown:
        mask = labels == value
        panels.append(
            '<div class="card facet">'
            f"<h3>Label {escape(_label_text(value))} "
            f'<span class="muted-inline">{int(mask.sum())} points</span></h3>'
            f"{_facet_svg(embedding, mask, bounds, _label_text(value))}</div>"
        )
    more = ""
    if len(label_values) > MAX_FACET_LABELS:
        more = (
            f'<p class="muted">Showing the {MAX_FACET_LABELS} most frequent of '
            f"{len(label_values)} labels.</p>"
        )
    return (
        '<p class="muted">Each panel highlights one label against all other '
        f"points in gray.{view_note}</p>"
        f'<div class="facet-grid">{"".join(panels)}</div>{more}'
    )


def _convergence_section(error_history) -> str:
    history = [(float(i), float(e)) for i, e in error_history if math.isfinite(e)]
    if len(history) < 2:
        return (
            '<p class="muted">Too few iterations to plot convergence. The cost is '
            "recorded every 10 iterations.</p>"
        )
    intro = (
        '<p class="muted">Kullback–Leibler divergence between the high- and '
        "low-dimensional similarities, averaged over sites and recorded every 10 "
        "iterations. Lower is better.</p>"
    )
    # The cost during early exaggeration is computed on inflated affinities
    # and would dominate the axis, so it is left out once later values exist.
    after = [h for h in history if h[0] > STOP_EXAGGERATION_ITERATION]
    if len(after) >= 2:
        history = after
        intro += (
            '<p class="muted">The first '
            f"{STOP_EXAGGERATION_ITERATION} iterations (early exaggeration) are not "
            "shown: their cost is computed on exaggerated similarities and is not "
            "comparable.</p>"
        )
    else:
        intro += (
            '<p class="muted">All iterations shown are in the early exaggeration '
            "phase, so their cost is inflated relative to a full run.</p>"
        )
    return f'{intro}<div class="card">{_line_chart(history)}</div>'


def _agreement_section(sources, own_name, site_names, agreement) -> str:
    if agreement is None:
        return '<p class="muted">Not computed: labels are not categorical.</p>'
    intro = (
        '<p class="muted">Share of points whose label matches the majority label '
        f"of their {NEIGHBORS} nearest neighbors in the embedding. Higher means "
        "labels form cleaner clusters. This is a descriptive check of the "
        "embedding, not a test of the labels.</p>"
    )
    sources_array = np.asarray(sources)
    rows = []
    for name in [REFERENCE_SOURCE] + list(site_names):
        mask = sources_array == name
        if not mask.any():
            continue
        display = "Reference" if name == REFERENCE_SOURCE else name
        own = name == own_name
        rows.append(
            f"<tr{' class=row-own' if own else ''}><td>{escape(display)}"
            f"{' (this site)' if own else ''}</td><td>{int(mask.sum())}</td>"
            f"<td>{_percent(agreement[mask].mean())}</td></tr>"
        )
    rows.append(
        f"<tr class=row-total><td>All points</td><td>{len(sources)}</td>"
        f"<td>{_percent(agreement.mean())}</td></tr>"
    )
    table = _table(["Source", "Points", "Agreement"], "".join(rows))
    return f'{intro}<div class="card">{table}</div>'


def _settings(parameters: Dict[str, Any]) -> str:
    settings = [
        ("Site data file", parameters.get("site_data", "site_data.txt")),
        ("Site label file", parameters.get("site_label", "site_label.txt")),
        ("Embedding dimensions", parameters.get("no_dims", 2)),
        ("PCA dimensions", parameters.get("initial_dims", 30)),
        ("Perplexity", parameters.get("perplexity", 30)),
        ("Iterations", parameters.get("max_iterations", 1000)),
        ("Random seed", parameters.get("random_seed", "not set (random)")),
    ]
    rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(str(value))}</td></tr>"
        for label, value in settings
    )
    return f'<div class="card">{_table(["Setting", "Value"], rows, "kv")}</div>'


def _output(output_files: Dict[str, str]) -> str:
    items = "".join(
        f'<li><a class="download" href="{escape(name)}" download>{escape(name)}</a>'
        f' <span class="muted-inline">{escape(description)}</span></li>'
        for name, description in output_files.items()
    )
    return (
        f'<div class="card"><ul class="downloads">{items}</ul></div>'
        '<p class="note">The final coordinates and labels of every site\'s points '
        "are shared with all participating sites to build this combined "
        "embedding. Raw site data never leaves its site.</p>"
    )


# ── Charts ──────────────────────────────────────────────────────────────────

_SCATTER_SIZE = 560
_SCATTER_PAD = 16


def _bounds(embedding: np.ndarray) -> Tuple[float, float, float]:
    """Return center x, center y and half-span for an equal-aspect square."""
    finite = embedding[np.all(np.isfinite(embedding), axis=1)]
    if finite.size == 0:
        return 0.0, 0.0, 1.0
    low = finite.min(axis=0)
    high = finite.max(axis=0)
    half = max(float(np.max(high - low)) / 2, 1e-9) * 1.04
    center = (low + high) / 2
    return float(center[0]), float(center[1]), half


def _project(x, y, bounds, size, pad):
    cx, cy, half = bounds
    scale = (size - 2 * pad) / (2 * half)
    return pad + (x - cx + half) * scale, pad + (cy + half - y) * scale


def _scatter(points, mode: str) -> str:
    """Return an equal-aspect SVG scatter plot with hover data."""
    size, pad = _SCATTER_SIZE, _SCATTER_PAD
    bounds = _bounds(np.array([[p["x"], p["y"]] for p in points]))
    circles = []
    for index, point in enumerate(points):
        px, py = _project(point["x"], point["y"], bounds, size, pad)
        circles.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" data-i="{index}" '
            f'data-g="{point["group"]}" data-l="{point["slot"]}"/>'
        )
    hover = [
        {
            "source": "Reference" if p["source"] == REFERENCE_SOURCE else p["source"],
            "label": p["label"],
        }
        for p in points
    ]
    return (
        f'<div class="scatter-wrap"><svg class="chart scatter by-{mode}" '
        f'viewBox="0 0 {size} {size}" role="img" '
        'aria-label="t-SNE embedding scatter plot">'
        f'<rect class="plot-frame" x="0.5" y="0.5" width="{size - 1}" '
        f'height="{size - 1}" rx="10"/>'
        f'<g class="points">{"".join(circles)}</g></svg>'
        f'<script type="application/json" class="hover-data">'
        f"{_json_for_script(hover)}</script></div>"
    )


def _unit_ball(embedding: np.ndarray) -> np.ndarray:
    """Center a 3-D embedding and scale it to fit inside the unit sphere."""
    finite = np.all(np.isfinite(embedding), axis=1)
    if not finite.any():
        return np.zeros_like(embedding)
    low = embedding[finite].min(axis=0)
    high = embedding[finite].max(axis=0)
    centered = embedding - (low + high) / 2
    radius = float(np.max(np.linalg.norm(centered[finite], axis=1))) or 1.0
    return np.where(finite[:, None], centered / radius, 0.0)


def _rotate(unit: np.ndarray, yaw: float, pitch: float):
    """Orthographic view: screen x, screen y (up) and depth (toward viewer).

    The page script uses the same formulas when the viewer rotates the plot.
    """
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    x1 = x * math.cos(yaw) + z * math.sin(yaw)
    z1 = -x * math.sin(yaw) + z * math.cos(yaw)
    y2 = y * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y * math.sin(pitch) + z1 * math.cos(pitch)
    return x1, y2, z2


def _scatter_3d(points, mode: str) -> str:
    """Return a rotatable 3-D scatter plot, pre-drawn at the default angle."""
    size, pad = _SCATTER_SIZE, _SCATTER_PAD + 8
    radius = size / 2 - pad
    unit = np.array([p["u"] for p in points])
    sx, sy, depth = _rotate(unit, DEFAULT_YAW, DEFAULT_PITCH)
    circles = []
    for index, point in enumerate(points):
        cx = size / 2 + sx[index] * radius
        cy = size / 2 - sy[index] * radius
        r = _depth_radius(depth[index])
        circles.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.2f}" data-i="{index}" '
            f'data-g="{point["group"]}" data-l="{point["slot"]}"/>'
        )
    corners = _box_corners(unit)
    data = {
        "size": size,
        "radius": radius,
        "yaw": DEFAULT_YAW,
        "pitch": DEFAULT_PITCH,
        "u": [p["u"] for p in points],
        "box": corners.round(4).tolist(),
        "edges": _BOX_EDGES,
    }
    hover = [
        {
            "source": "Reference" if p["source"] == REFERENCE_SOURCE else p["source"],
            "label": p["label"],
        }
        for p in points
    ]
    return (
        f'<div class="scatter-wrap scatter-3d-wrap"><svg class="chart scatter '
        f'scatter-3d by-{mode}" viewBox="0 0 {size} {size}" tabindex="0" '
        'role="img" aria-label="Rotatable 3-D t-SNE embedding scatter plot">'
        f'<rect class="plot-frame" x="0.5" y="0.5" width="{size - 1}" '
        f'height="{size - 1}" rx="10"/>'
        f"{_box_svg(corners, size, radius)}"
        f"{_gizmo(DEFAULT_YAW, DEFAULT_PITCH, size)}"
        f'<g class="points">{"".join(circles)}</g></svg>'
        f'<script type="application/json" class="hover-data">'
        f"{_json_for_script(hover)}</script>"
        f'<script type="application/json" class="view-data">'
        f"{_json_for_script(data)}</script></div>"
    )


# Corner index pairs of the 12 bounding-box edges (corner bits: x, y, z).
_BOX_EDGES = [
    [a, b] for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1
]


def _box_corners(unit: np.ndarray) -> np.ndarray:
    """Return the 8 corners of the points' axis-aligned bounding box."""
    low = unit.min(axis=0)
    high = unit.max(axis=0)
    return np.array(
        [
            [
                high[0] if c & 1 else low[0],
                high[1] if c & 2 else low[1],
                high[2] if c & 4 else low[2],
            ]
            for c in range(8)
        ]
    )


def _box_svg(corners: np.ndarray, size: int, radius: float) -> str:
    """Return the bounding-box wireframe drawn behind the points."""
    sx, sy, _ = _rotate(corners, DEFAULT_YAW, DEFAULT_PITCH)
    px = size / 2 + sx * radius
    py = size / 2 - sy * radius
    lines = "".join(
        f'<line x1="{px[a]:.1f}" y1="{py[a]:.1f}" x2="{px[b]:.1f}" y2="{py[b]:.1f}"/>'
        for a, b in _BOX_EDGES
    )
    return f'<g class="box">{lines}</g>'


def _depth_radius(depth: float) -> float:
    """Marker radius from depth in [-1, 1]: nearer points slightly larger."""
    return 3.4 + 1.2 * (float(depth) + 1) / 2


def _gizmo(yaw: float, pitch: float, size: int) -> str:
    """Return the x/y/z orientation indicator drawn in the plot corner."""
    origin_x, origin_y, length = 40, size - 40, 24
    axes = np.eye(3)
    gx, gy, _ = _rotate(axes, yaw, pitch)
    parts = ['<g class="gizmo">']
    for index, name in enumerate("xyz"):
        ex = origin_x + gx[index] * length
        ey = origin_y - gy[index] * length
        parts.append(
            f'<line data-axis="{index}" x1="{origin_x}" y1="{origin_y}" '
            f'x2="{ex:.1f}" y2="{ey:.1f}"/>'
            f'<text data-axis="{index}" x="{ex + (gx[index] * 8):.1f}" '
            f'y="{ey - (gy[index] * 8) + 4:.1f}">{name}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _facet_svg(embedding, mask, bounds, label_text: str) -> str:
    size, pad = 240, 8
    background, highlight = [], []
    for i in range(embedding.shape[0]):
        px, py = _project(embedding[i, 0], embedding[i, 1], bounds, size, pad)
        if mask[i]:
            highlight.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3"/>')
        else:
            background.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2"/>')
    return (
        f'<svg class="chart facet-plot" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="Points with label {escape(label_text)}">'
        f'<rect class="plot-frame" x="0.5" y="0.5" width="{size - 1}" '
        f'height="{size - 1}" rx="8"/>'
        f'<g class="context">{"".join(background)}</g>'
        f'<g class="highlight">{"".join(highlight)}</g></svg>'
    )


def _line_chart(history: List[Tuple[float, float]]) -> str:
    """Return an SVG line chart of KL error against iteration with a crosshair."""
    width, height = 720, 260
    left, right, top, bottom = 56, 20, 16, 40
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]
    x_low, x_high = min(xs), max(xs)
    y_low, y_high = 0.0, max(ys) * 1.05 or 1.0

    def x(value):
        return left + (value - x_low) / (x_high - x_low) * (width - left - right)

    def y(value):
        return top + (y_high - value) / (y_high - y_low) * (height - top - bottom)

    parts = [
        f'<svg class="chart line-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="KL error by iteration">'
    ]
    for tick in _nice_ticks(y_low, y_high, 4):
        ty = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{ty:.1f}" x2="{width - right}" '
            f'y2="{ty:.1f}"/><text class="tick-y" x="{left - 8}" y="{ty + 4:.1f}">'
            f"{_fmt(tick, 3)}</text>"
        )
    for tick in _nice_ticks(x_low, x_high, 6):
        tx = x(tick)
        parts.append(
            f'<text class="tick" x="{tx:.1f}" y="{height - bottom + 18}">'
            f"{_fmt(tick, 5)}</text>"
        )
    parts.append(
        f'<text class="axis-title" x="{(left + width - right) / 2:.0f}" '
        f'y="{height - 4}" text-anchor="middle">Iteration</text>'
        f'<text class="axis-title" x="14" y="{(top + height - bottom) / 2:.0f}" '
        f'text-anchor="middle" transform="rotate(-90 14 '
        f'{(top + height - bottom) / 2:.0f})">KL error</text>'
    )
    if x_low < STOP_EXAGGERATION_ITERATION < x_high:
        rx = x(STOP_EXAGGERATION_ITERATION)
        parts.append(
            f'<line class="reference" x1="{rx:.1f}" y1="{top}" x2="{rx:.1f}" '
            f'y2="{height - bottom}"/><text class="annotation" x="{rx + 6:.1f}" '
            f'y="{top + 12}">Exaggeration ends</text>'
        )
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(a):.1f},{y(b):.1f}"
        for i, (a, b) in enumerate(history)
    )
    parts.append(f'<path class="series-line" d="{path}"/>')
    parts.append(
        f'<line class="crosshair" x1="0" y1="{top}" x2="0" y2="{height - bottom}" '
        'visibility="hidden"/><circle class="crosshair-dot" r="4" '
        'visibility="hidden"/>'
        f'<rect class="hit-area" x="{left}" y="{top}" '
        f'width="{width - left - right}" height="{height - top - bottom}"/>'
    )
    parts.append("</svg>")
    points = [
        {"px": round(x(a), 1), "py": round(y(b), 1), "iteration": a, "error": b}
        for a, b in history
    ]
    return (
        f'<div class="line-wrap">{"".join(parts)}'
        f'<script type="application/json" class="line-data">'
        f"{_json_for_script(points)}</script></div>"
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _group(source: str, own_name: str) -> int:
    if source == own_name:
        return GROUP_THIS_SITE
    if source == REFERENCE_SOURCE:
        return GROUP_REFERENCE
    return GROUP_OTHER_SITES


def _is_categorical(labels: np.ndarray) -> bool:
    if labels.size == 0 or not np.all(np.isfinite(labels)):
        return False
    return bool(
        np.all(np.mod(labels, 1) == 0)
        and np.unique(labels).size <= MAX_CATEGORICAL_LABELS
    )


def _ordered_labels(labels: np.ndarray) -> list:
    """Return label values, most frequent first (ties by value)."""
    values, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(values)), key=lambda i: (-counts[i], values[i]))
    return [values[i] for i in order]


def _neighbor_agreement(embedding: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per point: does its label match the majority of its nearest neighbors?"""
    n = embedding.shape[0]
    k = min(NEIGHBORS, n - 1)
    if k < 1:
        return np.zeros(n, dtype=bool)
    squared = np.sum(embedding**2, axis=1)
    result = np.zeros(n, dtype=bool)
    for start in range(0, n, 1000):
        stop = min(start + 1000, n)
        block = (
            squared[start:stop, None]
            + squared[None, :]
            - 2 * embedding[start:stop] @ embedding.T
        )
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        neighbors = np.argpartition(block, k - 1, axis=1)[:, :k]
        matches = labels[neighbors] == labels[start:stop, None]
        result[start:stop] = matches.mean(axis=1) > 0.5
    return result


def _label_text(value) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _percent(value: float) -> str:
    return f"{100 * float(value):.0f}%"


def _json_for_script(value) -> str:
    """Serialize JSON safely for embedding inside a <script> element."""
    return (
        json.dumps(value, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _table(headers: List[str], rows_html: str, css_class: str = "") -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    return (
        f'<div class="table-scroll"><table class="stat-table {css_class}">'
        f"<thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )


def _nice_ticks(low: float, high: float, count: int = 5) -> List[float]:
    raw_step = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw_step)
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + 1e-12:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def _fmt(value, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—" if math.isnan(value) else "∞"
    return f"{value:.{digits}g}"


def _page(body: str) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        "<title>dSNE-MS Report</title>"
        f"<style>{_CSS}</style></head><body>{body}<script>{_JS}</script>"
        "</body></html>"
    )


# Shared NeuroFLAME report styles (as in nfc-combatdc / nfc-dpsvm), plus the
# chart and control styles this report needs. Series colors are the validated
# three-slot categorical palette (all-pairs safe for scatter plots) stepped
# separately for the light and dark surfaces.
_CSS = """
:root{--bg:#fff;--bg3:#f1f5f9;--border:#e2e8f0;--border2:#cbd5e1;--text:#0f172a;
--text2:#334155;--text3:#64748b;--header-bg:linear-gradient(135deg,#e0e9ff 0%,#f8fafc 100%);
--chip-bg:#f1f5f9;--chip-b:#4f46e5;--card-bg:#fff;--th-bg:#f8fafc;--accent:#4f46e5;
--muted-dot:#94a3b8;--own-bg:rgba(79,70,229,.12);--note-bg:#f8fafc;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--context-dot:#cbd5e1}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f172a;
--bg3:#1a2640;--border:#334155;--border2:#475569;--text:#e2e8f0;--text2:#cbd5e1;
--text3:#94a3b8;--header-bg:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);
--chip-bg:#1e293b;--chip-b:#a5b4fc;--card-bg:#1e293b;--th-bg:#161f30;--accent:#a5b4fc;
--muted-dot:#64748b;--own-bg:rgba(165,180,252,.15);--note-bg:#162033;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--context-dot:#475569}}
:root[data-theme="dark"]{--bg:#0f172a;--bg3:#1a2640;--border:#334155;--border2:#475569;
--text:#e2e8f0;--text2:#cbd5e1;--text3:#94a3b8;
--header-bg:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);--chip-bg:#1e293b;
--chip-b:#a5b4fc;--card-bg:#1e293b;--th-bg:#161f30;--accent:#a5b4fc;--muted-dot:#64748b;
--own-bg:rgba(165,180,252,.15);--note-bg:#162033;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--context-dot:#475569}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);line-height:1.5}
.page-header{background:var(--header-bg);border-bottom:1px solid var(--border);
padding:2rem 1rem;position:relative}
.page-header>*{max-width:1200px;margin-left:auto;margin-right:auto}
.page-header h1{font-size:1.6rem;font-weight:700;letter-spacing:-.02em;padding-right:8rem}
.page-header p{color:var(--text3);margin-top:.3rem}
.theme-toggle{position:absolute;top:1.5rem;right:1rem;background:var(--card-bg);
border:1px solid var(--border);border-radius:999px;padding:.3rem .8rem;font-size:.78rem;
font-weight:600;color:var(--text2);cursor:pointer}
.chips,.pills{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem}
.chip{background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;
padding:.25rem .75rem;font-size:.78rem;color:var(--text2)}
.chip b{color:var(--chip-b)}
.pill{border:1px solid var(--border2);border-radius:999px;padding:.15rem .6rem;
font-size:.75rem;color:var(--text2)}
.pill-own{background:var(--own-bg);border-color:var(--accent);color:var(--accent);
font-weight:600}
.nav{display:flex;gap:.4rem;flex-wrap:wrap;max-width:1200px;margin:1rem auto 0;
padding:0 1rem}
.nav-link{font-size:.8rem;color:var(--text3);text-decoration:none;padding:.3rem .7rem;
border-radius:8px}
.nav-link:hover{background:var(--bg3);color:var(--text)}
.container{max-width:1200px;margin:0 auto;padding:1.5rem 1rem 3rem}
.section{margin-bottom:2.5rem;scroll-margin-top:1rem}
.section-title{font-size:.8rem;font-weight:700;text-transform:uppercase;
letter-spacing:.08em;color:var(--text3);margin-bottom:1rem;padding-bottom:.5rem;
border-bottom:1px solid var(--border)}
h3{font-size:.9rem;margin-bottom:.6rem}
.muted{color:var(--text3);font-size:.85rem;margin-bottom:1rem}
.muted-inline{color:var(--text3);font-size:.8rem;font-weight:400}
.note{background:var(--note-bg);border:1px solid var(--border);border-radius:10px;
padding:.75rem 1rem;font-size:.88rem;color:var(--text2);margin-bottom:1rem}
.note a{color:var(--accent)}
.card{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;
padding:1rem;margin-bottom:1rem;min-width:0}
.chart{width:100%;height:auto;display:block}
.chart text{fill:var(--text3);font-size:11px}
.chart .axis-title{font-weight:600;fill:var(--text2)}
.chart .tick{text-anchor:middle}
.chart .tick-y{text-anchor:end}
.chart .annotation{fill:var(--text3);font-size:11px}
.chart .grid{stroke:var(--border);stroke-width:1}
.chart .reference{stroke:var(--border2);stroke-width:1.5;stroke-dasharray:4 3}
.chart .plot-frame{fill:none;stroke:var(--border);stroke-width:1}
.table-scroll{overflow-x:auto}
table.stat-table{width:100%;border-collapse:collapse;font-size:.8rem}
.stat-table th{color:var(--text3);font-weight:600;padding:.45rem .75rem;text-align:right;
background:var(--th-bg);white-space:nowrap}
.stat-table td{padding:.4rem .75rem;border-top:1px solid var(--border);text-align:right;
font-variant-numeric:tabular-nums}
.stat-table th:first-child,.stat-table td:first-child{text-align:left}
.stat-table.kv td:last-child{text-align:left}
.stat-table .row-own td{background:var(--own-bg);font-weight:600}
.stat-table .row-total td{font-weight:600}
.download{color:var(--accent);font-weight:600}
.downloads{list-style:none;display:grid;gap:.4rem;font-size:.88rem}
code{font-size:.85em}
.segmented{display:flex;align-items:center;gap:.35rem;flex-wrap:wrap;margin-bottom:.75rem}
.segmented-label{font-size:.78rem;color:var(--text3);margin-right:.25rem}
.seg-btn{background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;
padding:.2rem .75rem;font-size:.78rem;color:var(--text2);cursor:pointer}
.seg-btn.active{background:var(--own-bg);border-color:var(--accent);color:var(--accent);
font-weight:600}
.legend{display:flex;gap:.9rem;flex-wrap:wrap;margin-bottom:.75rem;font-size:.8rem;
color:var(--text2)}
.legend[hidden]{display:none}
.legend-item{display:inline-flex;align-items:center;gap:.4rem}
.swatch{width:10px;height:10px;border-radius:50%;display:inline-block}
.swatch.s1{background:var(--series-1)}.swatch.s2{background:var(--series-2)}
.swatch.s3{background:var(--series-3)}
.scatter-wrap{max-width:620px}
.scatter circle{stroke:var(--card-bg);stroke-width:2;fill:var(--muted-dot)}
.scatter.by-source circle[data-g="1"],.scatter.by-label circle[data-l="1"]{fill:var(--series-1)}
.scatter.by-source circle[data-g="2"],.scatter.by-label circle[data-l="2"]{fill:var(--series-2)}
.scatter.by-source circle[data-g="3"],.scatter.by-label circle[data-l="3"]{fill:var(--series-3)}
.scatter circle.hover{stroke:var(--text);stroke-width:2}
.scatter-3d{cursor:grab;touch-action:none}
.scatter-3d.dragging{cursor:grabbing}
.scatter-3d:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:10px}
.box line{stroke:var(--border);stroke-width:1}
.gizmo line{stroke:var(--text3);stroke-width:1.5}
.gizmo text{fill:var(--text3);font-size:11px;font-weight:600;text-anchor:middle}
.view-controls .segmented-label{margin-right:.5rem}
.facet-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
gap:1rem}
.facet{margin-bottom:0}
.facet-plot .context circle{fill:var(--context-dot)}
.facet-plot .highlight circle{fill:var(--series-1);stroke:var(--card-bg);stroke-width:1.5}
.line-wrap{max-width:760px}
.series-line{fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round}
.crosshair{stroke:var(--border2);stroke-width:1}
.crosshair-dot{fill:var(--accent);stroke:var(--card-bg);stroke-width:2}
.hit-area{fill:transparent}
.tooltip{position:fixed;pointer-events:none;background:var(--card-bg);
border:1px solid var(--border2);border-radius:8px;padding:.35rem .6rem;font-size:.78rem;
color:var(--text);box-shadow:0 4px 14px rgba(15,23,42,.12);z-index:10;
font-variant-numeric:tabular-nums}
.tooltip b{font-weight:600}
@media (max-width:600px){.page-header h1{font-size:1.3rem;padding-right:0}
.theme-toggle{position:static;margin-bottom:.8rem}}
"""

_JS = """
function toggleTheme(){var r=document.documentElement;
var dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&
window.matchMedia('(prefers-color-scheme: dark)').matches);
var next=dark?'light':'dark';r.setAttribute('data-theme',next);
try{localStorage.setItem('dsne-theme',next)}catch(e){}}
(function(){try{var t=localStorage.getItem('dsne-theme');
if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}})();
(function(){
var tip=document.getElementById('tooltip');
function show(lines,ev){tip.textContent='';lines.forEach(function(l,i){
var row=document.createElement('div');if(i===0){var b=document.createElement('b');
b.textContent=l;row.appendChild(b)}else{row.textContent=l}tip.appendChild(row)});
tip.hidden=false;var x=ev.clientX+14,y=ev.clientY+14;
if(x+tip.offsetWidth>window.innerWidth-8)x=ev.clientX-tip.offsetWidth-14;
if(y+tip.offsetHeight>window.innerHeight-8)y=ev.clientY-tip.offsetHeight-14;
tip.style.left=x+'px';tip.style.top=y+'px'}
function hide(){tip.hidden=true}
document.querySelectorAll('.scatter-wrap').forEach(function(wrap){
var svg=wrap.querySelector('svg'),data=JSON.parse(wrap.querySelector('.hover-data').textContent);
var circles=Array.prototype.slice.call(svg.querySelectorAll('.points circle'));
var active=null;
svg.addEventListener('pointermove',function(ev){
if(svg.classList.contains('dragging'))return;
var r=svg.getBoundingClientRect();
var s=svg.viewBox.baseVal.width/r.width,mx=(ev.clientX-r.left)*s,my=(ev.clientY-r.top)*s;
var best=-1,bd=Infinity;for(var i=0;i<circles.length;i++){
var dx=circles[i].cx.baseVal.value-mx,dy=circles[i].cy.baseVal.value-my,
d=dx*dx+dy*dy;if(d<bd){bd=d;best=i}}
if(active)active.classList.remove('hover');
if(best<0||Math.sqrt(bd)>24*s){active=null;hide();return}
active=circles[best];active.classList.add('hover');
var p=data[+active.getAttribute('data-i')];show([p.source,'Label '+p.label],ev)});
svg.addEventListener('pointerleave',function(){if(active)active.classList.remove('hover');
active=null;hide()});
var card=wrap.closest('.card');
card.querySelectorAll('.seg-btn[data-mode]').forEach(function(btn){btn.addEventListener('click',function(){
var mode=btn.getAttribute('data-mode');svg.classList.remove('by-label','by-source');
svg.classList.add('by-'+mode);
card.querySelectorAll('.seg-btn[data-mode]').forEach(function(b){var on=b===btn;
b.classList.toggle('active',on);b.setAttribute('aria-pressed',on?'true':'false')});
card.querySelectorAll('.legend').forEach(function(l){
l.hidden=l.getAttribute('data-legend')!==mode})})})});
document.querySelectorAll('.line-wrap').forEach(function(wrap){
var svg=wrap.querySelector('svg'),data=JSON.parse(wrap.querySelector('.line-data').textContent);
var hair=svg.querySelector('.crosshair'),dot=svg.querySelector('.crosshair-dot');
var area=svg.querySelector('.hit-area');
area.addEventListener('pointermove',function(ev){var r=svg.getBoundingClientRect();
var mx=(ev.clientX-r.left)*svg.viewBox.baseVal.width/r.width;var best=data[0];
data.forEach(function(d){if(Math.abs(d.px-mx)<Math.abs(best.px-mx))best=d});
hair.setAttribute('x1',best.px);hair.setAttribute('x2',best.px);
hair.setAttribute('visibility','visible');dot.setAttribute('cx',best.px);
dot.setAttribute('cy',best.py);dot.setAttribute('visibility','visible');
show(['Iteration '+best.iteration,'KL error '+best.error.toPrecision(4)],ev)});
area.addEventListener('pointerleave',function(){hair.setAttribute('visibility','hidden');
dot.setAttribute('visibility','hidden');hide()})});
document.querySelectorAll('.scatter-3d-wrap').forEach(function(wrap){
var svg=wrap.querySelector('svg'),v=JSON.parse(wrap.querySelector('.view-data').textContent);
var group=svg.querySelector('.points');
var circles=Array.prototype.slice.call(group.querySelectorAll('circle'));
var byIndex=[];circles.forEach(function(c){byIndex[+c.getAttribute('data-i')]=c});
var lines=svg.querySelectorAll('.gizmo line'),labels=svg.querySelectorAll('.gizmo text');
var boxLines=svg.querySelectorAll('.box line');
var yaw=v.yaw,pitch=v.pitch,half=v.size/2,R=v.radius,queued=false,goy=v.size-40;
function rot(p){var cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
var x1=p[0]*cy+p[2]*sy,z1=-p[0]*sy+p[2]*cy;return[x1,p[1]*cp-z1*sp,p[1]*sp+z1*cp]}
function render(){queued=false;var order=[];
for(var i=0;i<v.u.length;i++){var q=rot(v.u[i]),c=byIndex[i];
c.setAttribute('cx',(half+q[0]*R).toFixed(1));c.setAttribute('cy',(half-q[1]*R).toFixed(1));
c.setAttribute('r',(3.4+1.2*(q[2]+1)/2).toFixed(2));order.push([q[2],c])}
order.sort(function(a,b){return a[0]-b[0]});order.forEach(function(o){group.appendChild(o[1])});
var bp=v.box.map(function(c){var q=rot(c);return[half+q[0]*R,half-q[1]*R]});
v.edges.forEach(function(e,k){var l=boxLines[k];l.setAttribute('x1',bp[e[0]][0].toFixed(1));
l.setAttribute('y1',bp[e[0]][1].toFixed(1));l.setAttribute('x2',bp[e[1]][0].toFixed(1));
l.setAttribute('y2',bp[e[1]][1].toFixed(1))});
[[1,0,0],[0,1,0],[0,0,1]].forEach(function(a,k){var q=rot(a),ex=40+q[0]*24,ey=goy-q[1]*24;
lines[k].setAttribute('x2',ex.toFixed(1));lines[k].setAttribute('y2',ey.toFixed(1));
labels[k].setAttribute('x',(ex+q[0]*8).toFixed(1));
labels[k].setAttribute('y',(ey-q[1]*8+4).toFixed(1))})}
function schedule(){if(!queued){queued=true;requestAnimationFrame(render)}}
function clampPitch(){pitch=Math.max(-1.5,Math.min(1.5,pitch))}
var last=null;
svg.addEventListener('pointerdown',function(ev){last=[ev.clientX,ev.clientY];
svg.classList.add('dragging');hide();try{svg.setPointerCapture(ev.pointerId)}catch(e){}});
svg.addEventListener('pointermove',function(ev){if(!last)return;
yaw+=(ev.clientX-last[0])*0.01;pitch+=(ev.clientY-last[1])*0.01;clampPitch();
last=[ev.clientX,ev.clientY];schedule()});
function end(){last=null;svg.classList.remove('dragging')}
svg.addEventListener('pointerup',end);svg.addEventListener('pointercancel',end);
svg.addEventListener('keydown',function(ev){var step=0.1,k=ev.key;
if(k==='ArrowLeft')yaw-=step;else if(k==='ArrowRight')yaw+=step;
else if(k==='ArrowUp')pitch-=step;else if(k==='ArrowDown')pitch+=step;else return;
ev.preventDefault();clampPitch();hide();schedule()});
var reset=wrap.closest('.card').querySelector('.reset-view');
if(reset)reset.addEventListener('click',function(){yaw=v.yaw;pitch=v.pitch;schedule()});
});
})();
"""
