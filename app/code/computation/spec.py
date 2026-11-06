"""Declare the dSNE-MS iterative workflow.

Round 0 (setup): sites report data shape; central embeds the reference data.
Round 1 (initialize): sites build combined affinities; central starts optimizing.
Rounds 2+ (optimize): one gradient step per round until ``max_iterations``.
Final: every site writes the combined embedding.
"""

from framework import (
    ComputationSpec,
    iterative_workflow,
    local_step,
    remote_step,
    site_output_step,
)

from .inputs import load_site_inputs
from .local_math import compute_site_round
from .remote_math import aggregate_site_round, is_complete
from .results import build_outputs
from .types import MAX_SUPPORTED_ITERATIONS

# Setup and initialize rounds precede the gradient iterations.
_NON_GRADIENT_ROUNDS = 2

SPEC = ComputationSpec(
    workflow=iterative_workflow(
        local_step(fn=compute_site_round, input_fn=load_site_inputs),
        remote_step(fn=aggregate_site_round),
        site_output_step(fn=build_outputs),
        stop_when=is_complete,
        max_iterations=MAX_SUPPORTED_ITERATIONS + _NON_GRADIENT_ROUNDS,
    ),
)
