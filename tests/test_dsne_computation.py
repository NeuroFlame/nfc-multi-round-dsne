"""Tests for the dSNE-MS computation math and site/central round logic."""

import logging
import os
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "app", "code"))

from computation import tsne  # noqa: E402
from computation.inputs import load_site_inputs  # noqa: E402
from computation.local_math import OPTIMIZER_CACHE, compute_site_round  # noqa: E402
from computation.remote_math import aggregate_site_round, is_complete  # noqa: E402
from computation.report import build_report  # noqa: E402
from computation.types import PHASE_DONE  # noqa: E402
from framework import cache_file  # noqa: E402
from framework.workflow import _StatefulValue  # noqa: E402

LOGGER = logging.getLogger("test_dsne")
SITES = ("site1", "site2", "site3")


def _unwrap(result):
    if isinstance(result, _StatefulValue):
        return result.payload, result.state
    return result, None


def _run(max_iterations, random_seed=0):
    """Drive the workflow in-process; return final message and site caches."""
    caches = {site: tempfile.mkdtemp() for site in SITES}
    artifact_dir = tempfile.mkdtemp()
    states, messages = {}, {}
    for site in SITES:
        data_dir = os.path.join(REPO_ROOT, "test_data", site)
        messages[site], states[site] = _unwrap(load_site_inputs(data_dir, caches[site]))

    remote_state = None
    for _ in range(max_iterations + 2):
        updates = {}
        for site in SITES:
            update, new_state = _unwrap(
                compute_site_round(
                    messages[site],
                    states[site],
                    cache_dir=caches[site],
                    logger=LOGGER,
                    max_iterations=max_iterations,
                    random_seed=random_seed,
                )
            )
            states[site] = new_state or states[site]
            updates[site] = update
        message, new_remote_state = _unwrap(
            aggregate_site_round(
                updates,
                remote_state,
                artifact_dir=artifact_dir,
                logger=LOGGER,
                max_iterations=max_iterations,
                random_seed=random_seed,
            )
        )
        remote_state = new_remote_state or remote_state
        messages = {site: message for site in SITES}
        if is_complete(message):
            return message, caches
    raise AssertionError("workflow did not complete")


class TsneMathTests(unittest.TestCase):
    def test_vectorized_gradient_matches_per_point_loop(self):
        rng = np.random.RandomState(3)
        Y = rng.randn(25, 2)
        P = rng.rand(25, 25)
        P = (P + P.T) / np.sum(P + P.T)
        num, Q = tsne._student_t(Y)

        expected = np.zeros_like(Y)
        PQ = P - Q
        for i in range(Y.shape[0]):
            expected[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (2, 1)).T * (Y[i, :] - Y), 0
            )

        np.testing.assert_allclose(tsne._gradient(P, Q, num, Y), expected, atol=1e-12)

    def test_pca_keeps_leading_components(self):
        rng = np.random.RandomState(4)
        X = rng.randn(200, 5) * np.array([10.0, 0.1, 5.0, 0.01, 1.0])
        projected = tsne.pca(X, 2)
        variances = np.var(projected, axis=0)
        self.assertGreater(variances[0], variances[1])
        self.assertGreater(variances[1], 20)

    def test_exaggeration_is_fully_removed(self):
        P = np.full((4, 4), 0.25)
        Y = np.random.RandomState(5).randn(4, 2)
        exaggerated = P * tsne.EARLY_EXAGGERATION
        *_, P_after, _ = tsne.site_gradient_step(
            Y,
            np.zeros_like(Y),
            np.ones_like(Y),
            exaggerated,
            tsne.STOP_EXAGGERATION_ITERATION,
            0.0,
        )
        np.testing.assert_allclose(P_after, P)


class WorkflowRoundTests(unittest.TestCase):
    def test_short_run_assembles_final_embedding(self):
        message, _ = _run(max_iterations=5)

        self.assertEqual(message.phase, PHASE_DONE)
        self.assertEqual(message.embedding.shape, (160, 2))
        self.assertEqual(len(message.embedding_sources), 160)
        self.assertTrue(np.all(np.isfinite(message.embedding)))

    def test_site_gains_are_persisted(self):
        _, caches = _run(max_iterations=5)
        with np.load(cache_file(OPTIMIZER_CACHE, cache_dir=caches["site1"])) as cached:
            self.assertFalse(np.all(cached["gains"] == 1.0))

    def test_seeded_runs_are_reproducible(self):
        first, _ = _run(max_iterations=3, random_seed=7)
        second, _ = _run(max_iterations=3, random_seed=7)
        np.testing.assert_array_equal(first.embedding, second.embedding)

    def test_final_message_identifies_sites_and_records_history(self):
        message, _ = _run(max_iterations=20)

        self.assertEqual(sorted(message.site_names.values()), list(SITES))
        self.assertEqual([entry[0] for entry in message.error_history], [10, 20])


class ReportTests(unittest.TestCase):
    def _report(self, n_labels):
        rng = np.random.RandomState(0)
        n = 60
        labels = rng.randint(0, n_labels, n).astype(float)
        sources = ["reference"] * 30 + ["site1"] * 20 + ["site2"] * 10
        return build_report(
            embedding=rng.randn(n, 2),
            labels=labels,
            sources=sources,
            own_name="site2",
            site_names=["site1", "site2"],
            error_history=[[i, 10.0 / i] for i in range(10, 201, 10)],
            final_error=0.05,
            parameters={"max_iterations": 200, "perplexity": 30},
            output_files={"global_embedding.csv": "Coordinates"},
        )

    def test_few_labels_are_colored_directly(self):
        html = self._report(n_labels=2)
        self.assertIn('data-mode="label"', html)
        self.assertNotIn('id="sec-labels"', html)
        self.assertIn('<span class="pill pill-own">site2</span>', html)

    def test_many_labels_use_small_multiples(self):
        html = self._report(n_labels=6)
        self.assertNotIn('data-mode="label"', html)
        self.assertIn('id="sec-labels"', html)
        self.assertEqual(html.count('class="card facet"'), 6)

    def test_site_names_are_escaped(self):
        html = build_report(
            embedding=np.zeros((2, 2)) + [[0, 0], [1, 1]],
            labels=np.array([0.0, 1.0]),
            sources=["reference", "<b>x</b>"],
            own_name="<b>x</b>",
            site_names=["<b>x</b>"],
            error_history=[],
            final_error=0.0,
            parameters={},
            output_files={},
        )
        self.assertNotIn("<b>x</b>", html)


if __name__ == "__main__":
    unittest.main()
