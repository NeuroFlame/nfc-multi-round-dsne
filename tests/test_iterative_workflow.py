import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict
from unittest.mock import patch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from framework import (
    ComputationSpec,
    iterative_workflow,
    local_step,
    remote_step,
    site_output_step,
    with_state,
)
from framework.serialization import deserialize_value, serialize_value
from framework.types import ITERATION_INDEX_KEY, ITERATION_STOP_KEY
from framework.workflow import get_task_names

try:
    from framework.aggregator import ComputationAggregator
    from framework.controller import ComputationController
    from framework.executor import ComputationExecutor
    from nvflare.apis.shareable import Shareable

    NVFLARE_AVAILABLE = True
except ImportError:
    NVFLARE_AVAILABLE = False


@dataclass
class Model:
    value: int


@dataclass
class SiteUpdate:
    value: int


@dataclass
class LocalCache:
    offset: int


@dataclass
class RemoteCache:
    calls: int


def build_workflow(input_calls=None, output_calls=None, max_iterations=5):
    def load_initial_model(start=1):
        if input_calls is not None:
            input_calls.append(start)
        return with_state(Model(value=start), LocalCache(offset=2))

    def compute_local_update(model: Model, state: LocalCache):
        return SiteUpdate(value=model.value + state.offset)

    def compute_global_update(
        site_updates: Dict[str, SiteUpdate],
        state: RemoteCache = None,
    ):
        calls = 1 if state is None else state.calls + 1
        total = sum(update.value for update in site_updates.values())
        return with_state(Model(value=total), RemoteCache(calls=calls))

    def has_converged(model: Model, state: RemoteCache, target=6):
        return model.value >= target and state.calls >= 1

    def build_outputs(model: Model, state: LocalCache):
        if output_calls is not None:
            output_calls.append((model, state))
        return {
            "result.json": {
                "value": model.value,
                "offset": state.offset,
            }
        }

    return iterative_workflow(
        local_step(fn=compute_local_update, input_fn=load_initial_model),
        remote_step(fn=compute_global_update),
        site_output_step(fn=build_outputs),
        stop_when=has_converged,
        max_iterations=max_iterations,
    )


class IterativeWorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(
            current_round=0,
            data_dir="/data",
            output_dir="/output",
            logger=None,
        )

    def test_initial_input_then_remote_feedback_use_the_same_local_function(self):
        input_calls = []
        workflow = build_workflow(input_calls=input_calls)

        first = workflow.iteration_step.local_fn(None, {"start": 3}, None, self.runtime)
        self.runtime.current_round = 1
        second = workflow.iteration_step.local_fn(
            Model(value=10),
            {"start": 99},
            first.local_state,
            self.runtime,
        )

        self.assertEqual(first.payload, SiteUpdate(value=5))
        self.assertEqual(first.local_state, LocalCache(offset=2))
        self.assertEqual(second.payload, SiteUpdate(value=12))
        self.assertIsNone(second.local_state)
        self.assertEqual(input_calls, [3])

    def test_remote_state_is_available_to_stop_predicate(self):
        workflow = build_workflow()

        remote_result = workflow.iteration_step.remote_fn(
            {"site1": SiteUpdate(value=3), "site2": SiteUpdate(value=4)},
            {"target": 7},
            None,
            self.runtime,
        )
        should_stop = workflow.stop_fn(
            remote_result.payload,
            {"target": 7},
            remote_result.remote_state,
            self.runtime,
        )

        self.assertEqual(remote_result.payload, Model(value=7))
        self.assertEqual(remote_result.remote_state, RemoteCache(calls=1))
        self.assertTrue(should_stop)

    def test_types_and_task_names_are_inferred(self):
        workflow = build_workflow()

        self.assertIs(workflow.iteration_step.local_input_type, Model)
        self.assertIs(workflow.iteration_step.remote_site_result_type, SiteUpdate)
        self.assertIs(workflow.output_step.local_input_type, Model)
        self.assertIs(workflow.local_state_type, LocalCache)
        self.assertEqual(
            get_task_names(workflow),
            ["compute_local_update", "build_outputs"],
        )

    def test_stop_predicate_must_return_bool(self):
        workflow = iterative_workflow(
            local_step(fn=lambda model: model, name="update"),
            remote_step(fn=lambda site_results: site_results),
            site_output_step(fn=lambda result: {}, name="output"),
            stop_when=lambda result: "done",
        )

        with self.assertRaisesRegex(TypeError, "must return bool"):
            workflow.stop_fn({}, {}, None, self.runtime)

    def test_stop_predicate_accepts_numpy_boolean(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed in this Python environment")

        workflow = iterative_workflow(
            local_step(fn=lambda model: model, name="update"),
            remote_step(fn=lambda site_results: site_results),
            site_output_step(fn=lambda result: {}, name="output"),
            stop_when=lambda result: np.bool_(True),
        )

        self.assertTrue(workflow.stop_fn({}, {}, None, self.runtime))

    def test_max_iterations_must_be_positive_integer(self):
        steps = (
            local_step(fn=lambda model: model),
            remote_step(fn=lambda site_results: site_results),
            site_output_step(fn=lambda result: {}),
        )

        for invalid_value in (0, -1, 1.5, True):
            with self.subTest(max_iterations=invalid_value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    iterative_workflow(*steps, max_iterations=invalid_value)

    def test_iterative_workflow_requires_local_remote_output_order(self):
        with self.assertRaisesRegex(TypeError, "first step must be local_step"):
            iterative_workflow(
                remote_step(fn=lambda values: values),
                remote_step(fn=lambda values: values),
                site_output_step(fn=lambda value: {}),
            )


class FakeContext:
    def __init__(self, current_round=0, parameters=None):
        self.props = {
            "CURRENT_ROUND": current_round,
            "COMPUTATION_PARAMETERS": parameters or {},
        }

    def get_prop(self, key, default=None):
        return self.props.get(key, default)

    def set_prop(self, key, value, **_kwargs):
        self.props[key] = value


@unittest.skipUnless(
    NVFLARE_AVAILABLE, "NVFlare is not installed in this Python environment"
)
class IterativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        self.parameters_path = os.path.join(self.temp_dir.name, "parameters.json")
        with open(self.parameters_path, "w", encoding="utf-8") as parameters_file:
            json.dump({"start": 3, "target": 7}, parameters_file)
        environment = patch.dict(
            os.environ,
            {
                "DATA_DIR": self.temp_dir.name,
                "OUTPUT_DIR": self.output_dir,
                "PARAMETERS_FILE_PATH": self.parameters_path,
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def test_executor_persists_setup_state_and_clears_it_after_output(self):
        input_calls = []
        output_calls = []
        workflow = build_workflow(input_calls=input_calls, output_calls=output_calls)
        executor = self._executor_for(workflow)
        context = FakeContext()

        first_input = Shareable()
        first_input[ITERATION_INDEX_KEY] = 0
        first_output = executor.execute(
            "compute_local_update",
            first_input,
            context,
            None,
        )

        self.assertEqual(
            deserialize_value(first_output["result"], SiteUpdate),
            SiteUpdate(value=5),
        )
        self.assertTrue(os.path.exists(self._state_path()))

        second_input = Shareable()
        second_input["result"] = serialize_value(Model(value=10))
        second_input[ITERATION_INDEX_KEY] = 1
        second_output = executor.execute(
            "compute_local_update",
            second_input,
            context,
            None,
        )

        self.assertEqual(
            deserialize_value(second_output["result"], SiteUpdate),
            SiteUpdate(value=12),
        )
        self.assertEqual(input_calls, [3])

        output_input = Shareable()
        output_input["result"] = serialize_value(Model(value=10))
        output_input[ITERATION_INDEX_KEY] = 2
        executor.execute("build_outputs", output_input, context, None)

        self.assertEqual(
            output_calls,
            [(Model(value=10), LocalCache(offset=2))],
        )
        self.assertFalse(os.path.exists(self._state_path()))
        with open(
            os.path.join(self.output_dir, "result.json"), encoding="utf-8"
        ) as output_file:
            self.assertEqual(json.load(output_file), {"value": 10, "offset": 2})

    def test_aggregator_deserializes_sites_updates_state_and_signals_stop(self):
        workflow = build_workflow()
        spec = self._spec_for(workflow)

        class TestAggregator(ComputationAggregator):
            SPEC = spec

        aggregator = TestAggregator()
        aggregator.site_results = {
            0: {
                "site1": serialize_value(SiteUpdate(value=3)),
                "site2": serialize_value(SiteUpdate(value=4)),
            }
        }
        context = FakeContext(current_round=0, parameters={"target": 7})

        result = aggregator.aggregate(context)

        self.assertEqual(deserialize_value(result["result"], Model), Model(value=7))
        self.assertTrue(result[ITERATION_STOP_KEY])
        self.assertEqual(aggregator.remote_state, RemoteCache(calls=1))

    def test_controller_repeats_until_stop_then_runs_output_once(self):
        workflow = build_workflow(max_iterations=5)
        scheduled = []

        class StubAggregator:
            def __init__(self):
                self.calls = 0

            def aggregate(self, _fl_ctx):
                self.calls += 1
                result = Shareable()
                result["result"] = serialize_value(Model(value=self.calls))
                result[ITERATION_STOP_KEY] = self.calls == 2
                return result

        def broadcast_task(**kwargs):
            scheduled.append(
                (kwargs["task_name"], kwargs["data"].get(ITERATION_INDEX_KEY))
            )

        harness = SimpleNamespace(
            aggregator=StubAggregator(),
            _broadcast_task=broadcast_task,
            _accept_site_result=lambda *_args: True,
            _validate_site_result=lambda *_args: True,
        )
        context = FakeContext()

        ComputationController._run_iterative_workflow(
            harness,
            workflow,
            abort_signal=None,
            fl_ctx=context,
        )

        self.assertEqual(
            scheduled,
            [
                ("compute_local_update", 0),
                ("compute_local_update", 1),
                ("build_outputs", 2),
            ],
        )
        self.assertEqual(context.get_prop("CURRENT_ROUND"), 2)

    def test_controller_stops_at_iteration_cap_then_runs_output_once(self):
        workflow = build_workflow(max_iterations=2)
        scheduled = []

        class StubAggregator:
            def aggregate(self, _fl_ctx):
                result = Shareable()
                result["result"] = serialize_value(Model(value=1))
                result[ITERATION_STOP_KEY] = False
                return result

        def broadcast_task(**kwargs):
            scheduled.append(
                (kwargs["task_name"], kwargs["data"].get(ITERATION_INDEX_KEY))
            )

        harness = SimpleNamespace(
            aggregator=StubAggregator(),
            _broadcast_task=broadcast_task,
            _accept_site_result=lambda *_args: True,
            _validate_site_result=lambda *_args: True,
        )
        context = FakeContext()

        ComputationController._run_iterative_workflow(
            harness,
            workflow,
            abort_signal=None,
            fl_ctx=context,
        )

        self.assertEqual(
            scheduled,
            [
                ("compute_local_update", 0),
                ("compute_local_update", 1),
                ("build_outputs", 2),
            ],
        )
        self.assertEqual(context.get_prop("CURRENT_ROUND"), 2)

    def _spec_for(self, workflow):
        return ComputationSpec(workflow=workflow)

    def _executor_for(self, workflow):
        executor = ComputationExecutor()
        executor.SPEC = self._spec_for(workflow)
        return executor

    def _state_path(self):
        return os.path.join(self.output_dir, "_temp_state", "local_state.json")


if __name__ == "__main__":
    unittest.main()
