import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from framework import (
    ComputationSpec,
    local_step,
    remote_step,
    site_output_step,
    stepped_workflow,
    with_state,
)

try:
    from framework.aggregator import ComputationAggregator
    from framework.executor import ComputationExecutor
    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_constant import ReturnCode
    from nvflare.apis.shareable import Shareable

    NVFLARE_AVAILABLE = True
except ImportError:
    NVFLARE_AVAILABLE = False


@dataclass
class CachedState:
    value: int


class FakeContext:
    def __init__(self, current_round=0):
        self.current_round = current_round

    def get_prop(self, key, default=None):
        if key == "CURRENT_ROUND":
            return self.current_round
        return default


@unittest.skipUnless(
    NVFLARE_AVAILABLE, "NVFlare is not installed in this Python environment"
)
class StateLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        self.parameters_path = os.path.join(self.temp_dir.name, "parameters.json")
        with open(self.parameters_path, "w", encoding="utf-8") as parameters_file:
            json.dump({}, parameters_file)
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
        self.context = FakeContext()

    def test_state_survives_non_output_step_and_is_removed_after_direct_output(self):
        observed_state = []

        def initialize(inputs):
            return with_state("ready", CachedState(value=inputs))

        def write_outputs(*, state: CachedState, output_dir):
            observed_state.append(state)
            with open(
                os.path.join(output_dir, "special.bin"), "w", encoding="utf-8"
            ) as output_file:
                output_file.write(str(state.value))

        executor = self._executor_for(
            stepped_workflow(
                local_step(fn=initialize, input_fn=lambda: 7),
                remote_step(fn=lambda site_results: None),
                site_output_step(fn=write_outputs),
            )
        )

        executor.execute("initialize", Shareable(), self.context, None)
        self.assertTrue(os.path.exists(self._state_path()))

        executor.execute("write_outputs", Shareable(), self.context, None)

        self.assertEqual(observed_state, [CachedState(value=7)])
        self.assertFalse(os.path.exists(self._state_path()))
        with open(
            os.path.join(self.output_dir, "special.bin"), encoding="utf-8"
        ) as output_file:
            self.assertEqual(output_file.read(), "7")

    def test_failed_output_keeps_state_until_end_run(self):
        def initialize(inputs):
            return with_state("ready", CachedState(value=inputs))

        def fail_output(*, state: CachedState):
            raise RuntimeError(f"cannot write {state.value}")

        executor = self._executor_for(
            stepped_workflow(
                local_step(fn=initialize, input_fn=lambda: 9),
                remote_step(fn=lambda site_results: None),
                site_output_step(fn=fail_output),
            )
        )
        executor.execute("initialize", Shareable(), self.context, None)

        result = executor.execute("fail_output", Shareable(), self.context, None)
        self.assertEqual(result.get_return_code(), ReturnCode.EXECUTION_EXCEPTION)
        self.assertTrue(os.path.exists(self._state_path()))

        executor.handle_event(EventType.END_RUN, self.context)

        self.assertFalse(os.path.exists(self._state_path()))

    def test_end_run_releases_remote_state_and_site_results(self):
        workflow = stepped_workflow(site_output_step(fn=lambda: {}))

        class TestAggregator(ComputationAggregator):
            SPEC = ComputationSpec(workflow=workflow)

        aggregator = TestAggregator()
        aggregator.site_results = {0: {"site1": {"value": 1}}}
        aggregator.remote_state = CachedState(value=4)

        aggregator.handle_event(EventType.END_RUN, self.context)

        self.assertEqual(aggregator.site_results, {})
        self.assertIsNone(aggregator.remote_state)

    def _executor_for(self, workflow):
        executor = ComputationExecutor()
        executor.SPEC = ComputationSpec(workflow=workflow)
        return executor

    def _state_path(self):
        return os.path.join(self.output_dir, "_temp_state", "local_state.json")


if __name__ == "__main__":
    unittest.main()
