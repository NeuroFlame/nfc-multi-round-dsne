import inspect
import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

import framework
from computation.spec import SPEC
from framework import (
    ComputationSpec,
    local_step,
    remote_step,
    site_output_step,
    stepped_workflow,
    with_state,
)
from framework.serialization import DEFAULT_MAX_INLINE_ARRAY_BYTES
from framework.workflow import get_task_names


class PublicApiTests(unittest.TestCase):
    def test_public_api_is_narrow(self):
        self.assertEqual(
            set(framework.__all__),
            {
                "ArtifactRef",
                "ComputationSpec",
                "artifact",
                "cache_file",
                "iterative_workflow",
                "local_step",
                "remote_step",
                "site_output_step",
                "stepped_workflow",
                "with_state",
            },
        )
        self.assertFalse(hasattr(framework, "StepDefinition"))
        self.assertFalse(hasattr(framework, "StepResult"))
        self.assertFalse(hasattr(framework, "IterativeWorkflow"))

    def test_computation_spec_only_requires_workflow(self):
        workflow = stepped_workflow(site_output_step(fn=lambda: {}))
        spec = ComputationSpec(workflow=workflow)

        parameters = inspect.signature(ComputationSpec).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "workflow",
                "codecs",
                "max_inline_array_bytes",
                "max_artifact_bytes",
                "max_artifact_total_bytes",
                "artifact_timeout",
                "artifact_retries",
            ),
        )
        self.assertEqual(
            parameters["workflow"].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertEqual(parameters["codecs"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(
            parameters["max_inline_array_bytes"].kind, inspect.Parameter.KEYWORD_ONLY
        )
        self.assertEqual(spec.codecs, {})
        self.assertEqual(spec.max_inline_array_bytes, DEFAULT_MAX_INLINE_ARRAY_BYTES)
        for internal_name in (
            "aggregator_id",
            "data_mode",
            "get_data_dir",
            "get_output_dir",
            "get_parameters_path",
            "local_state_type",
            "create_logger",
            "write_outputs",
        ):
            self.assertFalse(hasattr(spec, internal_name))

    def test_computation_spec_uses_framework_defaults(self):
        from computation.types import LocalState

        self.assertEqual(SPEC.codecs, {})
        self.assertEqual(SPEC.max_inline_array_bytes, DEFAULT_MAX_INLINE_ARRAY_BYTES)
        self.assertIs(SPEC.workflow.local_state_type, LocalState)

    def test_advanced_serialization_options_are_keyword_only(self):
        class CustomValue:
            pass

        class CustomCodec:
            @staticmethod
            def encode(value):
                return {"value": str(value)}

            @staticmethod
            def decode(value):
                return value

        workflow = stepped_workflow(site_output_step(fn=lambda: {}))
        spec = ComputationSpec(
            workflow,
            codecs={CustomValue: CustomCodec},
            max_inline_array_bytes=1024,
        )

        self.assertEqual(spec.codecs, {CustomValue: CustomCodec})
        self.assertEqual(spec.max_inline_array_bytes, 1024)

    def test_computation_task_names_are_derived_from_site_functions(self):
        self.assertEqual(
            get_task_names(SPEC.workflow),
            ["compute_site_round", "build_outputs"],
        )


class WorkflowGrammarTests(unittest.TestCase):
    def test_workflow_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "requires at least one step"):
            stepped_workflow()

    def test_local_step_requires_an_immediate_remote_step(self):
        def compute(value):
            return value

        with self.assertRaisesRegex(
            ValueError, "must be immediately followed by remote_step"
        ):
            stepped_workflow(local_step(fn=compute))

    def test_remote_step_cannot_stand_alone(self):
        with self.assertRaisesRegex(ValueError, "must follow a local_step immediately"):
            stepped_workflow(remote_step(fn=lambda site_results: site_results))

    def test_site_output_step_must_be_final(self):
        with self.assertRaisesRegex(ValueError, "must be the final workflow step"):
            stepped_workflow(
                site_output_step(fn=lambda: {}),
                local_step(fn=lambda value: value),
                remote_step(fn=lambda site_results: site_results),
            )

    def test_inferred_step_names_must_be_unique(self):
        def compute(value):
            return value

        with self.assertRaisesRegex(ValueError, r"compute.*unique name="):
            stepped_workflow(
                local_step(fn=compute, input_fn=lambda: 1),
                remote_step(fn=lambda site_results: site_results),
                local_step(fn=compute),
                remote_step(fn=lambda site_results: site_results),
            )

    def test_reused_function_can_have_explicit_unique_names(self):
        def compute(value):
            return value

        workflow = stepped_workflow(
            local_step(fn=compute, input_fn=lambda: 1, name="compute_first"),
            remote_step(fn=lambda site_results: site_results),
            local_step(fn=compute, name="compute_second"),
            remote_step(fn=lambda site_results: site_results),
        )

        self.assertEqual(get_task_names(workflow), ["compute_first", "compute_second"])

    def test_explicit_step_name_must_be_nonempty(self):
        with self.assertRaisesRegex(TypeError, "name must be a non-empty string"):
            stepped_workflow(site_output_step(fn=lambda: {}, name="  "))

    def test_step_function_must_be_callable(self):
        with self.assertRaisesRegex(TypeError, "fn must be callable"):
            stepped_workflow(site_output_step(fn=42))


@dataclass
class CachedState:
    value: int


class WorkflowValueTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(
            data_dir="/data",
            output_dir="/output",
            logger=None,
        )

    def test_with_state_is_converted_to_internal_step_result(self):
        def initialize(inputs):
            return with_state(inputs + 1, CachedState(value=inputs))

        def consume(remote_value: int, state: CachedState):
            return remote_value + state.value

        workflow = stepped_workflow(
            local_step(fn=initialize, input_fn=lambda: 4),
            remote_step(fn=lambda site_results: 10),
            local_step(fn=consume),
            remote_step(fn=lambda site_results: 20),
        )

        first_result = workflow.steps[0].local_fn(None, {}, None, self.runtime)

        self.assertEqual(first_result.payload, 5)
        self.assertEqual(first_result.local_state, CachedState(value=4))
        self.assertIs(workflow.local_state_type, CachedState)

    def test_tuple_is_an_ordinary_payload(self):
        workflow = stepped_workflow(
            local_step(fn=lambda inputs: (inputs, inputs + 1), input_fn=lambda: 4),
            remote_step(fn=lambda site_results: None),
        )

        result = workflow.steps[0].local_fn(None, {}, None, self.runtime)

        self.assertEqual(result.payload, (4, 5))
        self.assertIsNone(result.local_state)

    def test_with_state_sets_remote_state(self):
        def aggregate(site_results):
            return with_state(sum(site_results.values()), CachedState(value=9))

        workflow = stepped_workflow(
            local_step(fn=lambda inputs: inputs, input_fn=lambda: 4),
            remote_step(fn=aggregate),
        )

        result = workflow.steps[0].remote_fn(
            {"site1": 4, "site2": 5}, {}, None, self.runtime
        )

        self.assertEqual(result.payload, 9)
        self.assertEqual(result.remote_state, CachedState(value=9))

    def test_framework_values_and_named_configuration_are_injected(self):
        observed = {}

        def load_inputs(data_dir, parameters, logger):
            observed.update(
                data_dir=data_dir,
                parameters=parameters,
                logger=logger,
            )
            return 1 / 3

        def compute(inputs, decimal_places=2):
            return round(inputs, decimal_places)

        workflow = stepped_workflow(
            local_step(fn=compute, input_fn=load_inputs),
            remote_step(fn=lambda site_results: None),
        )
        parameters = {"decimal_places": 4}

        result = workflow.steps[0].local_fn(None, parameters, None, self.runtime)

        self.assertEqual(result.payload, 0.3333)
        self.assertEqual(observed["data_dir"], "/data")
        self.assertIs(observed["parameters"], parameters)
        self.assertIsNone(observed["logger"])

    def test_keyword_only_function_can_ignore_payload(self):
        def build_outputs(*, output_dir):
            return {"results.json": {"output_dir": output_dir}}

        workflow = stepped_workflow(site_output_step(fn=build_outputs))

        result = workflow.steps[0].local_fn("ignored", {}, None, self.runtime)

        self.assertEqual(result.outputs, {"results.json": {"output_dir": "/output"}})

    def test_site_output_can_write_directly_and_return_none(self):
        observed = []

        def build_outputs(*, output_dir):
            observed.append(output_dir)

        workflow = stepped_workflow(site_output_step(fn=build_outputs))

        result = workflow.steps[0].local_fn("ignored", {}, None, self.runtime)

        self.assertEqual(observed, ["/output"])
        self.assertEqual(result.outputs, {})

    def test_site_output_rejects_non_mapping_result(self):
        workflow = stepped_workflow(site_output_step(fn=lambda: ["not", "files"]))

        with self.assertRaisesRegex(TypeError, "filename-to-value mapping or None"):
            workflow.steps[0].local_fn(None, {}, None, self.runtime)

    def test_injected_name_cannot_be_mistaken_for_payload(self):
        def build_outputs(output_dir):
            return {"results.json": {"output_dir": output_dir}}

        with self.assertRaisesRegex(
            TypeError, "reserved injected parameter 'output_dir'"
        ):
            stepped_workflow(site_output_step(fn=build_outputs))

    def test_missing_required_configuration_raises(self):
        def compute(inputs, required_setting):
            return inputs + required_setting

        workflow = stepped_workflow(
            local_step(fn=compute, input_fn=lambda: 4),
            remote_step(fn=lambda site_results: None),
        )

        with self.assertRaisesRegex(
            TypeError, "required_setting.*missing from computation parameters"
        ):
            workflow.steps[0].local_fn(None, {}, None, self.runtime)

    def test_missing_required_state_raises(self):
        def compute(inputs, state: CachedState):
            return inputs + state.value

        workflow = stepped_workflow(
            local_step(fn=compute, input_fn=lambda: 4),
            remote_step(fn=lambda site_results: None),
        )

        with self.assertRaisesRegex(TypeError, "state.*no state is available"):
            workflow.steps[0].local_fn(None, {}, None, self.runtime)

    def test_legacy_injection_alias_is_rejected(self):
        def compute(inputs, computation_parameters):
            return inputs

        with self.assertRaisesRegex(
            TypeError, "computation_parameters.*use 'parameters'"
        ):
            stepped_workflow(local_step(fn=compute))

    def test_variadic_signature_is_rejected(self):
        def compute(inputs, **kwargs):
            return inputs

        with self.assertRaisesRegex(TypeError, r"cannot use \*args or \*\*kwargs"):
            stepped_workflow(local_step(fn=compute))


if __name__ == "__main__":
    unittest.main()
