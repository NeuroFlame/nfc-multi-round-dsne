import importlib.util
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(REPO_ROOT, "app", "code")
sys.path.insert(0, CODE_DIR)

from framework import ComputationSpec, local_step, remote_step, stepped_workflow
from framework.errors import (
    ERROR_ENVELOPE_KEY,
    ERROR_ORIGIN_CENTRAL,
    ERROR_ORIGIN_SITE,
    build_error_envelope,
    emit_shared_error_summary,
    find_terminal_errors,
    raise_for_terminal_errors,
    record_terminal_error,
)

NVFLARE_AVAILABLE = importlib.util.find_spec("nvflare") is not None

if NVFLARE_AVAILABLE:
    from framework.artifact_transfer import ArtifactTransferError
    from framework.controller import ComputationController, RelayedSiteFailure
    from framework.executor import ComputationExecutor
    from nvflare.apis.controller_spec import TaskCompletionStatus
    from nvflare.apis.fl_constant import FLContextKey, ReturnCode
    from nvflare.apis.job_def import JobMetaKey, RunStatus
    from nvflare.apis.shareable import make_reply
    from nvflare.fuel.flare_api.api_spec import TargetType

    NVFLARE_AVAILABLE = True


class FakeContext:
    def __init__(self, current_round=0, parameters=None, client_name=None):
        self.props = {
            "CURRENT_ROUND": current_round,
            "COMPUTATION_PARAMETERS": parameters or {},
        }
        if NVFLARE_AVAILABLE and client_name is not None:
            self.props[FLContextKey.CLIENT_NAME] = client_name

    def get_prop(self, key, default=None):
        return self.props.get(key, default)

    def set_prop(self, key, value, **_kwargs):
        self.props[key] = value


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, relative_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TerminalErrorMarkerTests(unittest.TestCase):
    def test_terminal_error_marker_preserves_message_and_traceback(self):
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                raise ValueError("bad computation input")
            except ValueError as error:
                record_terminal_error(
                    output_dir,
                    "input",
                    error,
                    origin=ERROR_ORIGIN_CENTRAL,
                    stage="input_validation",
                )

            errors = find_terminal_errors(output_dir)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["scope"], "input")
            self.assertEqual(errors[0]["origin"], ERROR_ORIGIN_CENTRAL)
            self.assertEqual(errors[0]["stage"], "input_validation")
            self.assertEqual(errors[0]["message"], "bad computation input")
            self.assertIn("ValueError: bad computation input", errors[0]["traceback"])
            with self.assertRaisesRegex(RuntimeError, "bad computation input"):
                raise_for_terminal_errors(output_dir)

    def test_shared_summary_omits_central_message_and_sensitive_path(self):
        with tempfile.TemporaryDirectory() as output_dir:
            error = ValueError("failed at /sensitive/subject-123/image.nii")
            record_terminal_error(
                output_dir,
                "remote round 2",
                error,
                origin=ERROR_ORIGIN_CENTRAL,
                stage="aggregation",
            )
            with patch("builtins.print") as print_mock:
                emit_shared_error_summary(
                    output_dir,
                    fallback_origin="central",
                    fallback_stage="execution",
                )
            summary = print_mock.call_args.args[0]
            envelope = json.loads(summary.split(":", 1)[1])
            self.assertEqual(
                envelope,
                {
                    "schema_version": 1,
                    "origin": "central",
                    "stage": "aggregation",
                    "code": "central_computation_failed",
                },
            )
            self.assertNotIn("subject-123", summary)
            self.assertNotIn("/sensitive", summary)


@unittest.skipUnless(
    NVFLARE_AVAILABLE, "NVFlare is not installed in this Python environment"
)
class RuntimeErrorHandlingTests(unittest.TestCase):
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

    def test_non_ok_client_result_is_terminal(self):
        client_task = SimpleNamespace(
            result=make_reply(ReturnCode.EXECUTION_EXCEPTION),
            task=SimpleNamespace(name="compute"),
            client=SimpleNamespace(name="site1"),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "compute.*site1.*EXECUTION_EXCEPTION",
        ):
            ComputationController._validate_site_result(
                None,
                client_task=client_task,
                fl_ctx=None,
            )

    def test_structured_site_failure_is_relayed_without_message_parsing(self):
        result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
        result[ERROR_ENVELOPE_KEY] = build_error_envelope(
            ERROR_ORIGIN_SITE,
            "task_execution",
            "site task compute",
        )
        client_task = SimpleNamespace(
            result=result,
            task=SimpleNamespace(name="compute"),
            client=SimpleNamespace(name="site1"),
        )

        with self.assertRaises(RelayedSiteFailure):
            ComputationController._validate_site_result(
                None,
                client_task=client_task,
                fl_ctx=None,
            )

    def test_site_transfer_failure_is_relayed_with_sanitized_detail(self):
        def reject_transfer(_result, _fl_ctx):
            raise ArtifactTransferError(
                "failed reading /sensitive/subject-123/average.nii"
            )

        controller = SimpleNamespace(
            aggregator=SimpleNamespace(accept=reject_transfer),
            _validate_site_result=lambda *_args, **_kwargs: True,
        )
        client_task = SimpleNamespace(
            result=make_reply(ReturnCode.OK),
            task=SimpleNamespace(name="compute"),
            client=SimpleNamespace(name="site1"),
        )
        with self.assertRaises(RelayedSiteFailure) as raised:
            ComputationController._accept_site_result(
                controller, client_task=client_task, fl_ctx=None
            )
        self.assertIn("site1", str(raised.exception))
        self.assertNotIn("subject-123", str(raised.exception))
        self.assertNotIn("/sensitive", str(raised.exception))

    def test_non_ok_task_completion_is_terminal(self):
        def broadcast_and_wait(**kwargs):
            kwargs["task"].completion_status = TaskCompletionStatus.TIMEOUT

        harness = SimpleNamespace(
            _task_timeout=5,
            _min_clients=1,
            _wait_time_after_min_received=0,
            broadcast_and_wait=broadcast_and_wait,
        )

        with self.assertRaisesRegex(RuntimeError, "compute.*TIMEOUT|compute.*timeout"):
            ComputationController._broadcast_task(
                harness,
                task_name="compute",
                data=make_reply(ReturnCode.OK),
                result_cb=lambda *_args, **_kwargs: True,
                fl_ctx=None,
                abort_signal=None,
            )

    def test_executor_returns_safe_envelope_and_keeps_full_error_local(self):
        def fail_local(_payload):
            raise ValueError("local math failed")

        workflow = stepped_workflow(
            local_step(fn=fail_local),
            remote_step(fn=lambda site_results: site_results),
        )
        executor = ComputationExecutor()
        executor.SPEC = ComputationSpec(workflow)

        result = executor.execute(
            "fail_local",
            make_reply(ReturnCode.OK),
            FakeContext(client_name="site1"),
            None,
        )

        self.assertEqual(result.get_return_code(), ReturnCode.EXECUTION_EXCEPTION)
        self.assertEqual(
            result[ERROR_ENVELOPE_KEY],
            build_error_envelope(
                ERROR_ORIGIN_SITE,
                "task_execution",
                "site task fail_local",
            ),
        )
        self.assertNotIn("local math failed", json.dumps(result))

        with open(
            os.path.join(self.output_dir, "site1.log"), encoding="utf-8"
        ) as log_file:
            log_text = log_file.read()
        self.assertIn("Computation task 'fail_local' failed", log_text)
        self.assertIn("ValueError: local math failed", log_text)
        errors = find_terminal_errors(self.output_dir)
        self.assertEqual(errors[0]["message"], "local math failed")
        self.assertEqual(errors[0]["origin"], ERROR_ORIGIN_SITE)

    def test_controller_startup_error_records_terminal_marker(self):
        def compute(payload):
            return payload

        workflow = stepped_workflow(
            local_step(fn=compute),
            remote_step(fn=lambda site_results: site_results),
        )

        class TestController(ComputationController):
            SPEC = ComputationSpec(workflow)

        controller = TestController(min_clients=1)
        controller._engine = SimpleNamespace(get_component=lambda _component_id: Mock())
        with open(self.parameters_path, "w", encoding="utf-8") as parameters_file:
            parameters_file.write("not json")

        with self.assertRaises(json.JSONDecodeError):
            controller.start_controller(FakeContext())

        errors = find_terminal_errors(self.output_dir)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["scope"], "controller startup")
        self.assertIn("JSONDecodeError", errors[0]["traceback"])

    def test_executor_setup_error_records_terminal_marker(self):
        def compute(payload):
            return payload

        workflow = stepped_workflow(
            local_step(fn=compute),
            remote_step(fn=lambda site_results: site_results),
        )
        executor = ComputationExecutor()
        executor.SPEC = ComputationSpec(workflow)
        with open(self.parameters_path, "w", encoding="utf-8") as parameters_file:
            parameters_file.write("not json")

        result = executor.execute(
            "compute",
            make_reply(ReturnCode.OK),
            FakeContext(client_name="site1"),
            None,
        )

        self.assertEqual(result.get_return_code(), ReturnCode.EXECUTION_EXCEPTION)
        errors = find_terminal_errors(self.output_dir)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["scope"], "site task compute")
        self.assertIn("JSONDecodeError", errors[0]["traceback"])


@unittest.skipUnless(
    NVFLARE_AVAILABLE, "NVFlare is not installed in this Python environment"
)
class EntrypointErrorHandlingTests(unittest.TestCase):
    def test_central_deployment_timeout_reports_only_allowlisted_detail(self):
        entry_central = load_module(
            "test_entry_central_deployment_timeout", "system/entry_central.py"
        )
        session = Mock()
        session.submit_job.return_value = "job-id"
        session.wait_for_job.return_value = {
            JobMetaKey.STATUS.value: RunStatus.FAILED_TO_RUN.value,
            JobMetaKey.JOB_DEPLOY_DETAIL.value: [
                "server: OK",
                "site1: no reply (deployment timeout)",
                "site2: /sensitive/path/subject-name",
                "num_ok_sites 0 < required_min_sites 2",
            ],
        }

        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.dict(os.environ, {"OUTPUT_DIR": output_dir}),
                patch.object(entry_central, "start_server"),
                patch.object(
                    entry_central,
                    "new_secure_session",
                    return_value=session,
                ),
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Job deployment timed out waiting for participants: site1",
                ):
                    entry_central.main()

            errors = find_terminal_errors(output_dir)
            self.assertEqual(len(errors), 1)
            self.assertEqual(
                errors[0]["message"],
                "Job deployment timed out waiting for participants: site1",
            )
            self.assertNotIn("sensitive", errors[0]["message"])

    def test_central_session_startup_error_is_recorded_and_reraised(self):
        entry_central = load_module(
            "test_entry_central_session_startup", "system/entry_central.py"
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.dict(os.environ, {"OUTPUT_DIR": output_dir}),
                patch.object(entry_central, "start_server"),
                patch.object(
                    entry_central,
                    "new_secure_session",
                    side_effect=RuntimeError("cannot connect to server"),
                ),
                patch("builtins.print") as print_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot connect to server"):
                    entry_central.main()

            errors = find_terminal_errors(output_dir)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["origin"], ERROR_ORIGIN_CENTRAL)
            self.assertEqual(errors[0]["stage"], "controller_startup")
            self.assertEqual(errors[0]["scope"], "central runtime")
            self.assertEqual(errors[0]["message"], "cannot connect to server")
            summary = print_mock.call_args.args[0]
            self.assertNotIn("cannot connect to server", summary)
            self.assertEqual(
                json.loads(summary.split(":", 1)[1]),
                {
                    "schema_version": 1,
                    "origin": "central",
                    "stage": "startup",
                    "code": "central_computation_failed",
                },
            )

    def test_central_server_startup_error_does_not_create_a_session(self):
        entry_central = load_module(
            "test_entry_central_server_startup", "system/entry_central.py"
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.dict(os.environ, {"OUTPUT_DIR": output_dir}),
                patch.object(
                    entry_central,
                    "start_server",
                    side_effect=RuntimeError("server startup failed"),
                ),
                patch.object(entry_central, "new_secure_session") as new_session,
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(RuntimeError, "server startup failed"):
                    entry_central.main()

            new_session.assert_not_called()
            errors = find_terminal_errors(output_dir)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["stage"], "controller_startup")
            self.assertEqual(errors[0]["message"], "server startup failed")

    def test_central_abnormal_job_shuts_down_and_raises(self):
        entry_central = load_module("test_entry_central", "system/entry_central.py")
        session = Mock()
        session.submit_job.return_value = "job-id"
        session.wait_for_job.return_value = {
            JobMetaKey.STATUS.value: RunStatus.FINISHED_EXECUTION_EXCEPTION.value,
        }

        with (
            patch.object(entry_central, "start_server"),
            patch.object(
                entry_central,
                "new_secure_session",
                return_value=session,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "FINISHED:EXECUTION_EXCEPTION"):
                entry_central.main()

        session.shutdown.assert_called_once_with(TargetType.ALL)

    def test_central_completed_job_shuts_down_without_error(self):
        entry_central = load_module(
            "test_entry_central_success", "system/entry_central.py"
        )
        session = Mock()
        session.submit_job.return_value = "job-id"
        session.wait_for_job.return_value = {
            JobMetaKey.STATUS.value: RunStatus.FINISHED_COMPLETED.value,
        }

        with (
            patch.object(entry_central, "start_server"),
            patch.object(
                entry_central,
                "new_secure_session",
                return_value=session,
            ),
        ):
            entry_central.main()

        session.shutdown.assert_called_once_with(TargetType.ALL)

    def test_central_shutdown_error_does_not_mask_job_error(self):
        entry_central = load_module(
            "test_entry_central_shutdown_error", "system/entry_central.py"
        )
        session = Mock()
        session.submit_job.return_value = "job-id"
        session.wait_for_job.return_value = {
            JobMetaKey.STATUS.value: RunStatus.FINISHED_EXECUTION_EXCEPTION.value,
        }
        session.shutdown.side_effect = RuntimeError("shutdown failed")

        with (
            patch.object(entry_central, "start_server"),
            patch.object(
                entry_central,
                "new_secure_session",
                return_value=session,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "FINISHED:EXECUTION_EXCEPTION"
            ) as raised:
                entry_central.main()

        self.assertIn("shutdown failed", " ".join(raised.exception.__notes__))

    def test_edge_tracks_foreground_nvflare_daemon(self):
        entry_edge = load_module("test_entry_edge", "system/entry_edge.py")
        child = Mock()
        child.wait.return_value = 0

        with patch.object(
            entry_edge.subprocess,
            "Popen",
            return_value=child,
        ) as popen:
            entry_edge.main()

        popen.assert_called_once_with(
            ["/bin/bash", "/workspace/runKit/startup/sub_start.sh", "--once"],
            start_new_session=True,
        )
        child.wait.assert_called_once_with()

    def test_edge_reports_terminal_marker_before_subprocess_status(self):
        entry_edge = load_module("test_entry_edge_marker", "system/entry_edge.py")
        child = Mock()
        child.wait.return_value = 0

        with (
            patch.object(
                entry_edge.subprocess,
                "Popen",
                return_value=child,
            ),
            patch.object(
                entry_edge,
                "raise_for_terminal_errors",
                side_effect=RuntimeError("local math failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "local math failed"):
                entry_edge.main()

        child.wait.assert_called_once_with()

    def test_debugger_escalates_marker_after_zero_simulator_status(self):
        debugger = load_module("test_debugger", "debugger.py")
        completed_process = Mock(returncode=0)
        with tempfile.TemporaryDirectory() as workspace:
            simulator_args = SimpleNamespace(
                job_folder="job",
                workspace=workspace,
                clients="site1",
                n_clients=1,
                threads=None,
                gpu=None,
                log_config=None,
                max_clients=100,
                end_run_for_all=False,
            )

            with (
                patch.object(
                    debugger.subprocess,
                    "run",
                    return_value=completed_process,
                ),
                patch.object(
                    debugger,
                    "raise_for_terminal_errors",
                    side_effect=RuntimeError("remote math failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "remote math failed"):
                    debugger.run_simulator(simulator_args)


if __name__ == "__main__":
    unittest.main()
