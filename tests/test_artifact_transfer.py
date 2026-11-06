import hashlib
import os
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from framework import artifact
from framework import artifact_transfer as artifact_transfer_module
from framework.aggregator import ComputationAggregator
from framework.artifact_transfer import (
    _STREAM_TRANSACTION_KEY,
    ArtifactTransfer,
    ArtifactTransferError,
    _is_nvflare_simulator,
    _promote_verified_file,
    materialize_incoming_artifacts,
    prepare_outgoing_artifacts,
)
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable


class _QueuedReceiver(ArtifactTransfer):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.retrieve_calls = 0

    def retrieve(self, from_site, fl_ctx, timeout, **attributes):
        self.retrieve_calls += 1
        code, content = self.responses.pop(0)
        if content is None:
            return code, None, code in (ReturnCode.TIMEOUT, ReturnCode.TASK_ABORTED)
        descriptor, path = tempfile.mkstemp(dir=self._incoming_dir)
        with os.fdopen(descriptor, "wb") as received_file:
            received_file.write(content)
        return code, path, False


def _manifest(content=b"content", name="artifact.bin"):
    return {
        "transfer_id": "a" * 32,
        "name": name,
        "media_type": None,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "stage": "stage-one",
        "direction": "site_to_central",
    }


class ArtifactStagingTests(unittest.TestCase):
    def _initialize(self, transfer, output_dir):
        with patch.dict(
            os.environ,
            {
                "OUTPUT_DIR": output_dir,
                "NEUROFLAME_ARTIFACT_TEMP_DIR": output_dir,
            },
        ):
            transfer._initialize_run(object())

    def test_run_staging_uses_private_container_temporary_storage(self):
        transfer = ArtifactTransfer()
        with (
            tempfile.TemporaryDirectory() as output_dir,
            tempfile.TemporaryDirectory() as temporary_parent,
            patch.dict(
                os.environ,
                {
                    "OUTPUT_DIR": output_dir,
                    "NEUROFLAME_ARTIFACT_TEMP_DIR": temporary_parent,
                },
            ),
        ):
            transfer._initialize_run(object())

            self.assertEqual(
                Path(transfer._artifact_root).parent,
                Path(temporary_parent).resolve(),
            )
            self.assertFalse(
                Path(transfer._artifact_root).is_relative_to(Path(output_dir))
            )
            self.assertEqual(
                stat.S_IMODE(os.stat(transfer._artifact_root).st_mode), 0o700
            )
            transfer.cleanup()
            self.assertFalse(os.path.exists(transfer._artifact_root))

    def test_empty_small_and_multichunk_files_are_staged_without_payload_bytes(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source_root = os.path.join(output_dir, "author")
            os.mkdir(source_root, 0o700)
            transfer = ArtifactTransfer(chunk_size=8)
            self._initialize(transfer, output_dir)

            values = {}
            for name, content in (
                ("empty.bin", b""),
                ("small.bin", b"abc"),
                ("large.bin", b"0123456789" * 10),
            ):
                path = os.path.join(source_root, name)
                Path(path).write_bytes(content)
                values[name] = artifact(name, path)

            prepared, manifests = prepare_outgoing_artifacts(
                values,
                transfer=transfer,
                source_root=source_root,
                allowed_requesters=["server"],
                stage="stage-one",
                direction="site_to_central",
                max_file_bytes=1024,
                max_total_bytes=2048,
            )

            self.assertEqual([item["size"] for item in manifests], [0, 3, 100])
            self.assertNotIn(b"0123456789", repr(prepared).encode())
            self.assertEqual(len(transfer._registry), 3)

    def test_path_traversal_symlink_and_size_limit_are_rejected(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source_root = os.path.join(output_dir, "author")
            os.mkdir(source_root, 0o700)
            transfer = ArtifactTransfer()
            self._initialize(transfer, output_dir)
            outside = os.path.join(output_dir, "outside.bin")
            Path(outside).write_bytes(b"outside")
            link = os.path.join(source_root, "link.bin")
            os.symlink(outside, link)

            for ref, message in (
                (artifact("outside.bin", outside), "outside"),
                (artifact("link.bin", link), "symlink"),
                (
                    artifact("large.bin", self._write(source_root, "large.bin", b"xx")),
                    "size",
                ),
            ):
                limit = 1 if message == "size" else 1024
                with self.assertRaises(ArtifactTransferError):
                    transfer.register_outgoing(
                        ref,
                        source_root=source_root,
                        allowed_requesters=["server"],
                        stage="stage-one",
                        direction="site_to_central",
                        max_file_bytes=limit,
                    )

            with self.assertRaises(ValueError):
                artifact("../escape.bin", outside)

    def test_hard_link_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source_root = os.path.join(output_dir, "author")
            os.mkdir(source_root, 0o700)
            original = self._write(source_root, "original.bin", b"content")
            linked = os.path.join(source_root, "linked.bin")
            os.link(original, linked)
            transfer = ArtifactTransfer()
            self._initialize(transfer, output_dir)

            with self.assertRaisesRegex(ArtifactTransferError, "hard-linked"):
                transfer.register_outgoing(
                    artifact("linked.bin", linked),
                    source_root=source_root,
                    allowed_requesters=["server"],
                    stage="stage-one",
                    direction="site_to_central",
                    max_file_bytes=1024,
                )

    def test_aggregate_rejection_removes_all_partial_staging(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source_root = os.path.join(output_dir, "author")
            os.mkdir(source_root, 0o700)
            transfer = ArtifactTransfer()
            self._initialize(transfer, output_dir)
            values = [
                artifact("one.bin", self._write(source_root, "one.bin", b"1234")),
                artifact("two.bin", self._write(source_root, "two.bin", b"5678")),
            ]

            original_copy = artifact_transfer_module._copy_regular_file
            destination_existed_after_attempt = []

            def observe_copy(*args, **kwargs):
                try:
                    return original_copy(*args, **kwargs)
                finally:
                    destination_existed_after_attempt.append(os.path.exists(args[1]))

            with (
                patch(
                    "framework.artifact_transfer._copy_regular_file",
                    side_effect=observe_copy,
                ),
                self.assertRaisesRegex(ArtifactTransferError, "Aggregate"),
            ):
                prepare_outgoing_artifacts(
                    values,
                    transfer=transfer,
                    source_root=source_root,
                    allowed_requesters=["server"],
                    stage="stage-one",
                    direction="site_to_central",
                    max_file_bytes=8,
                    max_total_bytes=7,
                )

            self.assertEqual(destination_existed_after_attempt, [True, False])
            self.assertEqual(transfer._registry, {})
            self.assertEqual(os.listdir(transfer._outgoing_dir), [])

    def test_sender_shutdown_cancels_new_artifacts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source_root = os.path.join(output_dir, "author")
            os.mkdir(source_root, 0o700)
            transfer = ArtifactTransfer()
            self._initialize(transfer, output_dir)
            transfer._shutting_down = True
            with self.assertRaisesRegex(ArtifactTransferError, "shutting down"):
                transfer.register_outgoing(
                    artifact(
                        "cancelled.bin",
                        self._write(source_root, "cancelled.bin", b"bytes"),
                    ),
                    source_root=source_root,
                    allowed_requesters=["server"],
                    stage="stage-one",
                    direction="site_to_central",
                    max_file_bytes=1024,
                )

    def test_concurrent_shutdown_rejects_post_copy_registration(self):
        copy_completed = threading.Barrier(2)
        shutdown_completed = threading.Barrier(2)

        with tempfile.TemporaryDirectory() as output_dir:
            source_root = os.path.join(output_dir, "author")
            os.mkdir(source_root, 0o700)
            source = self._write(source_root, "artifact.bin", b"content")
            transfer = ArtifactTransfer()
            self._initialize(transfer, output_dir)
            original_copy = artifact_transfer_module._copy_regular_file
            outcome = []

            def pause_after_copy(*args, **kwargs):
                result = original_copy(*args, **kwargs)
                copy_completed.wait()
                shutdown_completed.wait()
                return result

            def register():
                try:
                    transfer.register_outgoing(
                        artifact("artifact.bin", source),
                        source_root=source_root,
                        allowed_requesters=["server"],
                        stage="stage-one",
                        direction="site_to_central",
                        max_file_bytes=1024,
                    )
                except Exception as error:
                    outcome.append(error)

            with patch(
                "framework.artifact_transfer._copy_regular_file",
                side_effect=pause_after_copy,
            ):
                worker = threading.Thread(target=register)
                worker.start()
                copy_completed.wait()
                transfer.handle_event(EventType.END_RUN, object())
                shutdown_completed.wait()
                worker.join()

            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], ArtifactTransferError)
            self.assertRegex(str(outcome[0]), "shutting down")
            self.assertEqual(transfer._registry, {})
            self.assertFalse(os.path.exists(transfer._artifact_root))

    def test_stream_requires_nvflare_p2p_security(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source = self._write(output_dir, "source.bin", b"bytes")
            transfer = ArtifactTransfer()
            self._initialize(transfer, output_dir)
            record = SimpleNamespace(path=source, direction="central_to_site")
            fl_ctx = SimpleNamespace(get_prop=lambda *_args, **_kwargs: False)
            with patch(
                "framework.artifact_transfer.FileStreamer.stream_file",
                return_value="done",
            ) as stream_file:
                transfer.do_stream("site1", {}, fl_ctx, {}, record)
            self.assertIs(stream_file.call_args.kwargs["secure"], True)

    def test_only_nvflare_simulator_mode_disables_p2p_certificate_exchange(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source = self._write(output_dir, "source.bin", b"bytes")
            transfer = ArtifactTransfer()
            self._initialize(transfer, output_dir)
            transfer._simulate_mode = True
            record = SimpleNamespace(path=source, direction="central_to_site")
            fl_ctx = SimpleNamespace(get_prop=lambda *_args, **_kwargs: True)
            with patch(
                "framework.artifact_transfer.FileStreamer.stream_file",
                return_value="done",
            ) as stream_file:
                transfer.do_stream("site1", {}, fl_ctx, {}, record)
            self.assertIs(stream_file.call_args.kwargs["secure"], False)

    def test_simulator_exception_cannot_be_enabled_by_an_arbitrary_engine_name(self):
        class SimulatorEngine:
            pass

        fl_ctx = SimpleNamespace(
            get_prop=lambda *_args, **_kwargs: False,
            get_engine=lambda: SimulatorEngine(),
        )
        self.assertFalse(_is_nvflare_simulator(fl_ctx))

    @staticmethod
    def _write(directory, name, content):
        path = os.path.join(directory, name)
        Path(path).write_bytes(content)
        return path


class ArtifactQuotaTests(unittest.TestCase):
    def test_concurrent_participants_cannot_overcommit_round_quota(self):
        class Spec:
            max_artifact_total_bytes = 1000

        class Aggregator(ComputationAggregator):
            SPEC = Spec()

        aggregator = Aggregator()
        barrier = threading.Barrier(2)
        outcomes = []

        def reserve(site_id, token):
            barrier.wait()
            try:
                aggregator._reserve_artifact_bytes(0, site_id, {token: {"size": 600}})
                outcomes.append("accepted")
            except ArtifactTransferError:
                outcomes.append("rejected")

        threads = [
            threading.Thread(target=reserve, args=("site1", "a" * 32)),
            threading.Thread(target=reserve, args=("site2", "b" * 32)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertCountEqual(outcomes, ["accepted", "rejected"])
        self.assertEqual(sum(aggregator.artifact_reservations[0].values()), 600)

    def test_fresh_retry_token_consumes_cumulative_quota(self):
        class Spec:
            max_artifact_total_bytes = 1000

        class Aggregator(ComputationAggregator):
            SPEC = Spec()

        aggregator = Aggregator()
        aggregator._reserve_artifact_bytes(0, "site1", {"a" * 32: {"size": 600}})
        with self.assertRaises(ArtifactTransferError):
            aggregator._reserve_artifact_bytes(0, "site1", {"b" * 32: {"size": 600}})

    def test_indeterminate_retrieval_keeps_round_quota_reserved(self):
        class Spec:
            max_artifact_total_bytes = 1000
            max_artifact_bytes = 1000
            artifact_timeout = 0.01
            artifact_retries = 0

        class Aggregator(ComputationAggregator):
            SPEC = Spec()

        aggregator = Aggregator()
        aggregator._current_step_name = lambda _round: "stage-one"
        result = Shareable(
            {
                "result": {},
                "__neuroflame_artifacts__": [_manifest(b"x")],
            }
        )
        result.set_header(
            ReservedHeaderKey.PEER_PROPS,
            {ReservedKey.IDENTITY_NAME: "site1"},
        )
        fl_ctx = SimpleNamespace(
            get_prop=lambda key, default=None: {
                "CURRENT_ROUND": 0,
                "COMPUTATION_PARAMETERS": {},
            }.get(key, default)
        )

        with (
            patch("framework.aggregator.get_artifact_transfer", return_value=object()),
            patch(
                "framework.aggregator.materialize_incoming_artifacts",
                side_effect=ArtifactTransferError(
                    "indeterminate transfer", indeterminate=True
                ),
            ),
        ):
            with self.assertRaises(ArtifactTransferError):
                aggregator.accept(result, fl_ctx)

        self.assertEqual(sum(aggregator.artifact_reservations[0].values()), 1)


class ArtifactReceiveTests(unittest.TestCase):
    def _receiver(self, responses, output_dir):
        receiver = _QueuedReceiver(responses)
        with patch.dict(os.environ, {"OUTPUT_DIR": output_dir}):
            receiver._initialize_run(object())
        return receiver

    def test_hash_mismatch_is_retried_and_duplicate_delivery_is_idempotent(self):
        content = b"verified-content"
        with tempfile.TemporaryDirectory() as output_dir:
            receiver = self._receiver(
                [(ReturnCode.OK, b"corrupt"), (ReturnCode.OK, content)], output_dir
            )
            manifest = _manifest(content)

            first = receiver.retrieve_artifact(
                from_site="site1",
                manifest=manifest,
                fl_ctx=object(),
                timeout=1,
                retries=1,
                max_file_bytes=1024,
            )
            second = receiver.retrieve_artifact(
                from_site="site1",
                manifest=manifest,
                fl_ctx=object(),
                timeout=1,
                retries=1,
                max_file_bytes=1024,
            )

            self.assertEqual(Path(first).read_bytes(), content)
            self.assertEqual(first, second)
            self.assertEqual(receiver.retrieve_calls, 2)

    def test_truncated_timeout_and_shutdown_fail_without_promoted_files(self):
        content = b"complete"
        cases = (
            ([(ReturnCode.OK, b"short")], False),
            ([(ReturnCode.TIMEOUT, None)], False),
            ([(ReturnCode.OK, content)], True),
        )
        for responses, shutting_down in cases:
            with self.subTest(responses=responses, shutting_down=shutting_down):
                with tempfile.TemporaryDirectory() as output_dir:
                    receiver = self._receiver(responses, output_dir)
                    receiver._shutting_down = shutting_down
                    with self.assertRaises(ArtifactTransferError):
                        receiver.retrieve_artifact(
                            from_site="site1",
                            manifest=_manifest(content),
                            fl_ctx=object(),
                            timeout=0.01,
                            retries=0,
                            max_file_bytes=1024,
                        )
                    promoted = list(Path(receiver._incoming_dir).glob("*/*"))
                    self.assertEqual(promoted, [])

    def test_timeout_is_not_retried_while_stream_completion_is_indeterminate(self):
        with tempfile.TemporaryDirectory() as output_dir:
            receiver = self._receiver([(ReturnCode.TIMEOUT, None)], output_dir)
            with self.assertRaises(ArtifactTransferError):
                receiver.retrieve_artifact(
                    from_site="site1",
                    manifest=_manifest(),
                    fl_ctx=object(),
                    timeout=0.01,
                    retries=3,
                    max_file_bytes=1024,
                )
            self.assertEqual(receiver.retrieve_calls, 1)

    def test_delivered_request_with_lost_reply_retains_tombstone_and_removes_late_file(
        self,
    ):
        class LostReplyEngine:
            request = None

            def send_aux_request(self, **kwargs):
                self.request = kwargs["request"]
                raise RuntimeError("reply lost")

        with tempfile.TemporaryDirectory() as output_dir:
            receiver = ArtifactTransfer()
            with patch.dict(os.environ, {"OUTPUT_DIR": output_dir}):
                receiver._initialize_run(object())
            engine = LostReplyEngine()
            fl_ctx = SimpleNamespace(
                get_engine=lambda: engine,
                get_run_abort_signal=lambda: None,
            )

            rc, path, indeterminate = receiver.retrieve(
                "site1", fl_ctx, 0.01, **_manifest()
            )

            self.assertEqual(rc, ReturnCode.EXECUTION_EXCEPTION)
            self.assertIsNone(path)
            self.assertTrue(indeterminate)
            tx_id = engine.request[_STREAM_TRANSACTION_KEY]
            self.assertIn(tx_id, receiver._transactions)

            descriptor, late_path = tempfile.mkstemp(dir=receiver._incoming_dir)
            os.close(descriptor)
            with (
                patch(
                    "framework.artifact_transfer.FileStreamer.get_rc",
                    return_value=ReturnCode.OK,
                ),
                patch(
                    "framework.artifact_transfer.FileStreamer.get_file_location",
                    return_value=late_path,
                ),
            ):
                receiver._on_stream_done({_STREAM_TRANSACTION_KEY: tx_id}, fl_ctx)

            self.assertFalse(os.path.exists(late_path))
            self.assertNotIn(tx_id, receiver._transactions)

    def test_terminal_shutdown_clears_never_completed_tombstone_and_staging(self):
        class NoReplyEngine:
            def send_aux_request(self, **_kwargs):
                raise RuntimeError("transport stopped")

        with tempfile.TemporaryDirectory() as output_dir:
            receiver = ArtifactTransfer()
            with patch.dict(os.environ, {"OUTPUT_DIR": output_dir}):
                receiver._initialize_run(object())
            fl_ctx = SimpleNamespace(
                get_engine=lambda: NoReplyEngine(),
                get_run_abort_signal=lambda: None,
            )
            _rc, _path, indeterminate = receiver.retrieve(
                "site1", fl_ctx, 0.01, **_manifest()
            )
            self.assertTrue(indeterminate)
            self.assertEqual(len(receiver._transactions), 1)

            receiver.handle_event(EventType.END_RUN, fl_ctx)

            self.assertEqual(receiver._transactions, {})
            self.assertFalse(os.path.exists(receiver._artifact_root))

    def test_concurrent_shutdown_prevents_transaction_insertion(self):
        engine_requested = threading.Barrier(2)
        shutdown_completed = threading.Barrier(2)

        class RacingContext:
            def get_engine(self):
                engine_requested.wait()
                shutdown_completed.wait()
                return SimpleNamespace()

            @staticmethod
            def get_run_abort_signal():
                return None

        with tempfile.TemporaryDirectory() as output_dir:
            receiver = ArtifactTransfer()
            with patch.dict(os.environ, {"OUTPUT_DIR": output_dir}):
                receiver._initialize_run(object())
            fl_ctx = RacingContext()
            result = []
            worker = threading.Thread(
                target=lambda: result.append(
                    receiver.retrieve("site1", fl_ctx, 0.01, **_manifest())
                )
            )
            worker.start()
            engine_requested.wait()

            receiver.handle_event(EventType.END_RUN, fl_ctx)
            shutdown_completed.wait()
            worker.join()

            self.assertEqual(result, [(ReturnCode.TASK_ABORTED, None, False)])
            self.assertEqual(receiver._transactions, {})
            self.assertFalse(os.path.exists(receiver._artifact_root))

    def test_cleanup_waits_for_late_retrieval_completion(self):
        with tempfile.TemporaryDirectory() as output_dir:
            receiver = self._receiver([], output_dir)
            receiver._transactions["pending"] = threading.Event()
            receiver.cleanup()
            self.assertTrue(os.path.isdir(receiver._artifact_root))
            receiver._finish_transaction("pending")
            self.assertFalse(os.path.exists(receiver._artifact_root))

    def test_received_path_swap_cannot_hash_or_promote_symlink_target(self):
        content = b"verified"
        with tempfile.TemporaryDirectory() as directory:
            temporary = os.path.join(directory, "temporary")
            final = os.path.join(directory, "final")
            outside = os.path.join(directory, "outside")
            Path(temporary).write_bytes(content)
            Path(outside).write_bytes(b"outside")
            outside_mode = os.stat(outside).st_mode
            real_lstat = os.lstat
            swapped = False

            def swap_before_promotion(path):
                nonlocal swapped
                if path == temporary and not swapped:
                    swapped = True
                    os.unlink(temporary)
                    os.symlink(outside, temporary)
                return real_lstat(path)

            with patch(
                "framework.artifact_transfer.os.lstat",
                side_effect=swap_before_promotion,
            ):
                promoted = _promote_verified_file(
                    temporary,
                    final,
                    len(content),
                    hashlib.sha256(content).hexdigest(),
                )

            self.assertFalse(promoted)
            self.assertFalse(os.path.lexists(final))
            self.assertEqual(Path(outside).read_bytes(), b"outside")
            self.assertEqual(os.stat(outside).st_mode, outside_mode)

    def test_integrity_failure_reports_safe_specific_reason(self):
        content = b"received"
        with tempfile.TemporaryDirectory() as output_dir:
            receiver = self._receiver([(ReturnCode.OK, content)], output_dir)
            manifest = _manifest(b"expected")

            with self.assertRaisesRegex(
                ArtifactTransferError,
                "received SHA-256 does not match the manifest",
            ):
                receiver.retrieve_artifact(
                    from_site="site1",
                    manifest=manifest,
                    fl_ctx=object(),
                    timeout=1,
                    retries=0,
                    max_file_bytes=1024,
                )

    def test_promotion_failure_reports_errno_without_path(self):
        content = b"verified"
        with tempfile.TemporaryDirectory() as directory:
            temporary = os.path.join(directory, "sensitive-name")
            final = os.path.join(directory, "final")
            Path(temporary).write_bytes(content)
            error = PermissionError(1, "Operation not permitted", temporary)

            with patch(
                "framework.artifact_transfer.os.fchmod",
                side_effect=error,
            ):
                promoted, detail = (
                    artifact_transfer_module._promote_verified_file_with_reason(
                        temporary,
                        final,
                        len(content),
                        hashlib.sha256(content).hexdigest(),
                    )
                )

            self.assertFalse(promoted)
            self.assertEqual(
                detail,
                "setting private file permissions failed "
                "(errno 1: Operation not permitted)",
            )
            self.assertNotIn(temporary, detail)

    def test_manifest_mismatch_and_destination_symlink_are_rejected(self):
        content = b"content"
        with tempfile.TemporaryDirectory() as output_dir:
            receiver = self._receiver([(ReturnCode.OK, content)], output_dir)
            manifest = _manifest(content)
            tagged = {
                "__neuroflame_type__": "neuroflame.artifact",
                "value": dict(manifest),
            }
            altered = dict(manifest)
            altered["sha256"] = "0" * 64
            with self.assertRaisesRegex(ArtifactTransferError, "does not match"):
                materialize_incoming_artifacts(
                    tagged,
                    [altered],
                    transfer=receiver,
                    from_site="site1",
                    fl_ctx=object(),
                    expected_stage="stage-one",
                    expected_direction="site_to_central",
                    timeout=1,
                    retries=0,
                    max_file_bytes=1024,
                    max_total_bytes=1024,
                )

            source_bucket = hashlib.sha256(b"site1").hexdigest()[:16]
            bucket_path = os.path.join(receiver._incoming_dir, source_bucket)
            os.symlink(output_dir, bucket_path)
            with self.assertRaisesRegex(ArtifactTransferError, "private directory"):
                receiver.retrieve_artifact(
                    from_site="site1",
                    manifest=manifest,
                    fl_ctx=object(),
                    timeout=1,
                    retries=0,
                    max_file_bytes=1024,
                )

    def test_cleanup_removes_successful_and_partial_transfer_directories(self):
        with tempfile.TemporaryDirectory() as output_dir:
            receiver = self._receiver([], output_dir)
            artifact_root = receiver._artifact_root
            output_artifact_root = receiver._output_artifact_root
            author_root = Path(output_artifact_root, "author", "round-1")
            author_root.mkdir(parents=True)
            Path(author_root, "intermediate.bin").write_bytes(b"intermediate")
            Path(receiver._incoming_dir, "partial").write_bytes(b"partial")

            receiver.cleanup()

            self.assertFalse(os.path.exists(artifact_root))
            self.assertFalse(os.path.exists(output_artifact_root))

    def test_later_failure_removes_artifacts_already_received_for_payload(self):
        first_content = b"first"
        second_content = b"second"
        first_manifest = _manifest(first_content, "first.bin")
        second_manifest = _manifest(second_content, "second.bin")
        second_manifest["transfer_id"] = "b" * 32
        tagged = [
            {
                "__neuroflame_type__": "neuroflame.artifact",
                "value": dict(first_manifest),
            },
            {
                "__neuroflame_type__": "neuroflame.artifact",
                "value": dict(second_manifest),
            },
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            receiver = self._receiver(
                [(ReturnCode.OK, first_content), (ReturnCode.TIMEOUT, None)],
                output_dir,
            )
            with self.assertRaises(ArtifactTransferError):
                materialize_incoming_artifacts(
                    tagged,
                    [first_manifest, second_manifest],
                    transfer=receiver,
                    from_site="site1",
                    fl_ctx=object(),
                    expected_stage="stage-one",
                    expected_direction="site_to_central",
                    timeout=0.01,
                    retries=0,
                    max_file_bytes=1024,
                    max_total_bytes=2048,
                )
            self.assertEqual(list(Path(receiver._incoming_dir).glob("*/*")), [])


if __name__ == "__main__":
    unittest.main()
