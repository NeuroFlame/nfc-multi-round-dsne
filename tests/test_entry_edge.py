"""Production edge-wrapper process supervision tests."""

import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from system import entry_edge


class EdgeWrapperTests(unittest.TestCase):
    """Verify startup failure and signal propagation at the process boundary."""

    def test_once_startup_failure_is_nonzero(self):
        """Preserve the client exit code from NVFlare's direct once branch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            startup_script = Path(temp_dir, "sub_start.sh")
            startup_script.write_text(
                'if [ "${1:-}" != "--once" ]; then exit 0; fi\nexit 23\n',
                encoding="utf-8",
            )

            with patch.object(entry_edge, "STARTUP_SCRIPT_PATH", str(startup_script)):
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    entry_edge.main()

        self.assertEqual(23, raised.exception.returncode)

    def test_sigterm_is_forwarded_to_client(self):
        """Forward a container SIGTERM to the active NVFlare child process."""
        child = Mock()
        child.pid = 4321
        installed_handlers = {}

        def install_handler(signum, handler):
            previous_handler = installed_handlers.get(signum, signal.SIG_DFL)
            installed_handlers[signum] = handler
            return previous_handler

        def wait_for_signal():
            installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
            return -signal.SIGTERM

        child.poll.return_value = None
        child.wait.side_effect = wait_for_signal

        with (
            patch.object(entry_edge.subprocess, "Popen", return_value=child),
            patch.object(entry_edge.signal, "signal", side_effect=install_handler),
            patch.object(entry_edge.os, "killpg") as killpg,
        ):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                entry_edge.main()

        killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(-signal.SIGTERM, raised.exception.returncode)
