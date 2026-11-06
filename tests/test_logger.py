import logging
import os
import sys
import tempfile
import unittest

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from framework.logger import close_computation_logger, create_computation_logger


class ComputationLoggerTests(unittest.TestCase):
    def test_framework_provides_a_standard_filtered_file_logger(self):
        with tempfile.TemporaryDirectory() as output_dir:
            logger = create_computation_logger(
                output_dir,
                "site1.log",
                {"log_level": "warning"},
            )

            self.assertIsInstance(logger, logging.Logger)
            logger.info("hidden")
            logger.warning("visible %s", "message")
            close_computation_logger(logger)

            self.assertEqual(logger.handlers, [])
            with open(
                os.path.join(output_dir, "site1.log"), encoding="utf-8"
            ) as log_file:
                self.assertEqual(log_file.read(), "visible message\n")

    def test_invalid_level_uses_info_default(self):
        with tempfile.TemporaryDirectory() as output_dir:
            logger = create_computation_logger(
                output_dir,
                "site1.log",
                {"log_level": "not-a-level"},
            )

            self.assertEqual(logger.level, logging.INFO)
            close_computation_logger(logger)


if __name__ == "__main__":
    unittest.main()
