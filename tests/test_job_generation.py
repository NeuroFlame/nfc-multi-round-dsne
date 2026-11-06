import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(ROOT_DIR, "app", "code")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CODE_DIR)

from computation.spec import SPEC
from framework.workflow import get_task_names

from makeJob import get_spec_task_names, update_tasks_in_client_config
from system.provision.code.create_job import create_job

SOURCE_APP_DIR = os.path.join(ROOT_DIR, "app")
SOURCE_CLIENT_CONFIG = os.path.join(SOURCE_APP_DIR, "config", "config_fed_client.json")


def read_tasks(config_path):
    with open(config_path, encoding="utf-8") as config_file:
        return json.load(config_file)["executors"][0]["tasks"]


class JobGenerationTests(unittest.TestCase):
    def test_source_config_does_not_contain_computation_task_names(self):
        self.assertEqual(read_tasks(SOURCE_CLIENT_CONFIG), [])

    def test_local_job_config_gets_task_names_from_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config_fed_client.json")
            shutil.copy(SOURCE_CLIENT_CONFIG, config_path)

            update_tasks_in_client_config(config_path, get_spec_task_names())

            self.assertEqual(read_tasks(config_path), get_task_names(SPEC.workflow))

    def test_production_job_uses_spec_tasks_and_waits_for_all_participants(self):
        for count in (1, 3, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp_dir:
                job_path = os.path.join(temp_dir, "job")
                create_job(SOURCE_APP_DIR, job_path, min_clients=count)
                config_path = os.path.join(
                    job_path, "app", "config", "config_fed_client.json"
                )
                self.assertEqual(read_tasks(config_path), get_task_names(SPEC.workflow))
                with open(
                    os.path.join(job_path, "meta.json"), encoding="utf-8"
                ) as stream:
                    self.assertEqual(json.load(stream)["min_clients"], count)
                server_config = os.path.join(
                    job_path, "app", "config", "config_fed_server.json"
                )
                with open(server_config, encoding="utf-8") as stream:
                    self.assertEqual(
                        json.load(stream)["workflows"][0]["args"]["min_clients"], count
                    )
        self.assertEqual(read_tasks(SOURCE_CLIENT_CONFIG), [])

    def test_invalid_participant_count_is_rejected_before_creating_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            job_path = os.path.join(temp_dir, "job")
            for count in (0, -1, True, "3", 2.5):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    create_job(SOURCE_APP_DIR, job_path, min_clients=count)
                self.assertFalse(os.path.exists(job_path))


if __name__ == "__main__":
    unittest.main()
