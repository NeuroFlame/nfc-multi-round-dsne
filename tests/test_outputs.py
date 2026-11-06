import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from framework.serialization import DEFAULT_MAX_INLINE_ARRAY_BYTES
from framework.writers import write_standard_outputs

try:
    import pandas
except ImportError:
    pandas = None


@dataclass
class Summary:
    value: int


class CsvValue:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def to_csv(self, path, **kwargs):
        self.calls.append((path, kwargs))
        with open(path, "w", encoding="utf-8") as output_file:
            output_file.write(self.content)


class StandardOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.runtime = SimpleNamespace(
            output_dir=self.temp_dir.name,
            max_inline_array_bytes=DEFAULT_MAX_INLINE_ARRAY_BYTES,
        )

    def test_filename_mapping_writes_json_csv_tsv_and_text(self):
        csv_value = CsvValue("column\n1\n")
        tsv_value = CsvValue("column\n1\n")

        write_standard_outputs(
            {
                "nested/results.json": Summary(value=3),
                "table.csv": csv_value,
                "table.tsv": tsv_value,
                "index.html": "<h1>Result</h1>",
                "notes.txt": "complete",
            },
            self.runtime,
        )

        with open(
            os.path.join(self.temp_dir.name, "nested", "results.json"),
            "r",
            encoding="utf-8",
        ) as output_file:
            self.assertEqual(json.load(output_file), {"value": 3})
        with open(
            os.path.join(self.temp_dir.name, "index.html"), encoding="utf-8"
        ) as output_file:
            self.assertEqual(output_file.read(), "<h1>Result</h1>")
        with open(
            os.path.join(self.temp_dir.name, "notes.txt"), encoding="utf-8"
        ) as output_file:
            self.assertEqual(output_file.read(), "complete")
        self.assertEqual(csv_value.calls[0][1], {})
        self.assertEqual(tsv_value.calls[0][1], {"sep": "\t"})

    def test_output_filename_must_stay_inside_output_directory(self):
        for file_name in (
            "../outside.json",
            os.path.join(os.sep, "tmp", "outside.json"),
        ):
            with self.subTest(file_name=file_name):
                with self.assertRaisesRegex(ValueError, "output_dir"):
                    write_standard_outputs({file_name: {}}, self.runtime)

    def test_unsupported_extension_points_to_direct_write_escape_hatch(self):
        with self.assertRaisesRegex(ValueError, "using output_dir"):
            write_standard_outputs({"result.nii": object()}, self.runtime)

    def test_old_writer_oriented_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported extension '<none>'"):
            write_standard_outputs({"json_files": {"result.json": {}}}, self.runtime)

    def test_csv_output_requires_to_csv(self):
        with self.assertRaisesRegex(TypeError, "requires a pandas DataFrame"):
            write_standard_outputs({"table.csv": [[1, 2]]}, self.runtime)


@unittest.skipIf(pandas is None, "pandas is not installed in this Python environment")
class PandasOutputTests(unittest.TestCase):
    def test_dataframe_index_name_controls_csv_header(self):
        dataframe = pandas.DataFrame({"value": [1.5]}, index=["region1"])
        dataframe.index.name = "ROI"

        with tempfile.TemporaryDirectory() as output_dir:
            runtime = SimpleNamespace(
                output_dir=output_dir,
                max_inline_array_bytes=DEFAULT_MAX_INLINE_ARRAY_BYTES,
            )
            write_standard_outputs({"statistics.csv": dataframe}, runtime)
            with open(
                os.path.join(output_dir, "statistics.csv"), encoding="utf-8"
            ) as output_file:
                self.assertEqual(output_file.read(), "ROI,value\nregion1,1.5\n")


if __name__ == "__main__":
    unittest.main()
