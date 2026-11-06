import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from framework.artifacts import ArtifactRef
from framework.cache import JsonStateStore
from framework.serialization import (
    DataFrameSplitJsonCodec,
    NumpyArrayCodec,
    deserialize_value,
    serialize_value,
)

try:
    import numpy
except ImportError:
    numpy = None

try:
    import pandas
except ImportError:
    pandas = None


@dataclass
class ChildValue:
    count: int


@dataclass
class NestedValue:
    child: ChildValue
    optional_child: Optional[ChildValue]
    alternative: Union[ChildValue, str]
    children: Dict[str, ChildValue]
    labels: List[str]
    coordinate: Tuple[int, str]


@dataclass
class ValueWithDefaults:
    name: str
    child: Optional[ChildValue] = None


if numpy is not None:

    @dataclass
    class ArrayValue:
        values: numpy.ndarray
        optional_values: Optional[numpy.ndarray] = None


if pandas is not None:

    @dataclass
    class FrameValue:
        table: pandas.DataFrame
        optional_table: Optional[pandas.DataFrame] = None


class DataclassSerializationTests(unittest.TestCase):
    def test_nested_dataclasses_collections_optional_and_union_round_trip(self):
        original = NestedValue(
            child=ChildValue(count=1),
            optional_child=ChildValue(count=2),
            alternative="raw",
            children={"site1": ChildValue(count=3)},
            labels=["a", "b"],
            coordinate=(4, "x"),
        )

        payload = serialize_value(original)
        restored = deserialize_value(payload, NestedValue)

        self.assertEqual(restored, original)
        self.assertIsInstance(restored.child, ChildValue)
        self.assertIsInstance(restored.optional_child, ChildValue)
        self.assertIsInstance(restored.coordinate, tuple)
        json.dumps(payload)

    def test_missing_optional_field_uses_dataclass_default(self):
        restored = deserialize_value({"name": "example"}, ValueWithDefaults)

        self.assertEqual(restored, ValueWithDefaults(name="example"))

    def test_plain_json_values_remain_plain(self):
        original = {"site1": [1, 2.5, True, None, {"label": "value"}]}

        payload = serialize_value(original)

        self.assertEqual(deserialize_value(payload), original)

    def test_paths_and_artifacts_fail_with_transfer_guidance(self):
        with self.assertRaisesRegex(TypeError, "framework.artifact"):
            serialize_value(Path("image.nii"))
        with self.assertRaisesRegex(TypeError, "returned from a computation step"):
            serialize_value(ArtifactRef(name="image.nii", path="image.nii"))


@unittest.skipIf(numpy is None, "NumPy is not installed in this Python environment")
class NumpySerializationTests(unittest.TestCase):
    def test_numpy_scalars_become_plain_json_values(self):
        payload = serialize_value(
            {
                "integer": numpy.int64(3),
                "floating": numpy.float32(1.5),
                "boolean": numpy.bool_(True),
            }
        )

        self.assertEqual(
            payload,
            {"integer": 3, "floating": 1.5, "boolean": True},
        )
        json.dumps(payload)

    def test_array_round_trips_typed_and_untyped_without_metadata(self):
        source = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)[:, ::2]
        original = ArrayValue(values=source)

        payload = serialize_value(original)
        typed = deserialize_value(payload, ArrayValue)
        untyped = deserialize_value(payload)

        numpy.testing.assert_array_equal(typed.values, source)
        numpy.testing.assert_array_equal(untyped["values"], source)
        self.assertEqual(typed.values.dtype, source.dtype)
        self.assertTrue(typed.values.flags.writeable)
        self.assertFalse(fields(ArrayValue)[0].metadata)
        json.dumps(payload)

    def test_array_over_inline_limit_is_rejected_on_encode_and_decode(self):
        source = numpy.arange(8, dtype=numpy.float64)
        payload = serialize_value(source)

        with self.assertRaisesRegex(
            ValueError, "exceeding the inline limit.*ArtifactRef"
        ):
            serialize_value(source, max_inline_array_bytes=source.nbytes - 1)
        with self.assertRaisesRegex(
            ValueError, "exceeding the inline limit.*ArtifactRef"
        ):
            deserialize_value(payload, max_inline_array_bytes=source.nbytes - 1)

    def test_object_array_is_rejected(self):
        source = numpy.array([{"not": "portable"}], dtype=object)

        with self.assertRaisesRegex(TypeError, "not safe for inline transport"):
            serialize_value(source)

    def test_legacy_untagged_array_payload_still_decodes_when_typed(self):
        source = numpy.arange(4, dtype=numpy.int16)

        restored = deserialize_value(NumpyArrayCodec.encode(source), numpy.ndarray)

        numpy.testing.assert_array_equal(restored, source)


@unittest.skipIf(pandas is None, "pandas is not installed in this Python environment")
class DataFrameSerializationTests(unittest.TestCase):
    def test_dataframe_round_trips_typed_and_untyped_without_metadata(self):
        source = pandas.DataFrame(
            {"age": [30, 40], "group": ["control", "patient"]},
            index=["subject1", "subject2"],
        )
        original = FrameValue(table=source)

        payload = serialize_value(original)
        typed = deserialize_value(payload, FrameValue)
        untyped = deserialize_value(payload)

        pandas.testing.assert_frame_equal(typed.table, source)
        pandas.testing.assert_frame_equal(untyped["table"], source)
        self.assertFalse(fields(FrameValue)[0].metadata)
        json.dumps(payload)

    def test_dataframe_state_store_round_trip_needs_no_codec_configuration(self):
        source = pandas.DataFrame({"value": [1.5, 2.5]}, index=["a", "b"])
        state = FrameValue(table=source)

        with tempfile.TemporaryDirectory() as output_dir:
            store = JsonStateStore(output_dir)
            store.save_state(state)
            restored = store.load_state(FrameValue)

        pandas.testing.assert_frame_equal(restored.table, source)

    def test_legacy_untagged_dataframe_payload_still_decodes_when_typed(self):
        source = pandas.DataFrame({"value": [1, 2]})

        restored = deserialize_value(
            DataFrameSplitJsonCodec.encode(source), pandas.DataFrame
        )

        pandas.testing.assert_frame_equal(restored, source)


if __name__ == "__main__":
    unittest.main()
