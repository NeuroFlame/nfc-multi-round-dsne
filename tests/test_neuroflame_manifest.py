import unittest

from scripts.neuroflame_manifest import merge_compatibility, validate_manifest


def _manifest() -> dict:
    return {
        "manifestVersion": 1,
        "computation": {"version": "1.2.3"},
        "compatibility": {
            "computationApiVersion": "0.1.0",
            "boilerplateVersion": "0.1.0",
        },
        "image": {
            "title": "example",
            "repository": "example/computation",
            "floatingTag": "latest",
            "tagPrefix": "",
            "source": "https://example.test/computation",
        },
    }


class NeuroFlameManifestTests(unittest.TestCase):
    def test_manifest_requires_supported_schema_version(self):
        manifest = _manifest()
        manifest["manifestVersion"] = 2

        with self.assertRaisesRegex(ValueError, "manifestVersion must be 1"):
            validate_manifest(manifest)

    def test_manifest_requires_strict_semantic_versions(self):
        manifest = _manifest()
        manifest["computation"]["version"] = "v1.2.3"

        with self.assertRaisesRegex(ValueError, "computation.version"):
            validate_manifest(manifest)

    def test_compatibility_merge_preserves_author_owned_fields(self):
        source = _manifest()
        source["compatibility"]["computationApiVersion"] = "0.2.0"
        source["compatibility"]["boilerplateVersion"] = "0.3.0"
        target = _manifest()
        target["computation"]["version"] = "4.5.6"
        target["image"]["repository"] = "author/custom"

        merged = merge_compatibility(source, target)

        self.assertEqual(merged["computation"]["version"], "4.5.6")
        self.assertEqual(merged["image"]["repository"], "author/custom")
        self.assertEqual(merged["compatibility"], source["compatibility"])


if __name__ == "__main__":
    unittest.main()
