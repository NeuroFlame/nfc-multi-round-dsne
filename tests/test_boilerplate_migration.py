import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import migrate_computation
from scripts.neuroflame_manifest import load_manifest, write_manifest


def _manifest(
    *, computation_version: str, api_version: str, boilerplate_version: str, title: str
) -> dict:
    return {
        "manifestVersion": 1,
        "computation": {"version": computation_version},
        "compatibility": {
            "computationApiVersion": api_version,
            "boilerplateVersion": boilerplate_version,
        },
        "image": {
            "title": title,
            "repository": f"example/{title}",
            "floatingTag": "latest",
            "tagPrefix": "",
            "source": f"https://example.test/{title}",
        },
    }


class RequirementMigrationTests(unittest.TestCase):
    def test_framework_pins_are_updated_and_custom_packages_are_preserved(self):
        source = "nvflare==2.8.0\nnumpy==1.24.4\n"
        target = "nvflare==2.4.0\ncustom-math==7.1\n"

        self.assertEqual(
            migrate_computation.merge_requirements(source, target),
            "nvflare==2.8.0\ncustom-math==7.1\nnumpy==1.24.4\n",
        )


class BoilerplateMigrationTests(unittest.TestCase):
    def test_apply_replaces_only_managed_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            (source / "managed").mkdir(parents=True)
            (source / "managed/current.py").write_text("new\n", encoding="utf-8")
            (source / "wrapper.py").write_text("new wrapper\n", encoding="utf-8")
            (source / "requirements.txt").write_text(
                "nvflare==2.8.0\n", encoding="utf-8"
            )
            (source / ".dockerignore").write_text(".git\n*.tgz\n", encoding="utf-8")
            write_manifest(
                source,
                _manifest(
                    computation_version="9.9.9",
                    api_version="0.2.0",
                    boilerplate_version="0.2.0",
                    title="boilerplate",
                ),
            )

            (target / "managed").mkdir(parents=True)
            (target / "managed/stale.py").write_text("old\n", encoding="utf-8")
            (target / "app/code/computation").mkdir(parents=True)
            author_file = target / "app/code/computation/remote_math.py"
            author_file.write_text("author math\n", encoding="utf-8")
            (target / "requirements.txt").write_text(
                "nvflare==2.4.0\ncustom-math==7.1\n", encoding="utf-8"
            )
            (target / ".dockerignore").write_text("custom-output\n", encoding="utf-8")
            write_manifest(
                target,
                _manifest(
                    computation_version="1.4.2",
                    api_version="0.1.0",
                    boilerplate_version="0.1.0",
                    title="author-computation",
                ),
            )

            with (
                patch.object(
                    migrate_computation,
                    "MANAGED_DIRECTORIES",
                    (Path("managed"),),
                ),
                patch.object(
                    migrate_computation,
                    "MANAGED_FILES",
                    (Path("wrapper.py"),),
                ),
            ):
                migrate_computation.apply_boilerplate(source, target)

            self.assertEqual(author_file.read_text(encoding="utf-8"), "author math\n")
            self.assertFalse((target / "managed/stale.py").exists())
            self.assertEqual(
                (target / "managed/current.py").read_text(encoding="utf-8"),
                "new\n",
            )
            self.assertEqual(
                (target / "requirements.txt").read_text(encoding="utf-8"),
                "nvflare==2.8.0\ncustom-math==7.1\n",
            )
            manifest = load_manifest(target)
            self.assertEqual(manifest["computation"]["version"], "1.4.2")
            self.assertEqual(manifest["image"]["title"], "author-computation")
            self.assertEqual(
                manifest["compatibility"],
                {
                    "computationApiVersion": "0.2.0",
                    "boilerplateVersion": "0.2.0",
                },
            )
            self.assertEqual(migrate_computation.read_version(target), "0.2.0")
            self.assertEqual(
                (target / ".dockerignore").read_text(encoding="utf-8"),
                "custom-output\n.git\n*.tgz\n",
            )

    def test_in_place_mode_requires_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            (target / "app/code/computation").mkdir(parents=True)
            (target / "requirements.txt").write_text("", encoding="utf-8")

            with (
                patch.object(migrate_computation, "planned_changes", return_value=[]),
                self.assertRaisesRegex(ValueError, "requires --force"),
                redirect_stdout(StringIO()),
            ):
                migrate_computation.main([str(target), "--in-place"])


if __name__ == "__main__":
    unittest.main()
