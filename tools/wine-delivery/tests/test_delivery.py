"""Delivery failure boundaries; all fixtures are synthetic except pinned ZIP tests."""

import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import validate_delivery as d


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, files=None, *, manifest=None, json_manifest=False, extras=()):
        files = {"run.py": b"print('hello')\n"} if files is None else files
        manifest = (
            {name: d.sha(value) for name, value in files.items()} if manifest is None else manifest
        )
        manifest_name = "SHA256SUMS.json" if json_manifest else "SHA256SUMS.txt"
        if json_manifest:
            data = json.dumps(manifest).encode()
        else:
            data = "".join(f"{value}  {name}\n" for name, value in manifest.items()).encode()
        path = self.root / "fixture.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                for name, content in files.items():
                    archive.writestr("fixture/" + name, content)
                archive.writestr("fixture/" + manifest_name, data)
                for name, value in extras:
                    archive.writestr(name, value)
        spec = {"root": "fixture", "manifest": manifest_name, "sha256": d.sha(path.read_bytes())}
        return path, spec

    def test_delivered_archives_and_manifests(self):
        for spec in d.PACKAGES:
            with self.subTest(archive=spec["archive"]):
                contents = d.inspect_archive(d.ROOT / "inputs" / spec["archive"], spec)
                target = self.root / spec["root"]
                d.extract_new(contents, target)
                self.assertTrue(d.verify_originals(target, contents))

    def test_text_and_json_manifest(self):
        for is_json in (True, False):
            path, spec = self.archive(json_manifest=is_json)
            self.assertIn("run.py", d.inspect_archive(path, spec))

    def test_sha_mismatch(self):
        path, spec = self.archive()
        spec["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pinned"):
            d.inspect_archive(path, spec)

    def test_member_tamper(self):
        path, spec = self.archive(manifest={"run.py": "0" * 64})
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            d.inspect_archive(path, spec)

    def test_manifest_missing_member(self):
        path, spec = self.archive(manifest={})
        with self.assertRaisesRegex(ValueError, "cover"):
            d.inspect_archive(path, spec)

    def test_manifest_extra_member(self):
        path, spec = self.archive(
            manifest={"run.py": d.sha(b"print('hello')\n"), "missing": "0" * 64}
        )
        with self.assertRaisesRegex(ValueError, "cover"):
            d.inspect_archive(path, spec)

    def test_invalid_hash(self):
        path, spec = self.archive(manifest={"run.py": "bad"})
        with self.assertRaisesRegex(ValueError, "invalid manifest hash"):
            d.inspect_archive(path, spec)

    def test_traversal_and_noncanonical_names(self):
        for name in ("../evil", "/evil", "a/../b", "a\\b", "C:evil", "a//b", "a/./b", "a\x01b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                d.safe_name(name)

    def test_wrong_archive_root(self):
        path, spec = self.archive(extras=(("other/file", b"x"),))
        with self.assertRaisesRegex(ValueError, "root"):
            d.inspect_archive(path, spec)

    def test_duplicate_archive_member(self):
        path, spec = self.archive(extras=(("fixture/run.py", b"x"),))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            d.inspect_archive(path, spec)

    def test_case_collision(self):
        path, spec = self.archive({"a": b"1", "A": b"2"})
        with self.assertRaisesRegex(ValueError, "colliding"):
            d.inspect_archive(path, spec)

    def test_file_prefix_collision(self):
        path, spec = self.archive({"a": b"1", "a/b": b"2"})
        with self.assertRaisesRegex(ValueError, "prefix"):
            d.inspect_archive(path, spec)

    def test_symlink_member(self):
        info = zipfile.ZipInfo("fixture/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        path, spec = self.archive(extras=((info, b"outside"),))
        with self.assertRaisesRegex(ValueError, "regular"):
            d.inspect_archive(path, spec)

    def test_archive_symlink_input(self):
        path, spec = self.archive()
        link = self.root / "link.zip"
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "invalid archive input"):
            d.inspect_archive(link, spec)

    def test_count_and_size_limits(self):
        path, spec = self.archive()
        for setting in ("MAX_FILES", "MAX_MEMBER", "MAX_TOTAL", "MAX_ARCHIVE"):
            with (
                self.subTest(setting=setting),
                patch.object(d, setting, 1),
                self.assertRaises(ValueError),
            ):
                d.inspect_archive(path, spec)

    def test_filename_nul_normalization_rejected(self):
        path, spec = self.archive({"run.pyXextra": b"pass\n"})
        raw = path.read_bytes().replace(b"run.pyXextra", b"run.py\0extra")
        path.write_bytes(raw)
        spec["sha256"] = d.sha(raw)
        with self.assertRaisesRegex(ValueError, "normalized or truncated"):
            d.inspect_archive(path, spec)

    def test_unsupported_compression_rejected(self):
        path, spec = self.archive()
        raw = bytearray(path.read_bytes())
        local = raw.find(b"PK\x03\x04")
        central = raw.find(b"PK\x01\x02")
        raw[local + 8 : local + 10] = b"\x63\x00"
        raw[central + 10 : central + 12] = b"\x63\x00"
        path.write_bytes(raw)
        spec["sha256"] = d.sha(raw)
        with self.assertRaisesRegex(ValueError, "unsupported ZIP compression"):
            d.inspect_archive(path, spec)

    def test_python_syntax_checked(self):
        path, spec = self.archive({"broken.py": b"def broken(\n"})
        with self.assertRaises(SyntaxError):
            d.inspect_archive(path, spec)

    def test_extract_refuses_existing_directory(self):
        with self.assertRaises(FileExistsError):
            d.extract_new({"safe": b"x"}, self.root)

    def test_original_modification_detected(self):
        contents = {"safe": b"x"}
        destination = self.root / "new"
        d.extract_new(contents, destination)
        (destination / "safe").write_bytes(b"changed")
        self.assertFalse(d.verify_originals(destination, contents))

    def test_environment_drops_injection(self):
        with patch.dict(
            os.environ,
            {"PYTHONPATH": "evil", "PYTEST_ADDOPTS": "evil", "ATEN_CPU_CAPABILITY": "default"},
        ):
            env = d.clean_env()
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTEST_ADDOPTS", env)
            self.assertNotIn("ATEN_CPU_CAPABILITY", env)
            self.assertEqual(env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")

    def test_json_duplicate_and_nonfinite(self):
        path = self.root / "bad.json"
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                d.read_json(path)

    def report(self):
        spec = d.PACKAGES[0]
        contents = d.inspect_archive(d.ROOT / "inputs" / spec["archive"], spec)
        return json.loads(contents["results/abba-reviewed.json"])

    def test_scientific_failure_is_preserved(self):
        path = self.root / "r.json"
        path.write_text(json.dumps(self.report()))
        result = d.compare_reports(path, path)
        self.assertTrue(result["capture_controls_pass"])
        self.assertFalse(result["replayed"]["comparisons_to_A1"][0]["within_8_ulps"])

    def test_scientific_tamper(self):
        report = self.report()
        report["samples"][0]["scientific_sha256"] = "0" * 64
        path = self.root / "r.json"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "scientific hash"):
            d.scientific_summary(path)

    def test_missing_controls_cannot_pass_vacuously(self):
        report = self.report()
        report["checks"] = {}
        path = self.root / "r.json"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "missing capture controls"):
            d.scientific_summary(path)

    def test_failed_controls_not_scientific_success(self):
        report = self.report()
        report["checks"]["all_four_workers_completed"] = False
        report["capture_valid"] = False
        path = self.root / "r.json"
        path.write_text(json.dumps(report))
        self.assertFalse(d.compare_reports(path, path)["capture_controls_pass"])

    def test_scientific_incomplete_order(self):
        report = self.report()
        report["order"] = ["A"]
        path = self.root / "r.json"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            d.scientific_summary(path)

    def test_schema_plan_and_control_types_rejected(self):
        mutations = (
            lambda r: r.update(plan_id="other"),
            lambda r: r.update(samples={}),
            lambda r: r["checks"].update(A_strict_repeat=1),
            lambda r: r["samples"][0]["scientific"].update(plan_id="other"),
            lambda r: r["samples"][0]["scientific"].update(fixture_sha256="0" * 64),
            lambda r: r["samples"][0]["checks"].clear(),
            lambda r: r.update(comparisons_to_A1=[]),
            lambda r: r.update(capture_valid=False),
            lambda r: r["samples"][0]["checks"].update(inputs_unchanged=False),
            lambda r: r["samples"][0].update(runtime=5),
            lambda r: r["samples"][0]["runtime"].update(cpu_models_and_features=[5]),
        )
        for i, mutate in enumerate(mutations):
            report = self.report()
            mutate(report)
            path = self.root / "r.json"
            path.write_text(json.dumps(report))
            with self.subTest(case=i), self.assertRaises(ValueError):
                d.scientific_summary(path)

    def test_cross_plan_comparison_rejected(self):
        files = []
        for i, spec in enumerate(d.PACKAGES):
            contents = d.inspect_archive(d.ROOT / "inputs" / spec["archive"], spec)
            path = self.root / f"{i}.json"
            path.write_bytes(contents["results/abba-reviewed.json"])
            files.append(path)
        with self.assertRaisesRegex(ValueError, "different scientific plans"):
            d.compare_reports(*files)

    def test_cross_report_hash_difference_is_reported(self):
        before = self.report()
        after = copy.deepcopy(before)
        for sample in after["samples"]:
            sample["scientific"]["stages"][0]["values"][0] += 1.0
            sample["scientific_sha256"] = d.sha(d.canonical(sample["scientific"]))
        left, right = self.root / "left.json", self.root / "right.json"
        left.write_text(json.dumps(before))
        right.write_text(json.dumps(after))
        self.assertFalse(d.compare_reports(left, right)["scientific_hashes_match_delivered"])

    def test_command_nonzero(self):
        result = d.run_command(
            [sys.executable, "-I", "-c", "raise SystemExit(7)"], self.root, self.root, "nonzero", 5
        )
        self.assertEqual(result["returncode"], 7)
        self.assertFalse(result["passed"])

    @unittest.skipUnless(os.name == "posix", "POSIX supervisor")
    def test_command_timeout(self):
        result = d.run_command(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            self.root,
            self.root,
            "timeout",
            0.1,
        )
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["passed"])

    def test_static_cli_and_output_no_overwrite(self):
        destination = self.root / "audit"
        args = [sys.executable, str(d.ROOT / "validate_delivery.py"), "--output", str(destination)]
        first = subprocess.run(args, capture_output=True, timeout=15, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        report = d.read_json(destination / "delivery-report.json")
        self.assertFalse(report["execute_requested"])
        self.assertTrue(report["passed"])
        second = subprocess.run(args, capture_output=True, timeout=15, check=False)
        self.assertNotEqual(second.returncode, 0)

    def test_cli_keeps_missing_input_failure_receipt(self):
        destination = self.root / "audit"
        result = subprocess.run(
            [
                sys.executable,
                str(d.ROOT / "validate_delivery.py"),
                "--archives",
                str(self.root / "missing"),
                "--output",
                str(destination),
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        report = d.read_json(destination / "delivery-report.json")
        self.assertFalse(report["passed"])
        self.assertIn("failure", report)


if __name__ == "__main__":
    unittest.main()
