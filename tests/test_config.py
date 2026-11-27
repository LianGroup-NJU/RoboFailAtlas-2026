import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from failatlas.config import load, resolve, write_new

class ConfigTests(unittest.TestCase):
    def test_nested_overrides_preserve_sibling_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "base.json").write_text(json.dumps({"loss": {"action": .1, "recovery": .05}, "seed": 42}))
            (p / "child.json").write_text(json.dumps({"_base_": "base.json", "loss": {"recovery": .025}}))
            self.assertEqual(resolve(p / "child.json"), {"loss": {"action": .1, "recovery": .025}, "seed": 42})

    def test_relative_bases_ignore_working_directory_and_replace_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p / "tasks").mkdir()
            (p / "base.json").write_text(json.dumps({"frames": [0, 4, 8]}))
            (p / "tasks/run.json").write_text(json.dumps({"_base_": "../base.json", "frames": [2]}))
            self.assertEqual(resolve(p / "tasks/run.json"), {"frames": [2]})

    def test_environment_expands_only_after_child_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "base.json").write_text(json.dumps({"model": "${MISSING}", "data": "${DATA}"}))
            (p / "run.json").write_text(json.dumps({"_base_": "base.json", "model": "/models/base"}))
            with patch.dict(os.environ, {"DATA": "/data"}, clear=True):
                self.assertEqual(load(p / "run.json"), {"model": "/models/base", "data": "/data"})
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "unset_configuration_variable:DATA"):
                    load(p / "run.json")

    def test_cycles_and_missing_bases_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "a.json").write_text('{"_base_": "b.json"}')
            (p / "b.json").write_text('{"_base_": "a.json"}')
            with self.assertRaisesRegex(ValueError, "inheritance_cycle"):
                resolve(p / "a.json")
            (p / "a.json").write_text('{"_base_": "missing.json"}')
            with self.assertRaises(FileNotFoundError):resolve(p / "a.json")

    def test_existing_run_config_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "resolved.json"
            write_new(p, {"seed": 42})
            with self.assertRaises(FileExistsError):write_new(p, {"seed": 1})
            self.assertEqual(json.loads(p.read_text()), {"seed": 42})

    def test_yaml_roundtrip_keeps_hydra_interpolation(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("Optional PyYAML dependency not installed")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "base.yaml").write_text("seed: 42\nmodel: ${oc.env:MODEL}\nnested:\n  rate: 0.000005\n")
            (p / "run.yaml").write_text("_base_: base.yaml\nnested:\n  count: 2\n")
            value = load(p / "run.yaml")
            write_new(p / "resolved.yaml", value)
            self.assertEqual(yaml.safe_load((p / "resolved.yaml").read_text()), value)
            self.assertEqual(value["model"], "${oc.env:MODEL}")
            self.assertEqual(value["nested"], {"rate": .000005, "count": 2})

    def test_ordered_bases_and_child_have_deterministic_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "one.json").write_text('{"seed": 1, "batch": 8}')
            (p / "two.json").write_text('{"seed": 2}')
            (p / "run.json").write_text('{"_base_": ["one.json", "two.json"], "batch": 16}')
            self.assertEqual(resolve(p / "run.json"), {"seed": 2, "batch": 16})
