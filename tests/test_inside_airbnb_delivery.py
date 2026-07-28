from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inside_airbnb_manifest import file_record
from inside_airbnb_release_gate import finalize_gate
from run_inside_airbnb_pipeline import pipeline_commands


class DeliveryPipelineTest(unittest.TestCase):
    def test_ci_plan_has_no_network_or_raw_download_commands(self) -> None:
        commands = pipeline_commands("ci")
        executed_scripts = [
            command.arguments[1]
            for command in commands
            if len(command.arguments) > 1
            and not command.arguments[1].startswith("-")
        ]
        self.assertNotIn("inside_airbnb_phase0.py", executed_scripts)
        self.assertNotIn(
            "inside_airbnb_snapshot_discovery.py", executed_scripts
        )
        self.assertEqual(
            [command.name for command in commands], ["compile", "unit_tests"]
        )

    def test_snapshot_monitor_is_read_only_and_actionable(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "sydney-snapshot-monitor.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("schedule:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("--fail-on-action-required", workflow)
        self.assertIn("inside_airbnb_snapshot_discovery.py", workflow)
        self.assertNotIn("inside_airbnb_phase0.py", workflow)
        self.assertNotIn("run_inside_airbnb_pipeline.py refresh", workflow)
        self.assertNotIn("data/raw", workflow)

    def test_release_gate_blocks_when_any_check_fails(self) -> None:
        result = finalize_gate(
            "production",
            [
                {"name": "one", "passed": True},
                {"name": "two", "passed": False},
            ],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_checks"], ["two"])

    def test_manifest_file_record_includes_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.txt"
            path.write_text("evidence", encoding="utf-8")
            record = file_record(path)
            self.assertTrue(record["exists"])
            self.assertEqual(len(record["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
