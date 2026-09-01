"""Contract test for the project performance benchmark CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest


class BenchmarkCliTest(unittest.TestCase):
    """The benchmark must emit machine-readable timing results."""

    def test_atprk_benchmark_reports_positive_elapsed_time(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.benchmark_kriging",
                "--method",
                "atprk",
                "--repeats",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(completed.stdout)

        self.assertEqual(payload["atprk"]["runs"], 1)
        self.assertGreater(payload["atprk"]["seconds_per_run"], 0.0)
