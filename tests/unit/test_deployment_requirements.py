import unittest
from pathlib import Path

from packaging.requirements import Requirement


class DeploymentRequirementsTests(unittest.TestCase):
    def test_windows_only_dependency_is_excluded_from_linux_install(self):
        root = Path(__file__).resolve().parents[2]
        requirements = [
            Requirement(line.strip())
            for line in (root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        linux = {r.name.lower() for r in requirements if r.marker is None or r.marker.evaluate({"sys_platform": "linux"})}
        windows = {r.name.lower() for r in requirements if r.marker is None or r.marker.evaluate({"sys_platform": "win32"})}
        self.assertNotIn("pywin32", linux)
        self.assertIn("pywin32", windows)


if __name__ == "__main__":
    unittest.main()
