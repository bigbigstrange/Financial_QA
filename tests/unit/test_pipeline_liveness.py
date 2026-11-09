import unittest
from scripts.pipeline_liveness import classify, ROOT


class PipelineLivenessTests(unittest.TestCase):
    def test_real_script_in_project(self):
        self.assertEqual(classify(["python.exe", "-u", "scripts/complete_full_parse.py"], ROOT), "全量解析流水线")

    def test_inspection_command_is_not_a_job(self):
        self.assertIsNone(classify(["python.exe", "-c", "scripts/complete_full_parse.py"], ROOT))

    def test_other_project_is_not_a_job(self):
        self.assertIsNone(classify(["python.exe", "scripts/complete_full_parse.py"], ROOT.parent))
