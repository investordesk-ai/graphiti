"""Validate complete pytest collections without relying on a manually maintained passing-test count."""

import importlib.util
import subprocess
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# the real runner is loaded without importing pytest, Graphiti backends or contacting Docker
class RecoveryCollectionTests(unittest.TestCase):

    def setUp(self):
        path = Path(__file__).with_name('run_ingest_recovery.py')
        spec = importlib.util.spec_from_file_location('collection_checked_runner', path)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        self.directory = tempfile.TemporaryDirectory(prefix = 'graphiti-collection-check-', dir = '/tmp')
        self.addCleanup(self.directory.cleanup)
        self.log = Path(self.directory.name)/'unit.log'
        self.arguments = ['run', '--rm', '--network', 'none', 'test-image', 'python', '-m', 'pytest',
                          'tests/test_ingest_recovery_counts.py', '-q']

    # preserve subprocess status independently of the text printed by a process
    def result(self, output, code = 0):
        return SimpleNamespace(returncode = code, stdout = output, stderr = '')

    # 305 is the observed CI result; future suite growth must not require editing the runner again
    def test_collection_is_authoritative_as_tests_are_added(self):
        for count in (1, 305, 417):
            with self.subTest(count = count):
                with patch.object(subprocess, 'run', side_effect = [
                    self.result(f'{count} test' + ('s' if count != 1 else '') + ' collected in 0.1s'),
                    self.result(f'{count} passed, 1 warning in 0.1s'),
                ]) as docker: # patch
                    self.assertEqual(self.runner.suite(self.arguments, self.log), count)
                self.assertEqual(docker.call_args_list[0].args[0], ['docker', *self.arguments, '--collect-only'])
                self.assertEqual(docker.call_args_list[1].args[0], ['docker', *self.arguments])
                self.assertTrue(self.log.with_name('unit-collection.log').is_file())

    # no tests, an unreadable summary or duplicate summaries cannot establish a complete collection
    def test_empty_or_ambiguous_collection_is_rejected(self):
        for summary in ('no tests collected', '0 tests collected in 0.1s', '',
                        '2 tests collected in 0.1s\n3 tests collected in 0.1s'):
            with self.subTest(summary = summary):
                with patch.object(subprocess, 'run', return_value = self.result(summary)) as docker:
                    with self.assertRaises(RuntimeError):
                        self.runner.suite(self.arguments, self.log)
                self.assertEqual(docker.call_count, 1)

    # missing dependencies or collection-time errors must stop before the suite can run
    def test_collection_process_failure_is_rejected(self):
        with patch.object(subprocess, 'run', return_value = self.result('305 tests collected in 0.1s', 2)):
            with self.assertRaises(RuntimeError):
                self.runner.suite(self.arguments, self.log)

    # a passing summary never overrides the subprocess failure status
    def test_execution_process_failure_is_rejected(self):
        with patch.object(subprocess, 'run', side_effect = [
            self.result('305 tests collected in 0.1s'), self.result('305 passed in 0.1s', 1),
        ]): # patch
            with self.assertRaises(RuntimeError):
                self.runner.suite(self.arguments, self.log)

    # either fewer or more reported passes means the executed suite differs from the collected suite
    def test_execution_count_must_match_collection(self):
        for passed in (1, 304, 306):
            with self.subTest(passed = passed):
                with patch.object(subprocess, 'run', side_effect = [
                    self.result('305 tests collected in 0.1s'), self.result(f'{passed} passed in 0.1s'),
                ]): # patch
                    with self.assertRaises(RuntimeError):
                        self.runner.suite(self.arguments, self.log)

    # skipped modules and selection filters must not silently reduce the suite at collection time
    def test_incomplete_collection_is_rejected(self):
        for label in ('skipped', 'deselected'):
            with self.subTest(label = label):
                with patch.object(subprocess, 'run', return_value = self.result(
                    f'305 tests collected / 1 {label} in 0.1s',
                )) as docker: # patch
                    with self.assertRaises(RuntimeError):
                        self.runner.suite(self.arguments, self.log)
                self.assertEqual(docker.call_count, 1)

    # runtime skips and deselection are rejected even when a summary also contains enough passes
    def test_incomplete_execution_is_rejected(self):
        for label in ('skipped', 'deselected'):
            with self.subTest(label = label):
                with patch.object(subprocess, 'run', side_effect = [
                    self.result('305 tests collected in 0.1s'), self.result(f'305 passed, 1 {label} in 0.1s'),
                ]): # patch
                    with self.assertRaises(RuntimeError):
                        self.runner.suite(self.arguments, self.log)

    # every normal recovery CI run collects this regression file too
    def test_collection_checks_are_in_standard_ci_selection(self):
        self.assertIn('tests/test_ingest_recovery_counts.py', self.runner.UNIT_TESTS)

if __name__ == '__main__':
    unittest.main()
