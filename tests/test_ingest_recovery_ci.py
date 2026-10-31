"""A CI result must represent complete recovery suites, propagate faults and clean disposable resources."""

import importlib.util
import subprocess

from pathlib import Path
from types import SimpleNamespace

import pytest

# load the repository-owned runner without starting Docker during test collection
@pytest.fixture
def runner(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('recovery_runner', Path(__file__).with_name('run_ingest_recovery.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('GRAPHITI_RECOVERY_IMAGE', 'recovery-test-image')
    monkeypatch.setenv('GRAPHITI_RECOVERY_LOG_DIR', str(tmp_path))
    return module

# model Docker's process boundary while retaining the real runner's orchestration and result validation
@pytest.mark.parametrize('failure', [None, 'unit-process', 'unit-count', 'unit-skips', 'graph-process', 'graph-count'])
def test_ci_runner_never_accepts_a_partial_suite(runner, monkeypatch, tmp_path, failure):

    calls = []
    def docker(arguments, **options):
        calls.append(arguments)
        output = ''
        result = 0
        if 'pytest' in arguments:
            unit = 'tests/test_ingest_recovery_ci.py' in arguments
            output = '230 passed in 1.0s' if unit else '3 passed in 1.0s'
            if failure == ('unit-process' if unit else 'graph-process'):
                result = 1
            if failure == ('unit-count' if unit else 'graph-count'):
                output = '1 passed in 1.0s'
            if unit and failure == 'unit-skips':
                output = '230 passed, 1 skipped in 1.0s'
        return SimpleNamespace(returncode = result, stdout = output, stderr = '')
    monkeypatch.setattr(subprocess, 'run', docker)
    if failure is None:
        runner.main()
        assert (tmp_path/'unit.log').read_text() == '230 passed in 1.0s'
        assert (tmp_path/'neo4j.log').read_text() == '3 passed in 1.0s'
    else:
        with pytest.raises(RuntimeError):
            runner.main()
    tests = [call for call in calls if 'pytest' in call]
    assert len(tests) == (1 if failure and failure.startswith('unit-') else 2)
    assert all('readonly' in ' '.join(call) and '--read-only' in call for call in tests)
    if not failure or failure.startswith('graph-'):
        assert any(call[1:4] == ['network', 'create', '--internal'] for call in calls)
        assert any(call[1:3] == ['rm', '-f'] for call in calls)
        assert any(call[1:3] == ['network', 'rm'] for call in calls)
    else:
        assert not any(call[1:3] == ['network', 'create'] for call in calls)
