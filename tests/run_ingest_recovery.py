"""Run the ingestion recovery suites with immutable dependencies and disposable Neo4j storage."""

import os
import re
import subprocess
import tempfile
import time
import uuid

from pathlib import Path

NEO4J_IMAGE = 'neo4j@sha256:362542416de6c09a971484d1893878016cc3b5cdec166e54b1c824a220ecd6b9'
UNIT_TESTS = [
    'tests/driver/test_neo4j_mention_replay.py', 'tests/driver/test_neo4j_episode_ops.py',
    'tests/utils/maintenance/test_bulk_utils.py', 'tests/llm_client/test_response_recovery.py',
    'tests/llm_client/test_errors.py', 'tests/llm_client/test_openai_client.py',
    'tests/llm_client/test_azure_openai_client.py', 'tests/utils/maintenance/test_edge_batch_recovery.py',
    'tests/utils/maintenance/test_semantic_recovery.py', 'tests/utils/maintenance/test_temporal_recovery.py',
    'tests/utils/maintenance/test_node_operations.py', 'tests/utils/maintenance/test_edge_operations.py',
    'tests/utils/maintenance/test_graph_preparation.py', 'tests/utils/maintenance/test_bounded_edge_resolution.py',
    'tests/utils/maintenance/test_financial_observation_resolution.py', 'tests/test_ingest_recovery_ci.py',
] # UNIT_TESTS
INTEGRATION_TESTS = ['tests/test_neo4j_ingest_recovery_int.py', 'tests/test_neo4j_episode_replay_int.py']

# retain command output and propagate failures instead of allowing a partial green CI result
def command(arguments, *, log = None, timeout = 120):
    result = subprocess.run(['docker', *arguments], capture_output = True, text = True, timeout = timeout)
    output = result.stdout + result.stderr
    if log is not None:
        log.write_text(output)
    if result.returncode:
        raise RuntimeError(f"Docker command failed: {arguments}; output: {output[-6000:]}")
    return output

# require the full selected suite without skips, missing collection or an incomplete integration run
def suite(arguments, expected, log):
    output = command(arguments, log = log)
    counts = re.findall(r"\b(\d+) passed\b", output)
    if not counts or int(counts[-1]) != expected or re.search(r"\b\d+ skipped\b", output):
        raise RuntimeError(f"Expected {expected} passing tests without skips; log: {log}")
    print(f"PASSED: {log.stem}: {expected} tests", flush = True)

# exercise the exact fork checkout; no production endpoint, source write or additional graph backend is used
def main():
    repo = Path(__file__).resolve().parents[1]
    image = os.environ['GRAPHITI_RECOVERY_IMAGE']
    logs = Path(os.environ.get('GRAPHITI_RECOVERY_LOG_DIR') or tempfile.mkdtemp(
        prefix = 'graphiti-ingest-recovery-', dir = '/tmp',
    )) # logs
    logs.mkdir(parents = True, exist_ok = True)
    common = [
        'run', '--rm', '--read-only', '--tmpfs', '/tmp',
        '--mount', f'type=bind,src={repo},dst=/candidate,readonly',
        '-e', 'PYTHONPATH=/candidate:/app', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1', '-e', 'GRAPHITI_TELEMETRY_ENABLED=false',
        '-e', 'DISABLE_FALKORDB=1', '--workdir', '/candidate',
    ] # common
    pytest = [
        '--entrypoint', '/app/venv/bin/python', image, '-B', '-m', 'pytest', '-c', '/dev/null',
        '-p', 'no:cacheprovider', '-p', 'pytest_asyncio.plugin',
        '-o', 'asyncio_default_fixture_loop_scope=function', '-o', 'markers=integration: database checks',
    ] # pytest
    network = f'graphiti-recovery-{uuid.uuid4().hex[:12]}'
    server = f'{network}-neo4j'
    created = False
    try:
        suite([*common, '--network', 'none', *pytest, *UNIT_TESTS, '-q'], 230, logs/'unit.log')
        command(['network', 'create', '--internal', network])
        created = True
        command([
            'run', '--rm', '--pull=missing', '-d', '--name', server, '--network', network,
            '--network-alias', 'graphdb', '--tmpfs', '/data', '--tmpfs', '/logs', '-e', 'NEO4J_AUTH=none',
            '-e', 'NEO4J_server_memory_heap_initial__size=256m', '-e', 'NEO4J_server_memory_heap_max__size=512m',
            '-e', 'NEO4J_server_memory_pagecache_size=256m', NEO4J_IMAGE,
        ]) # command
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            ready = subprocess.run(['docker', 'exec', server, 'cypher-shell', 'RETURN 1'],
                                   capture_output = True, timeout = 10)
            if ready.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('Disposable Neo4j did not become ready')
        suite([
            *common, '--network', network, '-e', 'NEO4J_RECOVERY_TEST_URI=bolt://graphdb:7687',
            *pytest, *INTEGRATION_TESTS, '-q',
        ], 3, logs/'neo4j.log') # suite
        print(f'PASSED: 233 Neo4j ingestion recovery tests; logs: {logs}', flush = True)
    finally:
        if created:
            server_logs = subprocess.run(['docker', 'logs', server], capture_output = True, text = True, timeout = 20)
            (logs/'neo4j-server.log').write_text(server_logs.stdout + server_logs.stderr)
            subprocess.run(['docker', 'rm', '-f', server], capture_output = True, timeout = 30)
            subprocess.run(['docker', 'network', 'rm', network], capture_output = True, timeout = 30)

if __name__ == '__main__':
    main()



