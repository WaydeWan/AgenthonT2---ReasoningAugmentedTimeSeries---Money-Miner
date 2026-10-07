"""Offline tests use synthetic source/configs and mocked Docker, never credentials."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load('release')
pull = load('verify_anonymous_pull')


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='scaffold-test-', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {'schema_version': 'money-miner-public-release-v1', 'repository': release.REPOSITORY,
                         'release_id': 'synthetic-test', 'files': [], 'candidates': []}
        self.lock = {'toolkit_version': '2.6.0', 'track_commit': '60509df4ad0756443f4af8dc9500aed99a995694', 'files': []}
        self.add('LICENSE', b'Synthetic test license')
        self.add('submission_candidates/entry.py', b'print("synthetic")\n')
        for number in range(3):
            name = f'submission_candidates/release_configs/config{number}.json'
            self.add(name, json.dumps({'module': 'submission_candidates.entry', 'environment': {'MONEY_MINER_TEST': str(number)}}).encode())
            self.manifest['candidates'].append({'id': f'choice-{number}', 'config': name})

    def add(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.manifest['files'].append({'path': name, 'sha256': hashlib.sha256(data).hexdigest()})

    def test_stage_excludes_unlisted_private_files_and_freezes_config(self):
        (self.root / 'private-history.csv').write_text('not part of public context')
        out = self.root / 'context'
        row = release.stage(self.root, self.manifest, 'choice-1', out, self.lock)
        self.assertFalse((out / 'private-history.csv').exists())
        self.assertFalse((out / 'submission_candidates/release_configs').exists())
        self.assertEqual(json.loads((out / 'release-config.json').read_text())['environment'], {'MONEY_MINER_TEST': '1'})
        self.assertEqual(json.loads((out / 'release-identity.json').read_text())['config_sha256'], row['config_sha256'])
        self.assertIn(':synthetic-test-choice-1-', row['tag'])

    def test_hash_change_and_unfrozen_config_fail(self):
        (self.root / 'submission_candidates/entry.py').write_bytes(b'changed')
        with self.assertRaisesRegex(release.ReleaseError, 'source_hash_mismatch'):
            release.validate(self.root, self.manifest, self.lock)
        draft = json.loads((HERE / 'release-manifest.template.json').read_text())
        with self.assertRaisesRegex(release.ReleaseError, 'release_id_unfrozen'):
            release.validate(self.root, draft, self.lock)

    def test_traversal_and_history_and_duplicate_configs_refused(self):
        with self.assertRaisesRegex(release.ReleaseError, 'file_path_invalid'):
            release.ref_bytes(self.root, {'path': '../escape.py', 'sha256': '0' * 64})
        self.add('submission_candidates/history/training.py', b'not allowed')
        with self.assertRaisesRegex(release.ReleaseError, 'source_path_not_allowed'):
            release.validate(self.root, self.manifest, self.lock)
        self.manifest['files'].pop()
        self.manifest['candidates'][1]['config'] = self.manifest['candidates'][0]['config']
        with self.assertRaisesRegex(release.ReleaseError, 'duplicate_candidate_configuration'):
            release.validate(self.root, self.manifest, self.lock)

    def test_official_lock_is_real_and_draft_workflow_pinned(self):
        lock = json.loads((HERE / 'official-sources.lock.json').read_text())
        self.assertGreater(len(lock['files']), 20)
        for ref in lock['files']:
            self.assertRegex(ref['sha256'], r'^[a-f0-9]{64}$')
            self.assertNotIn('__pycache__', ref['path'])
        workflow = (HERE / 'publish.yml').read_text()
        actions = re.findall(r'uses:\s+(\S+)', workflow)
        self.assertTrue(actions)
        self.assertTrue(all(re.fullmatch(r'(actions|docker)/[a-z-]+@[a-f0-9]{40}', x) for x in actions))
        self.assertIn('packages: write', workflow)
        self.assertIn('persist-credentials: false', workflow)
        self.assertNotIn('pull_request:', workflow)

    def test_credentials_cannot_become_candidate_environment_or_source(self):
        name = self.manifest['candidates'][0]['config']
        data = json.dumps({'module': 'submission_candidates.entry', 'environment': {'MONEY_MINER_API_KEY': 'synthetic'}}).encode()
        (self.root / name).write_bytes(data)
        next(x for x in self.manifest['files'] if x['path'] == name)['sha256'] = hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(release.ReleaseError, 'config_environment_key_refused'):
            release.validate(self.root, self.manifest, self.lock)
        source = b'example=' + b'nvapi-' + b'a' * 32
        (self.root / 'submission_candidates/entry.py').write_bytes(source)
        reference = next(x for x in self.manifest['files'] if x['path'] == 'submission_candidates/entry.py')
        reference['sha256'] = hashlib.sha256(source).hexdigest()
        with self.assertRaisesRegex(release.ReleaseError, 'credential_pattern_refused'):
            release.validate(self.root, self.manifest, self.lock)


class PullTests(unittest.TestCase):
    def test_empty_auth_exact_digest_and_environment_isolation(self):
        ref = pull.IMAGE + '@sha256:' + 'a' * 64
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(json.loads((Path(argv[2]) / 'config.json').read_text()), {'auths': {}})
            self.assertNotIn('GITHUB_TOKEN', kwargs['env'])
            self.assertNotIn('DOCKER_AUTH_CONFIG', kwargs['env'])
            self.assertNotIn('DOCKER_CONTEXT', kwargs['env'])
            self.assertEqual(argv[-1], ref)
            if 'pull' in argv:
                self.assertEqual(argv[-3:-1], ['--platform', 'linux/amd64'])
                return SimpleNamespace(returncode=0)
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'Os': 'linux', 'Architecture': 'amd64', 'RepoDigests': [ref], 'Config': {'Labels': {'qfbench2.interface_version': '2.0'}}}]))

        with tempfile.TemporaryDirectory(dir=HERE) as folder, patch.dict(os.environ, {'GITHUB_TOKEN': 'synthetic-only', 'DOCKER_AUTH_CONFIG': 'synthetic-only', 'DOCKER_CONTEXT': 'synthetic-only'}):
            result = pull.verify(ref, folder, runner=runner)
            self.assertEqual(list(Path(folder).iterdir()), [])
        self.assertTrue(result['anonymous_pull_passed'])
        self.assertFalse(result['credentials_used'])
        self.assertEqual(len(calls), 2)

    def test_wrong_digest_platform_failed_pull_and_timeout_fail_closed(self):
        ref = pull.IMAGE + '@sha256:' + 'b' * 64
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            for arch, digests, expected in [('arm64', [ref], 'platform_mismatch'), ('amd64', [], 'digest_mismatch')]:
                def runner(argv, **kwargs):
                    return SimpleNamespace(returncode=0, stdout=json.dumps([{'Os': 'linux', 'Architecture': arch, 'RepoDigests': digests}]))
                result = pull.verify(ref, folder, runner=runner)
                self.assertFalse(result['anonymous_pull_passed'])
                self.assertEqual(result['reason'], expected)
            self.assertEqual(pull.verify(ref, folder, runner=lambda *a, **kw: SimpleNamespace(returncode=1))['reason'], 'anonymous_pull_failed')
            def timeout(*args, **kwargs):
                raise subprocess.TimeoutExpired('synthetic', 1)
            self.assertEqual(pull.verify(ref, folder, runner=timeout)['reason'], 'docker_timeout')
        with self.assertRaises(ValueError):
            pull.verify(pull.IMAGE + ':mutable', HERE)


if __name__ == '__main__':
    unittest.main()
