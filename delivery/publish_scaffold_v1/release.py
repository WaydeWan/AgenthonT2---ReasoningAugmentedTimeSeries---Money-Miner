"""Offline, fail-closed release staging. Does not push, log in, or read secrets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

HERE = Path(__file__).resolve().parent
REPOSITORY = 'WaydeWan/AgenthonT2---ReasoningAugmentedTimeSeries---Money-Miner'
IMAGE = 'ghcr.io/' + REPOSITORY.lower()
HEX = re.compile(r'[a-f0-9]{64}\Z')
ID = re.compile(r'[a-z][a-z0-9-]{0,39}\Z')
MODULE = re.compile(r'submission_candidates\.[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\Z')
SECRET = re.compile(rb'(?:nvapi-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
BLOCKED_PARTS = {'tests', '__pycache__', 'experiments', 'evidence', 'data', 'history', 'training', 'results'}
LICENSE_PATHS = {'LICENSE', 'agenthon-t2-house/v8_bundle/source/LICENSE'}


class ReleaseError(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise ReleaseError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def ref_bytes(root, ref):
    need(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'file_reference_invalid')
    name = ref['path']
    need(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_/.-]+', name), 'file_path_invalid')
    parts = PurePosixPath(name).parts
    need(parts and PurePosixPath(name).as_posix() == name and not PurePosixPath(name).is_absolute() and all(p not in ('.', '..') and not p.startswith('.') for p in parts), 'file_path_invalid')
    need(isinstance(ref['sha256'], str) and HEX.fullmatch(ref['sha256']), 'file_hash_invalid')
    path = root / name
    need(path.resolve().is_relative_to(root.resolve()), 'file_path_escape')
    need(not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)), 'symlink_refused')
    need(path.is_file() and path.stat().st_size <= 32 * 1024 * 1024, 'file_missing_or_too_large')
    data = path.read_bytes()
    need(sha(data) == ref['sha256'], 'source_hash_mismatch')
    need(not SECRET.search(data), 'credential_pattern_refused')
    return data


def validate(root, manifest, official_lock=None):
    need(isinstance(manifest, dict) and set(manifest) == {'schema_version', 'repository', 'release_id', 'files', 'candidates'}, 'manifest_keys_invalid')
    need(manifest['schema_version'] == 'money-miner-public-release-v1' and manifest['repository'] == REPOSITORY, 'manifest_identity_invalid')
    need(isinstance(manifest['release_id'], str) and ID.fullmatch(manifest['release_id']), 'release_id_unfrozen')
    refs = manifest['files']
    need(isinstance(refs, list) and refs, 'source_allowlist_empty')
    files = {}
    for ref in refs:
        data = ref_bytes(root, ref)
        name = ref['path']
        parts = PurePosixPath(name).parts
        allowed = name in LICENSE_PATHS or (name.startswith('submission_candidates/') and not BLOCKED_PARTS.intersection(parts)
            and (name.endswith('.py') or (name.startswith(('submission_candidates/release_configs/', 'submission_candidates/release_artifacts/')) and name.endswith('.json')) or name == 'submission_candidates/release_artifacts/ARTIFACT_PROVENANCE.md'))
        need(allowed, 'source_path_not_allowed')
        need(name not in files, 'duplicate_source')
        files[name] = data
    need(len(LICENSE_PATHS.intersection(files)) == 1, 'exactly_one_team_license_required')
    lock = official_lock if official_lock is not None else json.loads((HERE / 'official-sources.lock.json').read_text(encoding='utf-8'))
    need(lock['toolkit_version'] == '2.6.0' and lock['track_commit'] == '60509df4ad0756443f4af8dc9500aed99a995694', 'official_pin_invalid')
    for ref in lock['files']:
        need(ref['path'].startswith(('vendor/agenthon-toolkit-v2.6.0/common/', 'vendor/track2-60509df/')), 'official_path_invalid')
        need(ref['path'] not in files, 'duplicate_official_source')
        files[ref['path']] = ref_bytes(root, ref)
    candidates = manifest['candidates']
    need(isinstance(candidates, list) and len(candidates) == 3, 'exactly_three_frozen_candidates_required')
    ids, config_hashes = set(), set()
    matrix = []
    for candidate in candidates:
        need(isinstance(candidate, dict) and set(candidate) == {'id', 'config'}, 'candidate_keys_invalid')
        cid, name = candidate['id'], candidate['config']
        need(isinstance(cid, str) and ID.fullmatch(cid) and cid not in ids, 'candidate_id_invalid')
        need(isinstance(name, str) and name.startswith('submission_candidates/release_configs/') and name in files, 'config_not_allowlisted')
        try:
            config = json.loads(files[name])
        except (UnicodeError, ValueError):
            raise ReleaseError('config_json_invalid') from None
        need(isinstance(config, dict) and set(config) == {'module', 'environment'}, 'config_keys_invalid')
        need(isinstance(config['module'], str) and MODULE.fullmatch(config['module']), 'entry_module_invalid')
        need(config['module'].replace('.', '/') + '.py' in files, 'entry_module_not_allowlisted')
        env = config['environment']
        need(isinstance(env, dict) and len(env) <= 16, 'config_environment_invalid')
        for key, value in env.items():
            need(re.fullmatch(r'MONEY_MINER_[A-Z0-9_]+', key) and not re.search('KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|ENDPOINT|BASE_URL', key), 'config_environment_key_refused')
            need(isinstance(value, str) and len(value) <= 256 and re.fullmatch(r'[A-Za-z0-9_./:@,+-]*', value), 'config_environment_value_refused')
            if value.startswith('/app/'):
                need(value[5:] in files, 'config_artifact_not_allowlisted')
        digest = sha(canonical(config))
        need(digest not in config_hashes, 'duplicate_candidate_configuration')
        ids.add(cid)
        config_hashes.add(digest)
        matrix.append({'id': cid, 'tag': f"{IMAGE}:{manifest['release_id']}-{cid}-{digest[:12]}", 'config_sha256': digest})
    return files, {'include': matrix}


def stage(root, manifest, candidate_id, out, official_lock=None):
    files, matrix = validate(root, manifest, official_lock)
    row = next((x for x in matrix['include'] if x['id'] == candidate_id), None)
    need(row is not None, 'candidate_unknown')
    need(not out.exists(), 'output_must_not_exist')
    config_name = next(x['config'] for x in manifest['candidates'] if x['id'] == candidate_id)
    # The build context is newly created, never the private repository root.
    out.mkdir(parents=True)
    for name, data in files.items():
        # Other candidate configs are not needed inside this image.
        if name.startswith('submission_candidates/release_configs/'):
            continue
        target = out / ('LICENSE' if name in LICENSE_PATHS else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (out / 'release-config.json').write_bytes(canonical(json.loads(files[config_name])) + b'\n')
    (out / 'release-identity.json').write_bytes(canonical({'id': candidate_id, 'config_sha256': row['config_sha256'], 'manifest_sha256': sha(canonical(manifest))}) + b'\n')
    for name in ('Dockerfile', 'forecast'):
        shutil.copyfile(HERE / name, out / name)
    return row


def record(root, manifest, cid, digest):
    _, matrix = validate(root, manifest)
    need(isinstance(digest, str) and digest.startswith('sha256:') and HEX.fullmatch(digest[7:]), 'build_digest_invalid')
    row = next((x for x in matrix['include'] if x['id'] == cid), None)
    need(row is not None, 'candidate_unknown')
    return {'schema_version': 'money-miner-build-record-v1', **row,
            'digest': digest, 'image_reference': IMAGE + '@' + digest,
            'platform': 'linux/amd64', 'manifest_sha256': sha(canonical(manifest)),
            'official_lock_sha256': sha((HERE / 'official-sources.lock.json').read_bytes()),
            'dockerfile_sha256': sha((HERE / 'Dockerfile').read_bytes()),
            'source_commit': os.environ.get('GITHUB_SHA'), 'github_run_id': os.environ.get('GITHUB_RUN_ID'),
            'github_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
            'anonymous_pull_verified': False, 'official_smoke_verified': False}


def export(root, manifest, out):
    """Prepare a new public checkout directory; never initializes/pushes Git."""
    files, _ = validate(root, manifest)
    need(not out.exists(), 'output_must_not_exist')
    out.mkdir(parents=True)
    for name, data in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    names = ['release.py', 'Dockerfile', 'forecast', 'official-sources.lock.json',
             'verify_anonymous_pull.py', 'tests/test_scaffold.py', 'release-manifest.template.json',
             'README.md', 'audit.json', 'publish.yml']
    for name in names:
        target = out / 'delivery/publish_scaffold_v1' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
    workflow = out / '.github/workflows/publish-candidates.yml'
    workflow.parent.mkdir(parents=True)
    shutil.copyfile(HERE / 'publish.yml', workflow)
    (out / 'release-manifest.json').write_bytes(canonical(manifest) + b'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['matrix', 'stage', 'record', 'export'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--candidate')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--digest')
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        if args.command == 'matrix':
            _, result = validate(args.root.resolve(), manifest)
            if args.github_output:
                with args.github_output.open('a', encoding='utf-8') as stream:
                    stream.write('matrix=' + canonical(result).decode() + '\n')
            else:
                print(canonical(result).decode())
        elif args.command == 'stage':
            need(args.out is not None, 'output_required')
            stage(args.root.resolve(), manifest, args.candidate, args.out)
        elif args.command == 'export':
            need(args.out is not None, 'output_required')
            export(args.root.resolve(), manifest, args.out)
        else:
            need(args.out is not None and not args.out.exists(), 'new_output_required')
            args.out.write_bytes(canonical(record(args.root.resolve(), manifest, args.candidate, args.digest)) + b'\n')
    except (ReleaseError, OSError, ValueError, KeyError, TypeError) as exc:
        # Never print source content, environment values or subprocess output.
        print('Release validation failed: ' + (str(exc) if isinstance(exc, ReleaseError) else type(exc).__name__))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
