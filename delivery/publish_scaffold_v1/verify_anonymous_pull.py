"""Later verification only: pull an actual GHCR digest with empty client auth.

No login, token lookup, image execution, registry visibility change or upload.
Use a local trusted Docker daemon; authenticated registry mirrors are outside
this client-level evidence. Success is not an official forecast smoke result.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

IMAGE = 'ghcr.io/waydewan/agenthont2---reasoningaugmentedtimeseries---money-miner'
REFERENCE = re.compile(re.escape(IMAGE) + r'@sha256:[a-f0-9]{64}\Z')


def verify(reference, scratch, *, runner=subprocess.run, timeout=900):
    if not isinstance(reference, str) or not REFERENCE.fullmatch(reference):
        raise ValueError('immutable_target_repository_reference_required')
    result = {'schema_version': 'money-miner-anonymous-pull-v1', 'image_reference': reference,
        'platform': 'linux/amd64', 'anonymous_pull_passed': False, 'credentials_used': False,
        'verified_digest': None, 'interface_label': None,
        'verification_method': 'empty_docker_config_pull',
        'verified_at': datetime.now(timezone.utc).isoformat(), 'reason': None}
    scratch = Path(scratch).resolve()
    # Minimal process environment; no GH/NV/team credentials or Docker auth env.
    permitted = {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'TMPDIR', 'COMSPEC', 'PATHEXT'}
    env = {k: v for k, v in os.environ.items() if k.upper() in permitted}
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pull-', dir=scratch) as folder:
            clean = Path(folder)
            (clean / 'config.json').write_text('{"auths":{}}\n', encoding='utf-8')
            env.update({'DOCKER_CONFIG': str(clean), 'HOME': str(clean), 'USERPROFILE': str(clean)})
            command = ['docker', '--config', str(clean)]
            pull = runner(command + ['pull', '--platform', 'linux/amd64', reference], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False)
            if pull.returncode != 0:
                result['reason'] = 'anonymous_pull_failed'
                return result
            inspected = runner(command + ['image', 'inspect', reference], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               timeout=min(timeout, 60), check=False)
            if inspected.returncode != 0:
                result['reason'] = 'local_inspection_failed'
                return result
            try:
                values = json.loads(inspected.stdout)
                if not isinstance(values, list) or len(values) != 1:
                    raise ValueError()
                item = values[0]
                labels = (item.get('Config') or {}).get('Labels') or {}
                if item.get('Os') != 'linux' or item.get('Architecture') != 'amd64':
                    result['reason'] = 'platform_mismatch'
                elif reference not in (item.get('RepoDigests') or []):
                    result['reason'] = 'digest_mismatch'
                elif labels.get('qfbench2.interface_version') != '2.0':
                    result['reason'] = 'interface_label_mismatch'
                else:
                    result.update(anonymous_pull_passed=True, verified_digest=reference.split('@', 1)[1], interface_label='2.0')
            except (ValueError, TypeError, AttributeError):
                result['reason'] = 'inspection_json_invalid'
    except subprocess.TimeoutExpired:
        result['reason'] = 'docker_timeout'
    except OSError:
        result['reason'] = 'docker_or_scratch_unavailable'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scratch-root', type=Path, default=Path.cwd() / '.anonymous-pull-work')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new; preserve prior evidence')
    try:
        result = verify(args.image, args.scratch_root)
    except ValueError:
        parser.error('supply an actual immutable digest from this release repository')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=True, sort_keys=True, indent=2)
        stream.write('\n')
    print('Anonymous image verification: ' + ('passed' if result['anonymous_pull_passed'] else 'failed'))
    return 0 if result['anonymous_pull_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
