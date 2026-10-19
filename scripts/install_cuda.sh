#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
exec "${PYTHON:-python}" - "${WHEEL_DIR:-outputs/wheels}" <<'PY'
import hashlib
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

if not shutil.which('curl'):
    raise SystemExit('curl is required for official wheel downloads (conda install -c conda-forge curl).')

if not (sys.version_info[:2] == (3, 11) and platform.system() == 'Linux'
        and platform.machine() == 'x86_64' and torch.__version__.split('+')[0] == '2.8.0'
        and torch.version.cuda and torch.version.cuda.startswith('12.')
        and torch._C._GLIBCXX_USE_CXX11_ABI):
    raise SystemExit('These wheels require Linux x86_64, Python 3.11, and PyTorch 2.8.0/CUDA 12/C++11 ABI. '
                     'For another build, use pip install -r requirements-cuda.txt --no-build-isolation.')

assets = [
    ('Dao-AILab/causal-conv1d', 522094245, 196371690,
     'causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl',
     'b28c3657a9d6855e1f95bc912ab94f893434276b437dec8008c92c9f38b6a1f4'),
    ('Dao-AILab/flash-attention', 282716558, 256043372,
     'flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl',
     '3d41b2fc55753faa7f45d6568ea73a96b96afb48b82994ab9b49bcbcb6c87588'),
]
directory = Path(sys.argv[1]).resolve()
directory.mkdir(parents=True, exist_ok=True)

def valid(path, expected):
    if not path.is_file():
        return False
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() == expected

wheels = []
for repository, asset_id, size, filename, digest in assets:
    wheel = directory / filename
    if not valid(wheel, digest):
        api_url = f'https://api.github.com/repos/{repository}/releases/assets/{asset_id}?download=1&fresh={time.time_ns()}'
        partial = wheel.with_suffix('.part')
        if valid(partial, digest):
            partial.replace(wheel)
            wheels.append(str(wheel))
            continue
        print(f'Downloading official wheel: {filename}', flush=True)
        response = subprocess.run(['curl', '-q', '-4', '--silent', '--show-error', '--location', '--fail',
            '--connect-timeout', '10', '--max-time', '60', '--retry', '2', '--range', '0-0',
            '--header', 'Accept: application/octet-stream', '--output', '/dev/null',
            '--write-out', '%{url_effective}', api_url], capture_output=True, text=True)
        if response.returncode:
            raise SystemExit(f'Could not reach the official release: {filename}. Rerun to retry.')
        download_url = response.stdout.strip()
        # Small HTTP ranges keep interrupted downloads resumable.
        chunk_size = 8 * 1024 * 1024
        start = partial.stat().st_size // chunk_size * chunk_size if partial.exists() else 0
        if start >= size:
            start = 0
        with partial.open('r+b' if partial.exists() else 'wb') as output:
            output.truncate(start)
            output.seek(start)
            for offset in range(start, size, chunk_size):
                end = min(offset + chunk_size, size) - 1
                chunk_path = partial.with_suffix(f'.chunk-{offset}')
                expected = end - offset + 1
                for attempt in range(6):
                    received = chunk_path.stat().st_size if chunk_path.exists() else 0
                    if received == expected:
                        break
                    if received > expected:
                        chunk_path.unlink()
                        received = 0
                    transfer = partial.with_suffix('.download')
                    response = subprocess.run(['curl', '-q', '-4', '--silent', '--show-error', '--location',
                        '--fail', '--connect-timeout', '10', '--max-time', '90', '--range',
                        f'{offset + received}-{end}', '--output', str(transfer), '--write-out',
                        '%{http_code}', download_url], capture_output=True)
                    if (response.stdout.strip() == b'206' and transfer.exists()
                            and 0 < transfer.stat().st_size <= expected - received):
                        with chunk_path.open('ab') as chunk_output:
                            chunk_output.write(transfer.read_bytes())
                    transfer.unlink(missing_ok=True)
                    if chunk_path.exists() and chunk_path.stat().st_size == expected:
                        break
                    if attempt == 5:
                        raise SystemExit(f'Download interrupted: {filename}. Rerun to resume.')
                output.write(chunk_path.read_bytes())
                chunk_path.unlink()
        if not valid(partial, digest):
            partial.unlink(missing_ok=True)
            raise SystemExit(f'SHA-256 mismatch: {filename}')
        partial.replace(wheel)
    print(f'SHA-256 verified: {filename}', flush=True)
    wheels.append(str(wheel))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', *wheels], check=True)
PY
