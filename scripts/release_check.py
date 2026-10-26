"""CPU-only Release Candidate gate: do not interpret as GPU correctness."""
from pathlib import Path
import json
import subprocess
import sys
import re

ROOT = Path(__file__).resolve().parents[1]


def run(*command):
    print('+', ' '.join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main():
    required = [
        'README.md', 'Makefile', '.gitignore',
        'triton/matmul.py', 'triton/paged_kv_write.py',
        'cuda/memory/matrix_transpose.cu', 'cuda/memory/transpose_kernels.cuh',
        'cuda/optimization/double_buffer.cu', 'cuda/optimization/double_buffer_kernels.cuh',
        'docs/cases/matmul_spill.md', 'docs/cases/paged_kv_write.md',
        'docs/cases/cuda_transpose.md', 'docs/cases/cuda_double_buffer.md',
        'docs/release/RELEASE_REVIEW.md', 'docs/release/GPU_ACCEPTANCE.md',
        'docs/release/RELEASE_STATUS.json', 'docs/environment.md',
        'results/historical/week13_matmul.json', 'results/historical/other_cases.json',
        'results/historical/week12_double_buffer.json',
    ]
    missing = [x for x in required if not (ROOT / x).is_file()]
    if missing:
        print('[FAIL] Missing:', ', '.join(missing))
        return 1

    status = json.loads((ROOT/'docs/release/RELEASE_STATUS.json').read_text(encoding='utf-8'))
    if status.get('gpu_validated') is not False or status.get('release_class') != 'offline_candidate':
        print('[FAIL] Release labels are inconsistent with offline-only gate')
        return 1

    # Minimal local Markdown-link existence check; ignore HTTP(S)/fragments.
    broken = []
    markdown = [ROOT/'README.md', *sorted((ROOT/'docs').rglob('*.md'))]
    for path in markdown:
        content = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
        for ref in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', content):
            location = ref.split('#', 1)[0]
            if not location or '://' in location or location.startswith('mailto:'):
                continue
            if not (path.parent / location).exists():
                broken.append(f'{path.relative_to(ROOT)} -> {ref}')
    if broken:
        print('[FAIL] Broken local Markdown links:', *broken, sep='\n')
        return 1

    if run(sys.executable, 'scripts/offline_audit.py') != 0:
        print('[FAIL] Structural audit')
        return 1
    if run(sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*mapping_cpu.py', '-v') != 0:
        print('[FAIL] CPU reference tests')
        return 1
    print('[PASS] OFFLINE RELEASE GATE: structure + historical provenance + CPU mapping/reference tests')
    print('[PENDING] GPU compilation, GPU correctness, performance reproduction, formal release')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
