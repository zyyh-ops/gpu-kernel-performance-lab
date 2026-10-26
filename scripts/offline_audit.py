"""CPU-only source + historical-metadata check. It does NOT execute a GPU test."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PY = [
    "triton/matmul.py",
    "triton/paged_kv_write.py",
    "tests/test_matmul.py",
    "tests/test_paged_kv.py",
    "tests/test_paged_kv_mapping_cpu.py",
    "benchmarks/bench_matmul.py",
    "benchmarks/bench_paged_kv.py",
    "scripts/inspect_matmul.py",
    "scripts/autotune_matmul.py",
    "scripts/inspect_paged_kv.py",
    "scripts/paged_kv_mapping_model.py",
]
REQUIRED_DOCS = [
    "README.md",
    "docs/inventory.md",
    "docs/benchmark_protocol.md",
    "docs/gpu_validation_queue.md",
    "docs/cases/matmul_spill.md",
    "docs/cases/paged_kv_write.md",
]
PENDING_SOURCES = [
    "triton/softmax.py",
    "triton/vector_add.py",
    "cuda/memory/matrix_transpose.cu",
    "cuda/optimization/double_buffer.cu",
]


def main():
    problems = []
    for name in REQUIRED_PY:
        path = ROOT / name
        if not path.is_file():
            problems.append(f"MISSING required Python file: {name}")
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            print(f"[OK] Python syntax: {name}")
        except (SyntaxError, UnicodeError) as e:
            problems.append(f"SYNTAX failure: {name}: {e}")

    for name in REQUIRED_DOCS:
        if not (ROOT / name).is_file():
            problems.append(f"MISSING documentation: {name}")

    for name in ["week13_matmul.json", "other_cases.json"]:
        file = ROOT / "results" / "historical" / name
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            if record.get("record_type") != "historical_reference":
                problems.append(f"Incorrect provenance field: {name}")
            if "historical" not in record.get("validation_status", ""):
                problems.append(f"Incorrect validation status: {name}")
            print(f"[OK] Historical JSON and status: {name}")
        except (OSError, ValueError) as e:
            problems.append(f"Bad historical JSON: {name}: {e}")

    for name in PENDING_SOURCES:
        print(f"[INFO] Source {'available' if (ROOT/name).is_file() else 'not yet migrated'}: {name}")

    if problems:
        for issue in problems:
            print("[FAIL]", issue)
        raise SystemExit(1)
    print("[PASS] OFFLINE structural audit ONLY — GPU correctness/performance UNVERIFIED")


if __name__ == "__main__":
    main()
