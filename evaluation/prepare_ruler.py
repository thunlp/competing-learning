#!/usr/bin/env python
"""Generate the paper's RULER task/length grid for a selected tokenizer.

Prepare the generator and source corpora with ``python -m evaluation.prepare_data
--benchmark ruler`` first. Raw RULER records are written to
``<out-root>/<tokenizer-name>/<length>/<task>/test.jsonl``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import json
import subprocess
import sys
import time
from pathlib import Path

from .ruler import ruler_v1_skip_reason

REPO_ROOT = Path(__file__).resolve().parent.parent
RULER_SRC = Path(os.environ.get("RULER_SOURCE", "third_party/RULER")).resolve()
RULER_PREPARE = RULER_SRC / "scripts" / "data" / "prepare.py"
RULER_JSON_DIR = RULER_SRC / "scripts" / "data" / "synthetic" / "json"

ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

ESSAY_TASKS = {
    "niah_single_2", "niah_single_3", "niah_multikey_1",
    "niah_multivalue", "niah_multiquery",
}
QA_FILES = {"qa_1": "squad.json", "qa_2": "hotpotqa.json"}

def task_length_skip_reason(task: str, length: int) -> str | None:
    return ruler_v1_skip_reason(task, length)


SUBPROCESS_TIMEOUT_SEC = 600


def preflight(tasks):
    missing = [name for name in ('wonderwords', 'tenacity', 'nltk', 'yaml', 'scipy', 'transformers')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Missing data-generation dependencies: {missing}. '
                           'Install requirements-eval.txt before preparing RULER.')
    import nltk
    nltk_dir = Path(os.environ.get('NLTK_DATA', REPO_ROOT / 'datasets/nltk_data')).resolve()
    nltk.data.path.insert(0, str(nltk_dir))
    os.environ.setdefault('NLTK_DATA', str(nltk_dir))
    for resource in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError as error:
            raise RuntimeError('Missing NLTK tokenizer data. Run '
                               'python -m evaluation.prepare_data --benchmark ruler first.') from error
    if not RULER_PREPARE.exists():
        raise FileNotFoundError(f"RULER prepare.py not found: {RULER_PREPARE}. Run "
                                "python -m evaluation.prepare_data --benchmark ruler first.")
    if any(t in ESSAY_TASKS for t in tasks):
        f = RULER_JSON_DIR / "PaulGrahamEssays.json"
        if not f.exists():
            raise FileNotFoundError(
                f"PaulGrahamEssays.json missing at {f}; run "
                f"`python download_paulgraham_essay.py` in that dir first."
            )
    for qa, fname in QA_FILES.items():
        if qa in tasks and not (RULER_JSON_DIR / fname).exists():
            raise FileNotFoundError(
                f"{fname} missing at {RULER_JSON_DIR}; run `bash download_qa_dataset.sh` "
                f"in that dir first."
            )
    if "cwe" in tasks and not (RULER_JSON_DIR / "english_words.json").exists():
        raise FileNotFoundError(f"english_words.json missing at {RULER_JSON_DIR}")


def _validate_jsonl(path: Path) -> tuple[int, str | None]:
    """Return (line_count, error_msg_or_None)."""
    if not path.exists():
        return 0, "missing"
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                return n, f"bad json line {n}: {e}"
            for k in ("index", "input", "outputs", "length", "answer_prefix"):
                if k not in obj:
                    return n, f"missing key {k!r} on line {n}"
            n += 1
    if n == 0:
        return 0, "empty"
    return n, None


def generate(
    tokenizer_name: str,
    tokenizer_path: str,
    lengths,
    tasks,
    num_samples: int,
    out_root: Path,
    overwrite: bool,
    expected_min_samples: int | None = None,
    timeout_sec: int = SUBPROCESS_TIMEOUT_SEC,
    force: bool = False,
):
    out_root = Path(out_root).resolve()
    if Path(tokenizer_path).exists():
        tokenizer_path = str(Path(tokenizer_path).resolve())
    subprocess_env = dict(os.environ)
    subprocess_env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    required_tasks = [task for task in tasks if force or any(
        task_length_skip_reason(task, length) is None for length in lengths)]
    preflight(required_tasks)
    if expected_min_samples is None:
        expected_min_samples = num_samples
    failures = []
    successes = []
    skipped = []
    for length in lengths:
        for task in tasks:
            reason = None if force else task_length_skip_reason(task, length)
            if reason is not None:
                print(f"[skip] L={length:>6} task={task:<18} ({reason})", flush=True)
                skipped.append((length, task, reason))
                continue

            out_dir = out_root / tokenizer_name / str(length)
            target = out_dir / task / "test.jsonl"
            if target.exists() and not overwrite:
                n, err = _validate_jsonl(target)
                if err is None and n >= expected_min_samples:
                    print(f"[cached] L={length:>6} task={task:<18} ({n} samples already)", flush=True)
                    successes.append((length, task, n, True))
                    continue
                print(f"[regen] L={length} task={task} ({err}); regenerating", flush=True)

            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(RULER_PREPARE),
                "--save_dir", str(out_dir),
                "--benchmark", "synthetic",
                "--task", task,
                "--subset", "test",
                "--tokenizer_path", tokenizer_path,
                "--tokenizer_type", "hf",
                "--max_seq_length", str(length),
                "--num_samples", str(num_samples),
                "--model_template_type", "base",
            ]
            t0 = time.time()
            print(f"[run]  L={length:>6} task={task:<18} ...", flush=True)
            try:
                result = subprocess.run(
                    cmd, cwd=RULER_PREPARE.parent, check=False,
                    capture_output=True, text=True, timeout=timeout_sec, env=subprocess_env,
                )
            except subprocess.TimeoutExpired as e:
                failures.append((length, task, f"timed out after {timeout_sec}s (likely infinite loop in upstream qa.py)"))
                continue
            except Exception as e:
                failures.append((length, task, f"subprocess exception: {e}"))
                continue
            elapsed = time.time() - t0
            if result.returncode != 0:
                failures.append((length, task, f"rc={result.returncode}; stderr_tail={result.stderr[-800:]}"))
                continue
            n, err = _validate_jsonl(target)
            if err is not None or n < expected_min_samples:
                failures.append((length, task, f"validation failed: {err} (n={n}); stdout_tail={result.stdout[-400:]}"))
                continue
            print(f"[done] L={length:>6} task={task:<18} n={n} ({elapsed:.1f}s)", flush=True)
            successes.append((length, task, n, False))
    return successes, failures, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-name", required=True, help="Short tag for this tokenizer, e.g. 'qwen3'.")
    p.add_argument("--tokenizer-path", required=True, help="Path or HF id of the tokenizer.")
    p.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    p.add_argument("--tasks", default="all", help="Comma-separated task list, or 'all'.")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--out-root", default="datasets/ruler")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--timeout-sec", type=int, default=SUBPROCESS_TIMEOUT_SEC,
                   help="Per-subprocess timeout for one (length, task) pair.")
    p.add_argument("--force", action="store_true",
                   help="Generate combinations outside the paper's evaluation grid; "
                        "additional source resources may be required.")
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = ALL_TASKS if args.tasks == "all" else args.tasks.split(",")
    for t in tasks:
        if t not in ALL_TASKS:
            raise SystemExit(f"unknown task {t!r}; must be one of {ALL_TASKS}")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizer: {args.tokenizer_name} ({args.tokenizer_path})")
    print(f"Lengths: {lengths}")
    print(f"Tasks: {tasks}")
    print(f"Output: {out_root / args.tokenizer_name}")

    successes, failures, skipped = generate(
        args.tokenizer_name, args.tokenizer_path,
        lengths, tasks, args.num_samples, out_root, args.overwrite,
        timeout_sec=args.timeout_sec, force=args.force,
    )
    print(f"\n[summary] successes={len(successes)} skipped={len(skipped)} failures={len(failures)}")
    if skipped:
        print("[SKIPPED] (outside the paper's evaluation grid)")
        for L, t, reason in skipped:
            print(f"  L={L} {t}: {reason}")
    if failures:
        print("[FAILED]:")
        for L, t, err in failures:
            print(f"  L={L} {t}: {err}")
        sys.exit(1)
    print("[OK] Generation complete (skipped combos are intentional).")


if __name__ == "__main__":
    main()
