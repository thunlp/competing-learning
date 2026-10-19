"""RULER v1 evaluation used by the paper (57 task/length entries)."""
import gc, json, os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
DEFAULT_RULER_V1_DATA_ROOT=Path("datasets/ruler")
DEFAULT_OUTPUT_DIR=Path("outputs/evaluation")
RULER_V1_MATCH_TYPE={
    "niah_single_1":"all", "niah_single_2":"all", "niah_single_3":"all",
    "niah_multikey_1":"all", "niah_multikey_2":"all", "niah_multikey_3":"all",
    "niah_multivalue":"all", "niah_multiquery":"all", "vt":"all", "cwe":"all",
    "fwe":"all", "qa_1":"part", "qa_2":"part",
}
DEFAULT_RULER_V1_LENGTHS=[1024,2048,4096,8192,16384]
DEFAULT_RULER_V1_TASKS=list(RULER_V1_MATCH_TYPE)
def ruler_v1_skip_reason(task: str, length: int) -> Optional[str]:
    """Combinations excluded from the paper's 57-entry evaluation grid."""
    if task == "cwe":
        return "CWE is excluded from the paper's evaluation grid"
    if task == "qa_1" and length < 2048:
        return "QA-1 is evaluated from 2048 tokens"
    if task == "qa_2" and length < 4096:
        return "QA-2 is evaluated from 4096 tokens"
    return None

def collate_inputs_answers(batch):
    inputs = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    return inputs, answers

def load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def make_entry(
    *,
    model_name: str,
    benchmark: str,
    task: str,
    metric: str,
    value: float,
    timestamp: str,
    model_path: str,
    n_shot: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> dict:
    entry = {
        "model_name": model_name,
        "timestamp": timestamp,
        "task": task,
        "version": None,
        "filter": benchmark,
        "n_shot": n_shot,
        "metric": metric,
        "direction": "higher",
        "value": value,
        "plusminus": None,
        "stderr": None,
        "benchmark": benchmark,
        "model_path": model_path,
    }
    if metadata:
        entry["metadata"] = metadata
    return entry

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not create a unique path for {path}")

def open_log(output_dir: Path, run_name: str, suffix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    return open(unique_path(output_dir / f"{run_name}{suffix}"), "w", encoding="utf-8")

def save_json_results(output_dir: Path, run_name: str, entries: Sequence[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = unique_path(output_dir / f"{run_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(entries), f, ensure_ascii=False, indent=2)
    return output_path

def write_and_print(accelerator, log, message: str) -> None:
    accelerator.print(message)
    if accelerator.is_main_process and log is not None:
        log.write(message + "\n")
        log.flush()

def infer_tokenizer_name(tokenizer_path: str) -> str:
    """Derive a short tag for a tokenizer to look up RULER v1 data folders.

    Examples:
      /.../qwen3-tokenizer        -> 'qwen3'
      /.../tokenizer/qwen3-tokenizer -> 'qwen3'
      meta-llama/Llama-3.1-8B     -> 'Llama-3.1-8B'
      gpt2                        -> 'gpt2'
    """
    raw = tokenizer_path.rstrip("/").split("/")[-1]
    for suffix in ("-tokenizer", "_tokenizer", ".tokenizer"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw

class RulerV1Dataset:
    def __init__(self, data: Sequence[dict]):
        self.inputs = [f"{item['input']}{item['answer_prefix']}" for item in data]
        self.answers = [item["outputs"] for item in data]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.answers[idx]

def _ruler_v1_match(output: str, expected: Sequence[str], match_type: str) -> bool:
    """Replicates nemo_skills.dataset.ruler.ruler_score match semantics.

    'all': every expected string must appear as a substring of `output`.
    'part': any expected string (case-insensitive) appears as a substring.
    """
    if match_type == "all":
        return all(ans in output for ans in expected)
    if match_type == "part":
        out_lower = output.lower()
        return any(str(ans).lower() in out_lower for ans in expected)
    raise ValueError(f"Unknown RULER v1 match_type: {match_type!r}")

def _ruler_v1_tokens_to_generate(task: str) -> int:
    """Per-task target generation length, matching RULER constants."""
    if task.startswith("niah"):
        return 128
    if task == "vt":
        return 30
    if task == "cwe":
        return 120
    if task == "fwe":
        return 50
    if task.startswith("qa"):
        return 32
    raise ValueError(f"Unknown RULER v1 task: {task!r}")

def _ruler_v1_check_data(
    data_root: Path,
    tokenizer_name: str,
    lengths: Sequence[int],
    tasks: Sequence[str],
    strict: bool = False,
) -> Path:
    """Raise FileNotFoundError with an actionable message if any required file is missing.

    Combos returned by ``ruler_v1_skip_reason`` are *not* required unless ``strict=True``.
    """
    base = Path(data_root) / tokenizer_name
    if not base.exists():
        raise FileNotFoundError(
            f"RULER v1 data for tokenizer '{tokenizer_name}' not found at {base}. "
            f"Generate it first with:\n"
            f"  python -m evaluation.prepare_ruler "
            f"--tokenizer-name {tokenizer_name} --tokenizer-path <path>"
        )
    missing = []
    for length in lengths:
        for task in tasks:
            if not strict and ruler_v1_skip_reason(task, length) is not None:
                continue
            p = base / str(length) / task / "test.jsonl"
            if not p.exists():
                missing.append(str(p.relative_to(data_root)))
    if missing:
        raise FileNotFoundError(
            f"RULER v1 data missing {len(missing)} file(s) under {base}:\n  "
            + "\n  ".join(missing[:10])
            + ("\n  ..." if len(missing) > 10 else "")
            + f"\nGenerate with:\n  python -m evaluation.prepare_ruler "
            f"--tokenizer-name {tokenizer_name} --tokenizer-path <path> "
            f"--lengths {','.join(str(L) for L in lengths)} --tasks {','.join(tasks)}"
        )
    return base

def run_ruler_v1(
    *,
    models: Dict[str, str],
    tokenizer_path: str,
    data_root: Path = DEFAULT_RULER_V1_DATA_ROOT,
    tokenizer_name: Optional[str] = None,
    lengths: Sequence[int] = DEFAULT_RULER_V1_LENGTHS,
    tasks: Sequence[str] = DEFAULT_RULER_V1_TASKS,
    base_batch_size: int = 64,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_name: Optional[str] = None,
    save_results: bool = True,
    max_samples: Optional[int] = None,
) -> List[dict]:
    """Evaluate RULER v1 (13 tasks) using the same HF + accelerate stack as run_niah.

    Data layout (raw RULER schema, produced by evaluation.prepare_ruler):
      ``<data_root>/<tokenizer_name>/<length>/<task>/test.jsonl``

    The ``tokenizer_name`` is derived from ``tokenizer_path`` if not given;
    missing data files raise ``FileNotFoundError`` *before* any model is loaded.
    """
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for t in tasks:
        if t not in RULER_V1_MATCH_TYPE:
            raise ValueError(f"Unknown RULER v1 task: {t!r}; must be one of {list(RULER_V1_MATCH_TYPE)}")

    if tokenizer_name is None:
        tokenizer_name = infer_tokenizer_name(tokenizer_path)

    base = _ruler_v1_check_data(Path(data_root), tokenizer_name, lengths, tasks)

    accelerator = Accelerator()
    device = accelerator.device
    timestamp = datetime.utcnow().isoformat()
    run_name = run_name or f"ruler_v1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    entries: List[dict] = []
    log = open_log(output_dir, run_name, ".log") if accelerator.is_main_process else None

    try:
        write_and_print(
            accelerator, log,
            f"RULER v1 tokenizer_name='{tokenizer_name}' data_root={base}",
        )
        write_and_print(accelerator, log, "Found models:\n" + "\n".join([f"{k}: {v}" for k, v in models.items()]))
        write_and_print(accelerator, log, "Loading tokenizer...")
        with accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        min_length = min(lengths)

        for model_name, ckpt in models.items():
            with accelerator.main_process_first():
                model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype="auto").to(device)
            model.eval()
            model = accelerator.prepare(model)
            write_and_print(accelerator, log, f"Using model: {model_name}, checkpoint: {ckpt}")

            for task in tasks:
                match_type = RULER_V1_MATCH_TYPE[task]
                tokens_to_generate = _ruler_v1_tokens_to_generate(task)
                for length in lengths:
                    reason = ruler_v1_skip_reason(task, length)
                    if reason is not None:
                        write_and_print(
                            accelerator, log,
                            f"RULER_v1[{tokenizer_name}] SKIP task={task} length={length} reason={reason}",
                        )
                        continue
                    data_file = base / str(length) / task / "test.jsonl"
                    dataset = load_jsonl(data_file)
                    if max_samples is not None:
                        dataset = dataset[:max_samples]
                    batch_size = max(base_batch_size * min_length // length, 1)
                    dataloader = DataLoader(
                        RulerV1Dataset(dataset),
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_inputs_answers,
                        num_workers=4,
                        pin_memory=True,
                    )
                    dataloader = accelerator.prepare(dataloader)

                    total = 0
                    correct_local = 0
                    for batch in tqdm(
                        dataloader,
                        desc=f"{task} L{length}",
                        leave=False,
                        disable=not accelerator.is_local_main_process,
                    ):
                        inputs, answers = batch
                        inputs_tokenized = tokenizer(
                            list(inputs),
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=65536,
                        )
                        inputs_tokenized = {k: v.to(device) for k, v in inputs_tokenized.items()}

                        with torch.no_grad():
                            raw_model = accelerator.unwrap_model(model)
                            output_ids = raw_model.generate(
                                **inputs_tokenized,
                                max_length=inputs_tokenized["input_ids"].shape[1] + tokens_to_generate,
                                do_sample=False,
                            )

                        decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                        trimmed_outputs = [out[len(inp):] for out, inp in zip(decoded_outputs, inputs)]
                        correct = 0
                        for output, expected in zip(trimmed_outputs, answers):
                            correct += int(_ruler_v1_match(output, expected, match_type))

                        correct_local += correct
                        total += len(inputs)

                    correct_total = accelerator.gather(torch.tensor([correct_local], device=device)).sum().item()
                    total_total = accelerator.gather(torch.tensor([total], device=device)).sum().item()
                    accuracy = correct_total / total_total if total_total else 0.0
                    message = (
                        f"RULER_v1[{tokenizer_name}] task={task} length={length} "
                        f"match={match_type} correct={correct_total}/{total_total} acc={accuracy:.4f}"
                    )
                    write_and_print(accelerator, log, message)

                    if accelerator.is_main_process:
                        entries.append(
                            make_entry(
                                model_name=model_name,
                                benchmark="RULER_v1",
                                task=f"RULER_v1/{tokenizer_name}/{task}/len_{length}",
                                metric="accuracy",
                                value=float(accuracy),
                                timestamp=timestamp,
                                model_path=ckpt,
                                metadata={
                                    "task": task,
                                    "length": length,
                                    "match_type": match_type,
                                    "tokenizer_name": tokenizer_name,
                                    "tokens_to_generate": tokens_to_generate,
                                    "correct": correct_total,
                                    "total": total_total,
                                    "max_samples": max_samples,
                                },
                            )
                        )

                if accelerator.is_main_process and log is not None:
                    log.write("\n")
                    log.flush()

            del model
            torch.cuda.empty_cache()
            gc.collect()

        del tokenizer
        if accelerator.is_main_process:
            results_path = save_json_results(output_dir, run_name, entries)
            write_and_print(accelerator, log, f"Saved structured results to {results_path}")
        return entries if accelerator.is_main_process else []
    finally:
        if log is not None:
            log.close()
