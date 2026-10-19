"""LongPPL (ICLR 2025, Eq. 2-4) with reusable reference annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def short_windows(length, context, stride):
    """Batch d targets with a common start; context ranges K..K+d-1."""
    if context < 1 or not 1 <= stride <= context:
        raise ValueError("Require K >= d >= 1")
    for start in range(context, length, stride):
        yield start - context, start, min(start + stride, length)


def key_mask(long_logp, short_logp, alpha, beta):
    return [
        math.isfinite(s) and l - s > alpha and l > beta
        for l, s in zip(long_logp, short_logp, strict=True)
    ]


def selected_spans(offsets, selected):
    spans = []
    for (start, end), keep in zip(offsets, selected, strict=True):
        if not keep or end <= start:
            continue
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return spans


def align_mask(offsets, spans):
    """Include only target tokens fully covered by selected character spans."""
    selected = []
    j = 0
    for start, end in offsets:
        while j < len(spans) and spans[j][1] <= start:
            j += 1
        selected.append(
            end > start and j < len(spans)
            and spans[j][0] <= start and end <= spans[j][1]
        )
    if selected:
        selected[0] = False
    return selected


def summarize(rows):
    if not rows:
        raise ValueError("No documents scored")
    count = sum(row["key_tokens"] for row in rows)
    if count == 0:
        raise ValueError("Reference selected zero aligned key tokens; LongPPL is undefined")
    total = sum(row["tokens_scored"] for row in rows)
    nonempty = [row for row in rows if row["key_tokens"]]
    return {
        "documents": len(rows),
        "documents_with_key_tokens": len(nonempty),
        "key_tokens": count,
        "tokens_scored": total,
        "key_token_fraction": count / total,
        "longppl_pooled": math.exp(sum(row["key_nll_sum"] for row in rows) / count),
        "ppl_pooled": math.exp(sum(row["nll_sum"] for row in rows) / total),
        "longppl_document_mean": sum(row["longppl"] for row in nonempty) / len(nonempty),
    }


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    return tokenizer


def load_model(path, device):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    print(json.dumps({"loading_model": path, "device": device}), flush=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    hybrids = {"kda", "gated_deltanet", "gla"}
    options = {"attn_implementation": "flash_attention_2"} if config.model_type not in hybrids else {}
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, **options,
    ).to(device).eval()
    return model


def log_probabilities(model, ids, device, first_target=1, head_chunk=256):
    """Teacher forcing with an unchanged decoder; chunk only the vocabulary head."""
    import torch
    import torch.nn.functional as F
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    if not 1 <= first_target < len(ids):
        raise ValueError("Need at least one context and one scored token")
    with torch.inference_mode():
        hidden = model.get_decoder()(
            input_ids=tokens[:, :-1], use_cache=False, return_dict=True,
        ).last_hidden_state
        values = []
        for start in range(first_target, len(ids), head_chunk):
            end = min(start + head_chunk, len(ids))
            logits = model.get_output_embeddings()(hidden[:, start - 1:end - 1]).float()
            logp = -F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tokens[0, start:end], reduction="none",
            )
            values.extend(logp.double().cpu().tolist())
    return [float("nan")] * first_target + values


def verify_scorer(model, ids, device):
    import torch
    import torch.nn.functional as F
    ids = ids[:128]
    actual = log_probabilities(model, ids, device, head_chunk=17)[1:]
    with torch.inference_mode():
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=tokens[:, :-1], use_cache=False).logits.float()
        expected = -F.cross_entropy(logits[0], tokens[0, 1:], reduction="none")
    error = (torch.tensor(actual, device=device) - expected).abs().max().item()
    if error > 0.02:
        raise ValueError(f"Chunked scoring differs from normal forward: {error}")
    return {"max_logp_error_vs_normal_forward": error, "tokens": len(ids)}


def prepare(args):
    from datasets import Dataset
    source = Dataset.from_file(args.arrow)
    models = read_json(args.models)
    tokenizer = load_tokenizer(models[0]["path"])
    documents = []
    for row in source.shuffle(seed=args.seed, keep_in_memory=True):
        index = row["_id"]
        text = row["context"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) > args.max_length:
            text = text[:encoded["offset_mapping"][args.max_length - 1][1]]
        encoded = tokenizer(text, add_special_tokens=False)
        while len(encoded["input_ids"]) > args.max_length:
            text = text[:-1]
            encoded = tokenizer(text, add_special_tokens=False)
        if len(encoded["input_ids"]) <= args.context + args.stride:
            continue
        documents.append({"id": index, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "target_tokens": len(encoded["input_ids"])})
        if len(documents) == args.documents:
            break
    if len(documents) != args.documents:
        raise ValueError(f"Only {len(documents)} suitable documents")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with (output / "documents.jsonl").open("w") as handle:
        for doc in documents:
            handle.write(json.dumps(doc) + "\n")
    write_json(output / "dataset.json", {
        "source": str(Path(args.arrow).resolve()), "dataset": "Xnhyacinth/LongBench",
        "subset": "gov_report", "split": "test", "text_field": "context",
        "seed": args.seed, "documents": len(documents), "max_length": args.max_length,
        "truncation_tokenizer": models[0]["path"], "context": args.context, "stride": args.stride,
        "ids": [doc["id"] for doc in documents], "fingerprint": digest(documents),
        "note": "GovReport subset via LongBench; not claimed to be the paper's exact 50 documents.",
    })
    print(json.dumps({"prepared": str(output), "documents": len(documents),
                      "min_tokens": min(doc["target_tokens"] for doc in documents),
                      "max_tokens": max(doc["target_tokens"] for doc in documents)}), flush=True)


def run(args):
    import gc
    import numpy as np
    import torch
    free, total = torch.cuda.mem_get_info(args.device)
    print(json.dumps({"gpu": torch.cuda.get_device_name(args.device), "free_bytes": free,
                      "total_bytes": total}), flush=True)
    if free < 24 * 1024**3:
        raise RuntimeError("Allocated GPU has less than 24 GiB free before reference loading")
    root = Path(args.data)
    documents = [json.loads(line) for line in (root / "documents.jsonl").read_text().splitlines()]
    if args.limit:
        documents = documents[:args.limit]
    models = read_json(args.models)
    if args.model_name:
        models = [model for model in models if model["name"] == args.model_name]
        if not models:
            raise ValueError("Unknown model name")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"reference": str(Path(args.reference).resolve()), "alpha": args.alpha, "beta": args.beta,
                "context": args.context, "stride": args.stride, "max_length": args.max_length,
                "dataset": read_json(root / "dataset.json"), "reference_label": args.reference_label,
                "short_context_policy": "K..K+d-1 tokens; common left boundary for each d targets",
                "bos": False, "chat_template": False, "aggregation": "pooled token NLL; document mean also reported",
                "reference_weight_files": weight_identity(args.reference),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = digest(protocol)
    annotations = output / "annotations" / fingerprint
    annotations.mkdir(parents=True, exist_ok=True)
    write_json(annotations / "protocol.json", protocol)
    reference_tokenizer = load_tokenizer(args.reference)
    reference = None
    for doc in documents:
        cache = annotations / f"{doc['sha256']}.json"
        if cache.exists():
            continue
        if reference is None:
            reference = load_model(args.reference, args.device)
        started = time.monotonic()
        encoded = reference_tokenizer(doc["text"], add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if not (annotations / "scorer_verification.json").exists():
            write_json(annotations / "scorer_verification.json", verify_scorer(reference, ids, args.device))
        if len(ids) > args.max_length:
            raise ValueError("Reference tokenization exceeds max length; prepare using a compatible tokenizer")
        long_logp = log_probabilities(reference, ids, args.device)
        short_logp = [float("nan")] * len(ids)
        for left, first, end in short_windows(len(ids), args.context, args.stride):
            values = log_probabilities(reference, ids[left:end], args.device, first_target=first-left)
            short_logp[first:end] = values[first-left:]
        mask = key_mask(long_logp, short_logp, args.alpha, args.beta)
        spans = selected_spans(encoded["offset_mapping"], mask)
        np.savez_compressed(cache.with_suffix(".npz"), ids=ids, long_logp=long_logp,
                            short_logp=short_logp, key_mask=mask)
        record = {"id": doc["id"], "text_sha256": doc["sha256"], "protocol": fingerprint,
                  "reference_tokens": len(ids), "key_tokens": sum(mask), "spans": spans,
                  "seconds": time.monotonic() - started}
        write_json(cache, record)
        print(json.dumps({"annotation": doc["id"], "key_tokens": sum(mask),
                          "tokens": len(ids), "seconds": record["seconds"]}), flush=True)
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    summaries = []
    for entry in models:
        started = time.monotonic()
        model = load_model(entry["path"], args.device)
        tokenizer = load_tokenizer(entry["path"])
        first_ids = tokenizer(documents[0]["text"], add_special_tokens=False)["input_ids"]
        verification = verify_scorer(model, first_ids, args.device)
        print(json.dumps({"model": entry["name"], "verification": verification}), flush=True)
        rows = []
        for doc in documents:
            encoded = tokenizer(doc["text"], add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded["input_ids"]
            if len(ids) > args.max_length:
                raise ValueError("Target exceeds max length; common text truncation required")
            annotation = read_json(annotations / f"{doc['sha256']}.json")
            with np.load(annotations / f"{doc['sha256']}.npz", allow_pickle=False) as saved:
                identical_tokens = ids == saved["ids"].tolist()
                mask = (saved["key_mask"].tolist() if identical_tokens
                        else align_mask(encoded["offset_mapping"], annotation["spans"]))
            logp = log_probabilities(model, ids, args.device)
            if not all(math.isfinite(value) for value in logp[1:]):
                raise ValueError("Non-finite model probabilities")
            key_nll = -sum(value for value, keep in zip(logp, mask) if keep)
            count = sum(mask)
            row = {"id": doc["id"], "text_sha256": doc["sha256"], "key_tokens": count,
                   "tokenizer_alignment": "identical_token_ids" if identical_tokens else "character_spans",
                   "tokens_scored": len(ids) - 1, "key_nll_sum": key_nll, "nll_sum": -sum(logp[1:]),
                   "longppl": math.exp(key_nll / count) if count else None,
                   "ppl": math.exp(-sum(logp[1:]) / (len(ids) - 1))}
            rows.append(row)
            print(json.dumps({"model": entry["name"], "document": len(rows), **row}), flush=True)
            write_json(output / entry["name"] / "documents.json", rows)
        result = {"model": entry, "protocol": protocol, "annotation_fingerprint": fingerprint,
                  "verification": verification, "summary": summarize(rows),
                  "parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "model_weight_files": weight_identity(entry["path"]),
                  "seconds": time.monotonic() - started,
                  "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}}
        write_json(output / entry["name"] / "result.json", result)
        summaries.append({"model": entry["name"], **result["summary"]})
        write_json(output / "summary.json", summaries)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"complete": True, "summary": summaries}), flush=True)


def weight_identity(path):
    files = sorted([*Path(path).glob("*.safetensors"), *Path(path).glob("pytorch_model*.bin")])
    if not files:
        raise FileNotFoundError(f"No model weight files in {path}")
    return [{"path": str(file.resolve()), "bytes": file.stat().st_size,
             "mtime_ns": file.stat().st_mtime_ns} for file in files]


def probe(args):
    models = read_json(args.models)
    entry = next(model for model in models if model["name"] == args.model_name)
    document = json.loads((Path(args.data) / "documents.jsonl").read_text().splitlines()[0])
    tokenizer = load_tokenizer(entry["path"])
    ids = tokenizer(document["text"], add_special_tokens=False)["input_ids"]
    model = load_model(entry["path"], args.device)
    verification = verify_scorer(model, ids, args.device)
    print(json.dumps({"verification": verification}), flush=True)
    logp = log_probabilities(model, ids, args.device)
    if not all(math.isfinite(value) for value in logp[1:]):
        raise ValueError("Non-finite log probabilities")
    result = {"model": entry, "verification": verification, "tokens": len(ids),
              "document": document["id"], "ppl": math.exp(-sum(logp[1:]) / (len(ids)-1)),
              "note": "Inference validation only; this PPL is not LongPPL."}
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


def validate_tokenizers(args):
    documents = [json.loads(line) for line in (Path(args.data) / "documents.jsonl").read_text().splitlines()]
    texts = [doc["text"] for doc in documents]
    reference = load_tokenizer(args.reference)
    expected = reference(texts, add_special_tokens=False, return_offsets_mapping=True)
    checks = []
    for entry in read_json(args.models):
        actual = load_tokenizer(entry["path"])(texts, add_special_tokens=False, return_offsets_mapping=True)
        ids_equal = expected["input_ids"] == actual["input_ids"]
        offsets_equal = expected["offset_mapping"] == actual["offset_mapping"]
        checks.append({"model": entry["name"], "identical_token_ids": ids_equal,
                       "identical_offsets": offsets_equal, "documents": len(documents)})
        if not ids_equal or not offsets_equal:
            raise ValueError(f"Tokenizers differ on this corpus: {entry['name']}")
        weight_identity(entry["path"])
    result = {"reference": args.reference, "checks": checks,
              "reference_weight_files": weight_identity(args.reference),
              "reference_tokenizer_sha256": hashlib.sha256((Path(args.reference)/"tokenizer.json").read_bytes()).hexdigest()}
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--arrow", required=True)
    prep.add_argument("--documents", type=int, default=50)
    prep.add_argument("--seed", type=int, default=42)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--reference-label", required=True)
    evaluate.add_argument("--alpha", type=float, default=2)
    evaluate.add_argument("--beta", type=float, default=-2)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--model-name")
    evaluate.add_argument("--device", default="cuda")
    check = sub.add_parser("probe")
    check.add_argument("--models", required=True)
    check.add_argument("--model-name", default="kda240_k3a1_full_stage2")
    check.add_argument("--data", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--device", default="cuda")
    tokenizers = sub.add_parser("validate-tokenizers")
    for flag in ("--models", "--reference", "--data", "--output"):
        tokenizers.add_argument(flag, required=True)
    for command in (prep, evaluate):
        command.add_argument("--models", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--context", type=int, default=4096)
        command.add_argument("--stride", type=int, default=1024)
        command.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "probe": probe, "validate-tokenizers": validate_tokenizers}[args.command](args)


if __name__ == "__main__":
    main()
