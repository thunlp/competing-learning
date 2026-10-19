"""Spaced Retrieval data and signed, bidirectional mixer-output attribution."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
KEY_LEN = 5
def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)

def save_tensor(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def key_digits(numbers):
    numbers = torch.as_tensor(numbers, dtype=torch.long)
    return (numbers[:, None] // torch.tensor([10000, 1000, 100, 10, 1]) % 10) + 2

def key_numbers(keys):
    return ((keys - 2) * torch.tensor([10000, 1000, 100, 10, 1])).sum(-1)

def build_data(config):
    rng = np.random.default_rng(config["data_seed"])
    count = config["eval_examples"]
    numbers = rng.choice(100000, size=count, replace=False)
    keys = key_digits(numbers)
    positions = torch.arange(count) % KEY_LEN
    alternate = keys.clone()
    offsets = torch.from_numpy(rng.integers(1, 10, size=count))
    alternate[torch.arange(count), positions] = (
        (keys[torch.arange(count), positions] - 2 + offsets) % 10 + 2
    )
    reserved = np.unique(np.concatenate([numbers, key_numbers(alternate).numpy()]))
    available = np.setdiff1d(np.arange(100000), reserved)
    train_numbers = rng.choice(available, size=config["train_examples"], replace=False)
    return {
        "train": {
            "keys": key_digits(train_numbers),
            "starts": torch.from_numpy(rng.integers(0, 21, size=len(train_numbers))),
        },
        "evaluation": {
            "keys": keys,
            "alternate": alternate,
            "positions": positions,
            "starts": torch.from_numpy(rng.integers(0, 21, size=count)),
        },
        "reserved_key_count": len(reserved),
    }

def select_rows(rows, index):
    return {key: value[index] for key, value in rows.items() if isinstance(value, torch.Tensor)}

def recall_batch(rows, gap, device, *, prefix=False, alternate=False):
    keys = rows["alternate" if alternate else "keys"].to(device)
    key_len = keys.shape[1]
    starts = rows["starts"].to(device)
    batch = torch.arange(len(keys), device=device)
    query = starts + key_len + 1 + gap
    ids = torch.zeros(len(keys), gap + 2 * key_len + 22, dtype=torch.long, device=device)
    ids[batch, starts] = 1
    ids[batch[:, None], starts[:, None] + 1 + torch.arange(key_len, device=device)] = keys
    ids[batch, query] = 1
    ids[batch[:, None], query[:, None] + 1 + torch.arange(key_len, device=device)] = keys
    prediction_positions = query[:, None] + torch.arange(key_len, device=device)
    if prefix:
        target_position = rows["positions"].to(device)
        query = query + target_position
        width = int(query.max()) + 1
        ids = ids[:, :width]
        mask = torch.arange(width, device=device)[None, :] <= query[:, None]
        ids = ids.masked_fill(~mask, 0)
        targets = keys[batch, target_position]
        return {"input_ids": ids, "attention_mask": mask.long()}, query, targets
    labels = torch.full_like(ids, -100)
    labels[batch[:, None], prediction_positions + 1] = keys
    return {"input_ids": ids, "labels": labels}, prediction_positions, keys

def logits_at(model, inputs, positions):
    output = model(**inputs, use_cache=False, return_dict=True,
                   return_last_hidden_state_only=True)
    hidden = output.last_hidden_state
    batch = torch.arange(len(hidden), device=hidden.device)
    selected = hidden[batch, positions] if positions.ndim == 1 else hidden[batch[:, None], positions]
    # Read the native head in FP32 to avoid rounding small diagnostic loss changes.
    return F.linear(selected.float(), model.lm_head.weight.float(), None)

def token_losses(model, rows, gap, device):
    inputs, positions, targets = recall_batch(rows, gap, device)
    inputs.pop("labels")
    logits = logits_at(model, inputs, positions)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)

@torch.no_grad()
def evaluate_losses(model, rows, gaps, device, batch_size):
    model.eval()
    result = {}
    for gap in gaps:
        pieces = []
        for start in range(0, len(rows["keys"]), batch_size):
            pieces.append(token_losses(model, select_rows(rows, slice(start, start + batch_size)),
                                       gap, device).cpu())
        result[str(gap)] = torch.cat(pieces).double()
    return result

def summarize_values(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    return {"mean": values.mean().item(),
            "se": (values.std(unbiased=True) / math.sqrt(len(values))).item() if len(values) > 1 else 0.0,
            "n": len(values)}

@torch.no_grad()
def native_accuracy(model, rows, gaps, device, batch_size):
    result = {}
    key_len = rows["keys"].shape[1]
    for gap in gaps:
        predictions = []
        for start in range(0, len(rows["keys"]), batch_size):
            part = select_rows(rows, slice(start, start + batch_size))
            first = {**part, "positions": torch.zeros(len(part["keys"]), dtype=torch.long)}
            inputs, query, _ = recall_batch(first, gap, device, prefix=True)
            ids = F.pad(inputs["input_ids"], (0, key_len))
            batch = torch.arange(len(ids), device=device)
            tokens = []
            for offset in range(key_len):
                pos = query + offset
                width = int(pos.max()) + 1
                mask = torch.arange(width, device=device)[None, :] <= pos[:, None]
                logits = logits_at(model, {"input_ids": ids[:, :width], "attention_mask": mask.long()}, pos)
                token = logits.argmax(-1)
                tokens.append(token)
                ids[batch, pos + 1] = token
            predictions.append(torch.stack(tokens, -1).cpu())
        correct = torch.cat(predictions) == rows["keys"]
        result[str(gap)] = {"first_accuracy": correct[:, 0].float().mean().item(),
                            "character_accuracy": correct.float().mean().item(),
                            "exact_accuracy": correct.all(-1).float().mean().item(), "n": len(correct)}
    return result

@contextmanager
def mixer_hooks(model, positions, captured=None, replacements=None):
    handles = []
    replacements = replacements or {}
    for layer_index, layer in enumerate(model.model.layers):
        def hook(module, inputs, output, index=layer_index):
            hidden = output[0]
            batch = torch.arange(len(hidden), device=hidden.device)
            if captured is not None:
                captured[index] = hidden[batch, positions].detach().clone()
            if index in replacements:
                patched = hidden.clone()
                patched[batch, positions] = replacements[index]
                return (patched, *output[1:])
        handles.append(layer.attn.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

@torch.no_grad()
def causal_contributions(model, rows, gaps, device, batch_size):
    model.eval()
    result = {}
    for gap in gaps:
        effects = {str(layer): [] for layer in range(4)}
        probability_effects = {str(layer): [] for layer in range(4)}
        clean_margins = []
        self_error = 0.0
        for start in range(0, len(rows["keys"]), batch_size):
            part = select_rows(rows, slice(start, start + batch_size))
            inputs_a, query, target_a = recall_batch(part, gap, device, prefix=True)
            inputs_b, query_b, target_b = recall_batch(part, gap, device, prefix=True, alternate=True)
            if not torch.equal(query, query_b) or bool((target_a == target_b).any()):
                raise AssertionError("Counterfactual alignment failed")
            saved_a, saved_b = {}, {}
            with mixer_hooks(model, query, captured=saved_a):
                logits_a = logits_at(model, inputs_a, query)
            with mixer_hooks(model, query, captured=saved_b):
                logits_b = logits_at(model, inputs_b, query)
            batch = torch.arange(len(query), device=device)
            margin_a = logits_a[batch, target_a] - logits_a[batch, target_b]
            margin_b = logits_b[batch, target_b] - logits_b[batch, target_a]
            clean_margins.append(((margin_a + margin_b) / 2).cpu())
            for layer in range(4):
                with mixer_hooks(model, query, replacements={layer: saved_b[layer]}):
                    patched_a = logits_at(model, inputs_a, query)
                with mixer_hooks(model, query, replacements={layer: saved_a[layer]}):
                    patched_b = logits_at(model, inputs_b, query)
                effect = ((margin_a - (patched_a[batch, target_a] - patched_a[batch, target_b]))
                          + (margin_b - (patched_b[batch, target_b] - patched_b[batch, target_a]))) / 2
                effects[str(layer)].append(effect.cpu())
                probability_effects[str(layer)].append(((
                    logits_a.softmax(-1)[batch, target_a] - patched_a.softmax(-1)[batch, target_a]
                    + logits_b.softmax(-1)[batch, target_b] - patched_b.softmax(-1)[batch, target_b]
                ) / 2).cpu())
                if start == 0:
                    with mixer_hooks(model, query, replacements={layer: saved_a[layer]}):
                        self_logits = logits_at(model, inputs_a, query)
                    self_error = max(self_error, (self_logits - logits_a).abs().max().item())
        if self_error > 1e-5:
            raise AssertionError(f"Same-input replacement changed logits: {self_error}")
        result[str(gap)] = {"effect": {key: torch.cat(value) for key, value in effects.items()},
                            "probability_effect": {key: torch.cat(value) for key, value in probability_effects.items()},
                            "clean_margin": torch.cat(clean_margins), "positions": rows["positions"],
                            "self_patch_max_error": self_error}
    return result

def states_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(states_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(states_equal(a, b) for a, b in zip(left, right))
    return left == right

def initialize_model(config, seed, device):
    from modeling.kda import KDAConfig, KDAForCausalLM
    torch.manual_seed(seed)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = KDAForCausalLM(KDAConfig.from_json_file(config["model_config"]))
    finally:
        torch.set_default_dtype(old_dtype)
    model.tie_weights()
    if model.lm_head.weight is not model.model.embeddings.weight:
        raise AssertionError("Expected tied native head")
    model.to(device)
    model.gradient_checkpointing_enable()
    return model
