"""Stream local DCLM parquet shards and pack EOS-terminated documents."""

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


def build_dataset(tokenizer, data_dir, max_len=16384):
    root = Path(data_dir)
    files = sorted(str(p) for p in root.rglob('*.parquet')) if root.is_dir() else []
    if not files:
        raise ValueError(f'Expected a directory containing DCLM parquet shards: {root}')
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define an EOS token')
    raw = load_dataset('parquet', data_files={'train': files}, split='train', streaming=True)

    def pack(examples):
        encoded = tokenizer(examples['text'], max_length=10**6, truncation=True)['input_ids']
        tokens = np.concatenate([np.append(ids, tokenizer.eos_token_id) for ids in encoded])
        usable = len(tokens) // max_len * max_len
        ids = torch.from_numpy(tokens[:usable].astype(np.int64).reshape(-1, max_len))
        return {'input_ids': ids, 'labels': ids.clone()}

    return raw.map(pack, batched=True, batch_size=1000, remove_columns=list(raw.column_names))
