from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100
FIM_PREFIX_TOKEN = "<|fim_prefix|>"
FIM_SUFFIX_TOKEN = "<|fim_suffix|>"
FIM_PAD_TOKEN = "<|fim_pad|>"


@dataclass(frozen=True)
class CopyTokenIds:
    prefix: int
    suffix: int
    digits: tuple[int, ...]
    pad: int | None = None


def _require_int(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer, received {value!r}.")
    if value < minimum:
        raise ValueError(
            f"`{name}` must be >= {minimum}, received {value}."
        )


def _resolve_token_id(
    tokenizer: PreTrainedTokenizerBase,
    token: str,
) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or (unk_token_id is not None and token_id == unk_token_id)
    ):
        raise ValueError(
            f"Tokenizer does not define a valid token ID for {token!r}."
        )
    return token_id


def resolve_copy_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    require_pad: bool,
) -> CopyTokenIds:
    """Resolve and validate the marker and single-digit token IDs."""
    prefix = _resolve_token_id(tokenizer, FIM_PREFIX_TOKEN)
    suffix = _resolve_token_id(tokenizer, FIM_SUFFIX_TOKEN)
    pad = (
        _resolve_token_id(tokenizer, FIM_PAD_TOKEN)
        if require_pad
        else None
    )

    digit_ids = []
    for digit in range(10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        if isinstance(encoded, Tensor):
            encoded = encoded.tolist()
        if len(encoded) != 1:
            raise ValueError(
                "Copy datasets require every ASCII digit to encode as exactly "
                f"one token; {digit!r} encoded to {encoded!r}."
            )
        digit_id = encoded[0]
        if isinstance(digit_id, bool) or not isinstance(digit_id, int):
            raise ValueError(
                f"Tokenizer returned an invalid ID for digit {digit!r}: "
                f"{digit_id!r}."
            )
        digit_ids.append(digit_id)

    required_ids = [prefix, suffix, *digit_ids]
    if pad is not None:
        required_ids.append(pad)
    if len(set(required_ids)) != len(required_ids):
        raise ValueError(
            "Copy marker, pad, and digit token IDs must all be distinct."
        )

    return CopyTokenIds(
        prefix=prefix,
        suffix=suffix,
        digits=tuple(digit_ids),
        pad=pad,
    )


def required_fill_tokens(
    seq_len: int,
    gap: int,
    key_len: int,
    *,
    shift_labels: bool = False,
) -> int:
    """
    Return the number of fill tokens in one example.

    When labels are shifted in the dataset, one extra raw token is assembled so
    the returned input and label tensors still both have length ``seq_len``.
    """
    _require_int("seq_len", seq_len, 1)
    _require_int("gap", gap, 0)
    _require_int("key_len", key_len, 1)

    marker_and_key_tokens = 2 * (key_len + 2)
    configured_fill_tokens = seq_len - marker_and_key_tokens
    if configured_fill_tokens < gap:
        minimum_seq_len = marker_and_key_tokens + gap
        raise ValueError(
            "Copy layout does not fit: "
            f"seq_len={seq_len}, gap={gap}, key_len={key_len}. "
            f"`seq_len` must be at least {minimum_seq_len}."
        )
    return configured_fill_tokens + int(shift_labels)


def sample_rng(seed: int, sample_index: int) -> np.random.Generator:
    """Create a stateless per-example RNG from a base seed and sample index."""
    _require_int("seed", seed, 0)
    _require_int("sample_index", sample_index, 0)
    entropy = [
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        sample_index & 0xFFFFFFFF,
        (sample_index >> 32) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def build_copy_example(
    *,
    token_ids: CopyTokenIds,
    fill_ids: Sequence[int],
    gap: int,
    key_len: int,
    seed: int,
    sample_index: int,
    shift_labels: bool = False,
) -> dict[str, Tensor]:
    """
    Assemble one copy example and mask loss outside the repeated key.

    ``fill_ids`` is consumed in order across the left, gap, and right regions.
    Removing the two marked key blocks therefore recovers the original fill
    stream exactly.
    """
    _require_int("gap", gap, 0)
    _require_int("key_len", key_len, 1)
    reserved_right_fill = int(shift_labels)
    if len(fill_ids) < gap + reserved_right_fill:
        raise ValueError(
            f"Need at least {gap + reserved_right_fill} fill tokens, "
            f"received {len(fill_ids)}."
        )

    rng = sample_rng(seed, sample_index)
    key_digit_indices = rng.integers(0, len(token_ids.digits), size=key_len)
    key = [token_ids.digits[int(index)] for index in key_digit_indices]

    free_fill_tokens = len(fill_ids) - gap - reserved_right_fill
    left_len = int(rng.integers(0, free_fill_tokens + 1))
    gap_end = left_len + gap
    fill = [int(token_id) for token_id in fill_ids]

    first_block = [token_ids.prefix, *key, token_ids.suffix]
    second_block = [token_ids.prefix, *key, token_ids.suffix]
    raw_ids = [
        *fill[:left_len],
        *first_block,
        *fill[left_len:gap_end],
        *second_block,
        *fill[gap_end:],
    ]
    input_ids = torch.tensor(raw_ids, dtype=torch.long)

    target_start = left_len + len(first_block) + gap + 1
    target_end = target_start + key_len
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[target_start:target_end] = input_ids[target_start:target_end]

    if shift_labels:
        input_ids = input_ids[:-1]
        labels = labels[1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
