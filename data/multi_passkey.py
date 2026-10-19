"""Deterministic, natural-text multi-passkey retrieval data."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk
from torch import Tensor
from torch.utils.data import Dataset

from .copy_utils import IGNORE_INDEX, sample_rng


REMEMBER_TEMPLATE = "\n\nRemember passkey {number}: "
REMEMBER_SUFFIX = "\n\n"
QUESTION_TEMPLATE = "\n\nWhat is passkey {number}?\nPasskey {number}: "
ANSWER_SUFFIX = "\n"
PASSKEY_ALPHABET = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
)

_JSON_SUFFIXES = (".json", ".jsonl", ".jsonl.zst")
_SAVED_DATASET_MARKERS = ("state.json", "dataset_dict.json")


def _require_int(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer, received {value!r}.")
    if value < minimum:
        raise ValueError(
            f"`{name}` must be >= {minimum}, received {value!r}."
        )


def _require_depth_ratio(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number, received {value!r}.")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"`{name}` must be finite and between 0 and 1, "
            f"received {value!r}."
        )
    return ratio


def _as_token_ids(encoded: Any, *, description: str) -> tuple[int, ...]:
    if isinstance(encoded, Tensor):
        encoded = encoded.tolist()
    elif hasattr(encoded, "tolist") and not isinstance(encoded, Sequence):
        encoded = encoded.tolist()

    if (
        isinstance(encoded, (str, bytes))
        or not isinstance(encoded, Sequence)
    ):
        raise ValueError(
            f"Tokenizer returned invalid IDs for {description}: {encoded!r}."
        )

    token_ids: list[int] = []
    for token_id in encoded:
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise ValueError(
                f"Tokenizer returned an invalid ID for {description}: "
                f"{token_id!r}."
            )
        if token_id < 0:
            raise ValueError(
                f"Tokenizer returned a negative ID for {description}: "
                f"{token_id!r}."
            )
        token_ids.append(int(token_id))
    return tuple(token_ids)


def _encode(tokenizer: Any, text: str, *, description: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError as error:
        raise TypeError(
            "Tokenizer must provide encode(text, add_special_tokens=False)."
        ) from error
    return _as_token_ids(encoded, description=description)


def _resolve_digit_token_ids(tokenizer: Any) -> tuple[int, ...]:
    digit_ids = tuple(
        _encode(tokenizer, digit, description=f"ASCII digit {digit!r}")
        for digit in "0123456789"
    )
    invalid = [
        digit
        for digit, token_ids in zip("0123456789", digit_ids)
        if len(token_ids) != 1
    ]
    if invalid:
        details = ", ".join(
            f"{digit!r}->{list(token_ids)!r}"
            for digit, token_ids in zip("0123456789", digit_ids)
            if len(token_ids) != 1
        )
        raise ValueError(
            "Multi-passkey data requires every ASCII digit to encode as "
            f"exactly one token; invalid encodings: {details}."
        )

    flattened = tuple(token_ids[0] for token_ids in digit_ids)
    if len(set(flattened)) != len(flattened):
        raise ValueError(
            "Multi-passkey data requires the ten ASCII digits to have "
            "distinct token IDs."
        )
    return flattened


def _resolve_word_token_ids(
    tokenizer: Any,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    resolved = []
    for word in PASSKEY_ALPHABET:
        first = _encode(
            tokenizer,
            word,
            description=f"passkey word {word!r}",
        )
        continuation = _encode(
            tokenizer,
            f" {word}",
            description=f"spaced passkey word {word!r}",
        )
        if not first or not continuation:
            raise ValueError(
                f"Passkey word {word!r} must have non-empty tokenizations.",
            )
        resolved.append((first, continuation))
    return tuple(resolved)


def _depth_positions(
    *,
    background_token_count: int,
    gap: int,
    min_ratio: float,
    max_ratio: float,
) -> tuple[int, int]:
    if background_token_count == 0:
        if min_ratio > 0.0:
            raise ValueError(
                "Multi-passkey depth range cannot be represented with "
                "zero background tokens; set `min_depth_ratio=0` or "
                "increase `seq_len`."
            )
        return 0, 0

    rounding_tolerance = 1e-12 * background_token_count
    min_depth_position = math.ceil(
        min_ratio * background_token_count - rounding_tolerance
    )
    max_depth_position = math.floor(
        max_ratio * background_token_count + rounding_tolerance
    )
    max_depth_position = min(
        max_depth_position,
        background_token_count - gap,
    )
    if min_depth_position > max_depth_position:
        raise ValueError(
            "Multi-passkey depth/gap constraints have no valid "
            "insertion position: "
            f"background_tokens={background_token_count}, gap={gap}, "
            f"depth_range=[{min_ratio}, {max_ratio}]."
        )
    return min_depth_position, max_depth_position


def _resolve_local_path(data_dir: str | os.PathLike[str]) -> Path:
    if isinstance(data_dir, bool) or not isinstance(data_dir, (str, os.PathLike)):
        raise TypeError(
            "`data_dir` must be a local filesystem path, "
            f"received {data_dir!r}."
        )

    raw_path = os.fspath(data_dir)
    if not raw_path or "://" in raw_path:
        raise ValueError(
            f"`data_dir` must be an existing local path, received {raw_path!r}."
        )

    path = Path(raw_path).expanduser()
    if not path.exists():
        raise ValueError(
            f"`data_dir` must be an existing local path, received {raw_path!r}."
        )
    return path.resolve()


def _select_source_split(source: Any, *, data_dir: Path) -> Any:
    if isinstance(source, Mapping):
        if "train" in source:
            source = source["train"]
        elif len(source) == 1:
            source = next(iter(source.values()))
        else:
            available = ", ".join(str(name) for name in source)
            raise ValueError(
                f"Dataset at {data_dir} has no 'train' split; "
                f"available splits: {available}."
            )

    if hasattr(source, "__len__") and hasattr(source, "__getitem__"):
        try:
            source_size = len(source)
        except TypeError:
            source_size = None
        if source_size is not None and source_size < 1:
            raise ValueError(f"Dataset at {data_dir} is empty.")
    elif not hasattr(source, "__iter__"):
        raise TypeError(f"Dataset at {data_dir} is not iterable.")
    return source


def _load_local_source(data_dir: str | os.PathLike[str]) -> tuple[Any, Path]:
    path = _resolve_local_path(data_dir)
    if path.is_file():
        lower_name = path.name.lower()
        if not lower_name.endswith(_JSON_SUFFIXES):
            raise ValueError(
                "`data_dir` files must end in .json, .jsonl, or .jsonl.zst; "
                f"received {path}."
            )
        source = load_dataset(
            "json",
            data_files=str(path),
            split="train",
            streaming=True,
        )
    elif path.is_dir():
        is_saved_dataset = any(
            (path / marker).is_file() for marker in _SAVED_DATASET_MARKERS
        )
        if is_saved_dataset:
            source = load_from_disk(str(path))
        else:
            # The absolute, already-existing path prevents an accidental Hub
            # dataset lookup while retaining support for local builders. Only
            # a small deterministic prefix is needed for the reservoir, so
            # stream large local corpora rather than materializing them.
            source = load_dataset(str(path), split="train", streaming=True)
    else:
        raise ValueError(f"`data_dir` is not a regular local path: {path}.")

    return _select_source_split(source, data_dir=path), path


def _build_background_reservoir(
    source: Any,
    *,
    tokenizer: Any,
    text_field: str,
    target_tokens: int,
    data_dir: Path,
) -> tuple[int, ...]:
    columns = getattr(source, "column_names", None)
    if columns is not None and text_field not in columns:
        raise ValueError(
            f"Dataset at {data_dir} does not contain text field "
            f"{text_field!r}; available fields: {list(columns)!r}."
        )

    separator_ids = _encode(
        tokenizer,
        "\n\n",
        description="background document separator",
    )
    reservoir: list[int] = []
    source_size: int | None = None
    if hasattr(source, "__len__") and hasattr(source, "__getitem__"):
        try:
            source_size = len(source)
        except TypeError:
            pass
    rows = (
        (source[row_index] for row_index in range(source_size))
        if source_size is not None
        else iter(source)
    )
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"Dataset row {row_index} at {data_dir} is not a mapping."
            )
        if text_field not in row:
            raise ValueError(
                f"Dataset row {row_index} at {data_dir} does not contain "
                f"text field {text_field!r}."
            )
        text = row[text_field]
        if not isinstance(text, str):
            raise TypeError(
                f"Dataset field {text_field!r} in row {row_index} must be "
                f"a string, received {type(text).__name__}."
            )
        text_ids = _encode(
            tokenizer,
            text,
            description=f"background row {row_index}",
        )
        if not text_ids:
            continue
        if reservoir:
            reservoir.extend(separator_ids)
        reservoir.extend(text_ids)
        if len(reservoir) >= target_tokens:
            break

    if not reservoir:
        raise ValueError(
            f"Dataset at {data_dir} has no tokenizable text in "
            f"field {text_field!r}."
        )

    base_reservoir = tuple(reservoir)
    if len(reservoir) < target_tokens:
        repeats = math.ceil(target_tokens / len(base_reservoir))
        reservoir = list((base_reservoir * repeats)[:target_tokens])
    elif len(reservoir) > target_tokens:
        reservoir = reservoir[:target_tokens]
    return tuple(reservoir)


class MultiPasskeyDataset(Dataset):
    """Generate fixed-length multi-passkey examples by global sample index."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        data_dir: str | os.PathLike[str],
        seq_len: int,
        gap: int,
        key_len: int,
        num_passkeys: int,
        num_samples: int,
        seed: int = 0,
        min_depth_ratio: float = 0.1,
        max_depth_ratio: float = 0.9,
        shift_labels: bool = False,
        data_start_index: int | None = None,
        text_field: str = "text",
        passkey_format: str = "digits",
    ) -> None:
        _require_int("seq_len", seq_len, 1)
        _require_int("gap", gap, 0)
        _require_int("key_len", key_len, 1)
        _require_int("num_passkeys", num_passkeys, 1)
        _require_int("num_samples", num_samples, 1)
        _require_int("seed", seed, 0)
        if not isinstance(shift_labels, bool):
            raise TypeError(
                "`shift_labels` must be a boolean, "
                f"received {shift_labels!r}."
            )
        if shift_labels:
            raise ValueError(
                "`shift_labels=True` is unsupported for multi-passkey data: "
                "distillation supervises each answer token's own hidden state."
            )

        if data_start_index is None:
            data_start_index = 0
        _require_int("data_start_index", data_start_index, 0)
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("`text_field` must be a non-empty string.")
        if passkey_format not in {"digits", "words"}:
            raise ValueError(
                "`passkey_format` must be either 'digits' or 'words'."
            )

        min_ratio = _require_depth_ratio(
            "min_depth_ratio",
            min_depth_ratio,
        )
        max_ratio = _require_depth_ratio(
            "max_depth_ratio",
            max_depth_ratio,
        )
        if min_ratio > max_ratio:
            raise ValueError(
                "`min_depth_ratio` must be <= `max_depth_ratio`, "
                f"received {min_ratio} > {max_ratio}."
            )

        if passkey_format == "digits":
            digit_token_ids = _resolve_digit_token_ids(tokenizer)
            passkey_units = tuple(
                ((token_id,), (token_id,))
                for token_id in digit_token_ids
            )
        else:
            digit_token_ids = ()
            passkey_units = _resolve_word_token_ids(tokenizer)
        remember_prefixes = tuple(
            _encode(
                tokenizer,
                REMEMBER_TEMPLATE.format(number=number),
                description=f"remember template {number}",
            )
            for number in range(1, num_passkeys + 1)
        )
        remember_suffix = _encode(
            tokenizer,
            REMEMBER_SUFFIX,
            description="remember suffix",
        )
        question_prefixes = tuple(
            _encode(
                tokenizer,
                QUESTION_TEMPLATE.format(number=number),
                description=f"question template {number}",
            )
            for number in range(1, num_passkeys + 1)
        )
        answer_suffix = _encode(
            tokenizer,
            ANSWER_SUFFIX,
            description="answer suffix",
        )

        template_token_count = sum(
            len(prefix) + len(remember_suffix)
            for prefix in remember_prefixes
        ) + sum(
            len(prefix) + len(answer_suffix)
            for prefix in question_prefixes
        )
        maximum_key_token_count = max(
            len(first) for first, _ in passkey_units
        ) + (key_len - 1) * max(
            len(continuation) for _, continuation in passkey_units
        )
        minimum_key_token_count = min(
            len(first) for first, _ in passkey_units
        ) + (key_len - 1) * min(
            len(continuation) for _, continuation in passkey_units
        )
        maximum_fixed_token_count = (
            template_token_count
            + 2 * num_passkeys * maximum_key_token_count
        )
        if seq_len - maximum_fixed_token_count < gap:
            minimum_seq_len = maximum_fixed_token_count + gap
            raise ValueError(
                "Multi-passkey layout does not fit: "
                f"seq_len={seq_len}, gap={gap}, key_len={key_len}, "
                f"num_passkeys={num_passkeys}. `seq_len` must be at least "
                f"{minimum_seq_len} for the templates, worst-case tokenized "
                "passkeys, and the required post-needle background."
            )

        if passkey_format == "digits":
            fixed_token_count = (
                template_token_count + 2 * num_passkeys * key_len
            )
            background_token_count = seq_len - fixed_token_count
            min_depth_position, max_depth_position = _depth_positions(
                background_token_count=background_token_count,
                gap=gap,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )
        else:
            fixed_token_count = None
            background_token_count = None
            min_depth_position = None
            max_depth_position = None

        source, local_path = _load_local_source(data_dir)
        maximum_background_token_count = (
            seq_len
            - template_token_count
            - 2 * num_passkeys * minimum_key_token_count
        )
        reservoir_target = max(
            1,
            maximum_background_token_count + seq_len,
        )
        background_reservoir = _build_background_reservoir(
            source,
            tokenizer=tokenizer,
            text_field=text_field,
            target_tokens=reservoir_target,
            data_dir=local_path,
        )

        # Validate the exact per-example RNG namespace eagerly.
        sample_rng(int(seed), int(data_start_index))

        self.seq_len = int(seq_len)
        self.gap = int(gap)
        self.key_len = int(key_len)
        self.num_passkeys = int(num_passkeys)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.data_start_index = int(data_start_index)
        self.min_depth_ratio = min_ratio
        self.max_depth_ratio = max_ratio
        self.text_field = text_field
        self.passkey_format = passkey_format
        self.passkey_units = passkey_units
        self.digit_token_ids = digit_token_ids
        self.remember_prefixes = remember_prefixes
        self.remember_suffix = remember_suffix
        self.question_prefixes = question_prefixes
        self.answer_suffix = answer_suffix
        self.fixed_token_count = fixed_token_count
        self.background_token_count = background_token_count
        self.min_depth_position = min_depth_position
        self.max_depth_position = max_depth_position
        self.background_reservoir = background_reservoir

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError(f"Dataset index must be an integer, got {index!r}.")
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        global_index = self.data_start_index + index
        rng = sample_rng(self.seed, global_index)

        unit_indices = rng.integers(
            0,
            len(self.passkey_units),
            size=(self.num_passkeys, self.key_len),
        )
        passkeys: tuple[tuple[int, ...], ...] = tuple(
            tuple(
                token_id
                for unit_position, unit_index in enumerate(row)
                for token_id in self.passkey_units[int(unit_index)][
                    0 if unit_position == 0 else 1
                ]
            )
            for row in unit_indices
        )
        template_token_count = sum(
            len(prefix) + len(self.remember_suffix)
            for prefix in self.remember_prefixes
        ) + sum(
            len(prefix) + len(self.answer_suffix)
            for prefix in self.question_prefixes
        )
        fixed_token_count = template_token_count + 2 * sum(
            len(passkey) for passkey in passkeys
        )
        background_token_count = self.seq_len - fixed_token_count
        if background_token_count < self.gap:
            raise RuntimeError(
                "Sampled passkeys exceeded the validated token budget: "
                f"fixed_tokens={fixed_token_count}, seq_len={self.seq_len}."
            )
        min_depth_position, max_depth_position = _depth_positions(
            background_token_count=background_token_count,
            gap=self.gap,
            min_ratio=self.min_depth_ratio,
            max_ratio=self.max_depth_ratio,
        )
        num_depth_positions = (
            max_depth_position - min_depth_position + 1
        )
        insertion_positions = sorted(
            int(position) + min_depth_position
            for position in rng.choice(
                num_depth_positions,
                size=self.num_passkeys,
                replace=num_depth_positions < self.num_passkeys,
            )
        )

        max_offset = (
            len(self.background_reservoir) - background_token_count
        )
        background_offset = int(rng.integers(0, max_offset + 1))
        background = self.background_reservoir[
            background_offset:
            background_offset + background_token_count
        ]

        input_ids: list[int] = []
        labels: list[int] = []

        def append(tokens: Sequence[int], *, supervise: bool = False) -> None:
            input_ids.extend(tokens)
            if supervise:
                labels.extend(tokens)
            else:
                labels.extend([IGNORE_INDEX] * len(tokens))

        background_cursor = 0
        for passkey_index, insertion_position in enumerate(
            insertion_positions
        ):
            append(background[background_cursor:insertion_position])
            append(self.remember_prefixes[passkey_index])
            append(passkeys[passkey_index])
            append(self.remember_suffix)
            background_cursor = insertion_position
        append(background[background_cursor:])

        for passkey_index, passkey in enumerate(passkeys):
            append(self.question_prefixes[passkey_index])
            append(passkey, supervise=True)
            append(self.answer_suffix)

        expected_targets = sum(len(passkey) for passkey in passkeys)
        target_count = sum(label != IGNORE_INDEX for label in labels)
        if len(input_ids) != self.seq_len or len(labels) != self.seq_len:
            raise RuntimeError(
                "Internal multi-passkey layout error: "
                f"expected length {self.seq_len}, got "
                f"input_ids={len(input_ids)}, labels={len(labels)}."
            )
        if target_count != expected_targets:
            raise RuntimeError(
                "Internal multi-passkey label error: "
                f"expected {expected_targets} answer targets, "
                f"got {target_count}."
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataset(
    tokenizer: Any,
    *,
    data_dir: str | os.PathLike[str],
    seq_len: int,
    gap: int,
    key_len: int,
    num_passkeys: int,
    num_samples: int,
    seed: int = 0,
    min_depth_ratio: float = 0.1,
    max_depth_ratio: float = 0.9,
    shift_labels: bool = False,
    data_start_index: int | None = None,
    text_field: str = "text",
    passkey_format: str = "digits",
    **kwargs: Any,
) -> MultiPasskeyDataset:
    return MultiPasskeyDataset(
        tokenizer,
        data_dir=data_dir,
        seq_len=seq_len,
        gap=gap,
        key_len=key_len,
        num_passkeys=num_passkeys,
        num_samples=num_samples,
        seed=seed,
        min_depth_ratio=min_depth_ratio,
        max_depth_ratio=max_depth_ratio,
        shift_labels=shift_labels,
        data_start_index=data_start_index,
        text_field=text_field,
        passkey_format=passkey_format,
    )
