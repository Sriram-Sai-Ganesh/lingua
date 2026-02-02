"""
Convert tokenized .npy shards into JSONL text files for lingua training.

Usage:
    python setup/convert_npy_to_jsonl.py /path/to/npy/dir --nchunks 1

The script assumes each .npy file is a flat uint16 token buffer with documents
separated by the tokenizer's end-of-text token. Training documents are written
to {dataset_name}.chunk.00.jsonl, validation documents to {dataset_name}.val.jsonl,
and test documents to {dataset_name}.test.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer
from tqdm import tqdm

DEFAULT_EOS_TOKEN_ID = 50279


def _choose_tokenizer_file(repo_files: list[str], tokenizer_hint: str) -> str | None:
    tokenizer_files = [path for path in repo_files if path.endswith("tokenizer.json")]
    if not tokenizer_files:
        return None

    def score(path: str) -> tuple[bool, bool, int, int, str]:
        return (
            tokenizer_hint not in path,
            Path(path).name != "tokenizer.json",
            path.count("/"),
            len(path),
            path,
        )

    return min(tokenizer_files, key=score)


def _load_tokenizer(
    tokenizer_cache_dir: Path,
    tokenizer_hint: str = "gpt-neox-olmo-dolma-v1_5",
) -> Tokenizer:
    tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)

    search_order = [
        ("allenai/DataDecide-data-recipes", "dataset"),
        (f"allenai/{tokenizer_hint}", "model"),
    ]

    for repo_id, repo_type in search_order:
        try:
            repo_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
        except Exception as exc:
            print(f"Could not list files in {repo_id}: {exc}")
            continue

        tokenizer_file = _choose_tokenizer_file(repo_files, tokenizer_hint)
        if tokenizer_file is None:
            continue

        print(f"Downloading tokenizer from {repo_id}:{tokenizer_file}")
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=tokenizer_file,
            local_dir=str(tokenizer_cache_dir),
            local_dir_use_symlinks=False,
        )
        return Tokenizer.from_file(tokenizer_path)

    raise FileNotFoundError(
        "Could not find a tokenizer.json in allenai/DataDecide-data-recipes or "
        f"allenai/{tokenizer_hint}."
    )


def _stable_unit_interval(dataset_name: str, doc_index: int, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{dataset_name}:{seed}:{doc_index}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _assign_split(
    dataset_name: str,
    doc_index: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> str:
    bucket = _stable_unit_interval(dataset_name, doc_index, seed)
    if bucket < val_ratio:
        return "val"
    if bucket < val_ratio + test_ratio:
        return "test"
    return "train"


def _iter_documents(tokens: np.ndarray, eos_token_id: int) -> Iterable[np.ndarray]:
    start = 0
    for stop in np.flatnonzero(tokens == eos_token_id):
        stop = int(stop)
        if stop > start:
            yield tokens[start:stop]
        start = stop + 1

    if start < len(tokens):
        yield tokens[start:]


def _write_batch(
    tokenizer: Tokenizer,
    batch_token_ids: list[list[int]],
    batch_splits: list[str],
    train_writers,
    split_writers,
    counts: dict[str, int],
    train_written: int,
) -> int:
    texts = tokenizer.decode_batch(batch_token_ids, skip_special_tokens=True)

    for text, split in zip(texts, batch_splits):
        if not text or not text.strip():
            continue

        if split == "train":
            handle = train_writers[train_written % len(train_writers)]
            train_written += 1
        else:
            handle = split_writers[split]

        handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        counts[split] += 1

    return train_written


def convert_npy_to_jsonl(
    input_dir: str,
    output_dir: str | None = None,
    nchunks: int = 1,
    val_ratio: float = 0.005,
    test_ratio: float = 0.005,
    batch_size: int = 512,
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID,
    seed: int = 0,
    tokenizer_hint: str = "gpt-neox-olmo-dolma-v1_5",
    overwrite: bool = False,
):
    if nchunks < 1:
        raise ValueError("nchunks must be at least 1")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1")

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "jsonl"
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_name = input_path.name
    npy_files = sorted(input_path.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    tokenizer_cache_dir = output_path / "_tokenizer_cache"
    tokenizer = _load_tokenizer(
        tokenizer_cache_dir=tokenizer_cache_dir,
        tokenizer_hint=tokenizer_hint,
    )
    eos_token_id = tokenizer.token_to_id("<|endoftext|>") or eos_token_id
    print(f"Using eos_token_id={eos_token_id}")

    train_paths = [
        output_path / f"{dataset_name}.chunk.{chunk_idx:02d}.jsonl"
        for chunk_idx in range(nchunks)
    ]
    split_paths = {
        "val": output_path / f"{dataset_name}.val.jsonl",
        "test": output_path / f"{dataset_name}.test.jsonl",
    }

    all_output_paths = [*train_paths, *split_paths.values()]
    existing = [path for path in all_output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing files. Re-run with overwrite=True or "
            f"remove: {', '.join(str(path) for path in existing)}"
        )
    for path in existing:
        path.unlink()

    counts = {"train": 0, "val": 0, "test": 0}
    global_doc_index = 0
    train_written = 0

    with ExitStack() as stack:
        train_writers = [
            stack.enter_context(path.open("w", encoding="utf-8")) for path in train_paths
        ]
        split_writers = {
            split: stack.enter_context(path.open("w", encoding="utf-8"))
            for split, path in split_paths.items()
        }

        for npy_file in tqdm(npy_files, desc="Converting shards"):
            tokens = np.memmap(npy_file, dtype=np.uint16, mode="r")

            batch_token_ids: list[list[int]] = []
            batch_splits: list[str] = []
            for doc_tokens in _iter_documents(tokens, eos_token_id):
                split = _assign_split(
                    dataset_name=dataset_name,
                    doc_index=global_doc_index,
                    seed=seed,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
                global_doc_index += 1

                batch_token_ids.append(doc_tokens.tolist())
                batch_splits.append(split)

                if len(batch_token_ids) >= batch_size:
                    train_written = _write_batch(
                        tokenizer=tokenizer,
                        batch_token_ids=batch_token_ids,
                        batch_splits=batch_splits,
                        train_writers=train_writers,
                        split_writers=split_writers,
                        counts=counts,
                        train_written=train_written,
                    )
                    batch_token_ids = []
                    batch_splits = []

            if batch_token_ids:
                train_written = _write_batch(
                    tokenizer=tokenizer,
                    batch_token_ids=batch_token_ids,
                    batch_splits=batch_splits,
                    train_writers=train_writers,
                    split_writers=split_writers,
                    counts=counts,
                    train_written=train_written,
                )

    print("Conversion complete.")
    print(f"train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}")
    for path in all_output_paths:
        print(path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert tokenized .npy shards into JSONL text files"
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .npy shards")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where JSONL files will be written (default: input_dir/jsonl)",
    )
    parser.add_argument(
        "--nchunks",
        type=int,
        default=1,
        help="Number of training JSONL chunks to write (default: 1)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.005,
        help="Fraction of documents assigned to validation (default: 0.005)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.005,
        help="Fraction of documents assigned to test (default: 0.005)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of documents to decode per batch (default: 512)",
    )
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=DEFAULT_EOS_TOKEN_ID,
        help="Fallback end-of-text token id used to split documents",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic train/val/test assignment",
    )
    parser.add_argument(
        "--tokenizer-hint",
        type=str,
        default="gpt-neox-olmo-dolma-v1_5",
        help="Tokenizer repo suffix to fall back to if the dataset repo has no tokenizer",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing JSONL outputs",
    )
    args = parser.parse_args()

    convert_npy_to_jsonl(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        nchunks=args.nchunks,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        eos_token_id=args.eos_token_id,
        seed=args.seed,
        tokenizer_hint=args.tokenizer_hint,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
