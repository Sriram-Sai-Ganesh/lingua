"""
Convert parquet files to JSONL format for lingua training.

Usage:
    python setup/convert_parquet_to_jsonl.py /path/to/parquet/dir [--nchunks 1]

The script converts all parquet files in the given directory to JSONL format.
Output files are named: {dataset_name}.chunk.00.jsonl, etc.
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def convert_parquet_to_jsonl(input_dir: str, nchunks: int = 1):
    """
    Convert parquet files in input_dir to JSONL format.

    Args:
        input_dir: Directory containing parquet files
        nchunks: Number of output chunks to split the data into
    """
    input_path = Path(input_dir)
    dataset_name = input_path.name

    # Find all parquet files
    parquet_files = list(input_path.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    # Process each parquet file (train, val, test)
    for pq_file in parquet_files:
        # Determine output filename based on the parquet file name
        # e.g., train-00000-of-00001.parquet -> wikitext-2-v1.chunk.00.jsonl
        file_prefix = pq_file.stem.split("-")[0]  # 'train', 'test', 'validation'

        # Read parquet file
        print(f"Reading {pq_file}...")
        table = pq.read_table(pq_file)

        # Get the text column
        if "text" in table.schema.names:
            text_col = "text"
        elif "content" in table.schema.names:
            text_col = "content"
        else:
            print(f"Warning: No 'text' or 'content' column found in {pq_file}")
            print(f"Available columns: {table.schema.names}")
            continue

        texts = table.column(text_col).to_pylist()

        # Filter out empty texts
        texts = [t for t in texts if t and t.strip()]

        print(f"  Found {len(texts)} non-empty texts")

        # For training data, split into chunks
        if file_prefix == "train":
            chunk_size = len(texts) // nchunks + 1
            for chunk_idx in range(nchunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(texts))
                chunk_texts = texts[start_idx:end_idx]

                if not chunk_texts:
                    continue

                output_file = input_path / f"{dataset_name}.chunk.{chunk_idx:02d}.jsonl"
                print(f"  Writing {len(chunk_texts)} texts to {output_file}")

                with open(output_file, "w") as f:
                    for text in chunk_texts:
                        f.write(json.dumps({"text": text}) + "\n")
        else:
            # For validation/test, create a single file with appropriate naming
            output_file = input_path / f"{dataset_name}.{file_prefix}.jsonl"
            print(f"  Writing {len(texts)} texts to {output_file}")

            with open(output_file, "w") as f:
                for text in texts:
                    f.write(json.dumps({"text": text}) + "\n")

    print("Conversion complete!")
    print(f"\nGenerated files in {input_dir}:")
    for f in sorted(input_path.glob("*.jsonl")):
        print(f"  {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files to JSONL format for lingua training"
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing parquet files to convert"
    )
    parser.add_argument(
        "--nchunks",
        type=int,
        default=1,
        help="Number of chunks to split training data into (default: 1)",
    )

    args = parser.parse_args()
    convert_parquet_to_jsonl(args.input_dir, args.nchunks)


if __name__ == "__main__":
    main()
"""
Sample usage:
python setup/convert_parquet_to_jsonl.py /scratch/jeisner1/ssaigan1/datasets/wikitext/wikitext-103-v1 --nchunks 1

"""
