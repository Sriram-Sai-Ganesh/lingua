#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Helper script to run text generation using a trained checkpoint.

Usage:
    # Interactive mode (prompts from stdin):
    python generate.py --ckpt /scratch/jeisner1/ssaigan1/lingua/dump_dir/consolidated

    # Single prompt mode:
    python generate.py --ckpt /scratch/jeisner1/ssaigan1/lingua/dump_dir/consolidated --prompt "Once upon a time"

    # With generation parameters:
    python generate.py --ckpt /scratch/jeisner1/ssaigan1/lingua/dump_dir/checkpoints/0000005200/consolidated \
        --prompt "The thirsty elephant" \
        --max_gen_len 1024 \
        --temperature 0.7 \
        --top_p 0.9
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Lingua model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode:
    python generate.py --ckpt checkpoints/0000005200/consolidated

    # Single prompt:
    python generate.py --ckpt checkpoints/0000005200/consolidated --prompt "The meaning of life is"

    # Multiple prompts from file:
    cat prompts.txt | python generate.py --ckpt checkpoints/0000005200/consolidated --interactive
        """,
    )

    # Required arguments
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the consolidated checkpoint directory (containing consolidated.pth and params.json)",
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to generate from. If not provided, enters interactive mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode, accepting prompts from stdin.",
    )

    # Generation parameters
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. 0.0 for greedy decoding (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold (default: None)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling threshold (default: None)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens in context (default: 2048)",
    )

    # Output options
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Show generation statistics (tokens/sec, etc.)",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming output (print all at once)",
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run generation on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "bf16"],
        help="Data type for model (default: bf16)",
    )

    return parser.parse_args()


def validate_checkpoint_path(ckpt_path: str) -> Path:
    """Validate that the checkpoint path exists and contains required files."""
    ckpt_dir = Path(ckpt_path)

    if not ckpt_dir.exists():
        print(
            f"Error: Checkpoint directory does not exist: {ckpt_dir}", file=sys.stderr
        )
        sys.exit(1)

    required_files = ["consolidated.pth", "params.json"]
    for f in required_files:
        if not (ckpt_dir / f).exists():
            print(
                f"Error: Required file '{f}' not found in {ckpt_dir}", file=sys.stderr
            )
            sys.exit(1)

    return ckpt_dir


def load_model(ckpt_path: str, device: str = "cuda"):
    """Load the model and tokenizer from a checkpoint."""
    print(f"Loading model from {ckpt_path}...", file=sys.stderr)

    model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt_path)

    if device == "cpu":
        model = model.cpu()

    print("Model loaded successfully!", file=sys.stderr)
    return model, tokenizer, config


def generate_text(generator, prompts: list, show_stats: bool = False):
    """Generate text from a list of prompts and return results."""
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    results = []
    for i, (prompt, gen) in enumerate(zip(prompts, generation)):
        results.append(
            {
                "prompt": prompt,
                "generated": gen,
                "loglikelihood": loglikelihood[i] if loglikelihood else None,
            }
        )

    stats = None
    if show_stats:
        total_tokens = sum(len(gen) for gen in generation)
        elapsed_time = end_time - start_time
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        stats = {
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time,
            "tokens_per_second": tokens_per_sec,
        }

    return results, stats


def print_generation(prompt: str, generated: str, stats: dict = None):
    """Print the generated text in a formatted way."""
    print("\n" + "=" * 60)
    print(f"Prompt: {prompt}")
    print("-" * 60)
    print(f"Generated:\n{generated}")
    print("=" * 60)

    if stats:
        print(
            f"\nStats: {stats['total_tokens']} tokens in {stats['elapsed_time']:.2f}s "
            f"({stats['tokens_per_second']:.2f} tokens/sec)"
        )


def interactive_mode(generator, show_stats: bool = False):
    """Run the generator in interactive mode."""
    print(
        "\nInteractive mode. Enter prompts (Ctrl+D or empty line to exit):\n",
        file=sys.stderr,
    )

    while True:
        try:
            prompt = input(">>> ")
            if not prompt.strip():
                continue

            results, stats = generate_text(generator, [prompt], show_stats=show_stats)
            print_generation(prompt, results[0]["generated"], stats)
            print()

        except EOFError:
            print("\nExiting...", file=sys.stderr)
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...", file=sys.stderr)
            break


def main():
    args = parse_args()

    # Validate checkpoint path
    ckpt_path = validate_checkpoint_path(args.ckpt)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    # Load model
    model, tokenizer, config = load_model(str(ckpt_path), args.device)

    # Create generator configuration
    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_gen_len=args.max_gen_len,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        device=args.device,
    )

    # Create generator
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Run generation
    if args.prompt:
        # Single prompt mode
        results, stats = generate_text(
            generator, [args.prompt], show_stats=args.show_stats
        )
        print_generation(args.prompt, results[0]["generated"], stats)
    elif args.interactive or sys.stdin.isatty():
        # Interactive mode
        interactive_mode(generator, show_stats=args.show_stats)
    else:
        # Read prompts from stdin (pipe mode)
        prompts = [line.strip() for line in sys.stdin if line.strip()]
        if prompts:
            results, stats = generate_text(
                generator, prompts, show_stats=args.show_stats
            )
            for result in results:
                print_generation(result["prompt"], result["generated"], stats)


if __name__ == "__main__":
    main()
