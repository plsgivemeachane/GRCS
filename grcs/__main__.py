"""
GRCS v1.0 — CLI Entry Point

Usage:
    py main.py generate [--prompts prompts.json] [--k 3] [--output data/samples.jsonl] [--backend lmstudio]
    py main.py judge [--port 5000]
"""

import argparse
import json
import sys
from pathlib import Path

from .generator import LMStudioGenerator, OpenAIGenerator, run_generation
from .judge_ui import run_judge
from .builder import run_builder
from .engine import run_engine_inference

# Default prompt: one imaginary product landing page
DEFAULT_PROMPTS = [
    (
        "Build a complete landing page for 'Lumora' — an AI-powered ambient lighting "
        "system that syncs with your music, movies, and mood. The page should include: "
        "a hero section with a bold headline and CTA button, a features grid (3 cards), "
        "a pricing section with 3 tiers, and a footer. Use a dark theme with subtle "
        "glowing accent colors (soft purple, warm amber). Make it feel premium and futuristic."
    )
]


def cmd_generate(args):
    prompts_path = args.prompts
    if prompts_path and Path(prompts_path).exists():
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if isinstance(prompts, str):
            prompts = [prompts]
        print(f"[*] Loaded {len(prompts)} prompt(s) from {prompts_path}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"[*] Using {len(prompts)} default prompt(s)")

    if args.backend == "lmstudio":
        generator = LMStudioGenerator(
            model=args.model, 
            base_url=args.base_url,
            max_workers=args.workers
        )
    else:
        generator = OpenAIGenerator(
            model=args.model,
            max_workers=args.workers
        )

    output = run_generation(
        prompts=prompts,
        generator=generator,
        output_path=args.output,
        k=args.k,
        expected_type=args.type,
    )
    print(f"\nNext step: py main.py judge --port 5000")


def cmd_judge(args):
    run_judge(host=args.host, port=args.port, debug=args.debug)


def cmd_build(args):
    run_builder(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        n_clusters=50,
        alpha=args.alpha,
        expected_type=args.type,
    )
    print("Success!")


def cmd_run(args):
    run_engine_inference(
        map_path=args.map,
        prompt=args.prompt,
        k=args.k,
        alpha=args.alpha,
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        workers=args.workers,
        expected_type=args.type,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="grcs",
        description="GRCS v1.0 — Group Relative Completion Selection",
    )
    sub = parser.add_subparsers(dest="command")

    # -- generate --
    gen = sub.add_parser("generate", help="Generate UI completions from prompts")
    gen.add_argument(
        "--prompts", type=str, default=None, help="Path to JSON file with prompt list"
    )
    gen.add_argument(
        "--k", type=int, default=3, help="Completions per prompt (default: 3)"
    )
    gen.add_argument(
        "--output", type=str, default="data/samples.jsonl", help="Output JSONL path"
    )
    gen.add_argument(
        "--backend", type=str, default="lmstudio", choices=["lmstudio", "openai"]
    )
    gen.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio API base URL",
    )
    gen.add_argument(
        "--type", type=str, default="html", help="Expected content type (default: html)"
    )
    gen.add_argument(
        "--model", type=str, default="local-model", help="Model name for LM Studio"
    )
    gen.add_argument(
        "--workers", type=int, default=4, help="Max concurrency workers (default: 4)"
    )

    # -- judge --
    jud = sub.add_parser("judge", help="Launch the labeling UI")
    jud.add_argument("--host", type=str, default="127.0.0.1")
    jud.add_argument("--port", type=int, default=5000)
    jud.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    # -- build --
    bld = sub.add_parser("build", help="Analyze labeled data and build a .grcs map")
    bld.add_argument(
        "--input", type=str, default="data/samples.jsonl", help="Path to labeled data"
    )
    bld.add_argument(
        "--output", type=str, default="maps/v1.grcs", help="Output .grcs map path"
    )
    bld.add_argument(
        "--k", type=int, default=10, help="Maximum number of centroids to store"
    )
    bld.add_argument(
        "--alpha", type=float, default=0.1, help="Contrastive scoring alpha (default: 0.1)"
    )
    bld.add_argument(
        "--model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Embedding model from HF",
    )
    bld.add_argument(
        "--type", type=str, default="html", help="Expected content type (default: html)"
    )

    # -- run --
    run = sub.add_parser("run", help="Run steered inference using a GRCS map")
    run.add_argument(
        "--map", type=str, default="maps/v1.grcs", help="Path to GRCS map"
    )
    run.add_argument(
        "--prompt", type=str, required=True, help="User prompt to generate UI"
    )
    run.add_argument("--k", type=int, default=3, help="Batch size k (default: 3)")
    run.add_argument(
        "--alpha", type=float, default=None, help="Contrastive scoring alpha override"
    )
    run.add_argument(
        "--backend", type=str, default="lmstudio", choices=["lmstudio", "openai"]
    )
    run.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio API base URL",
    )
    run.add_argument(
        "--type", type=str, default="html", help="Expected content type (default: html)"
    )
    run.add_argument(
        "--model", type=str, default="qwen3.5-4b", help="Model name for LM Studio"
    )
    run.add_argument(
        "--workers", type=int, default=4, help="Max concurrency workers (default: 4)"
    )

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "judge":
        cmd_judge(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
