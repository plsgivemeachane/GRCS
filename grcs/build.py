import argparse
import sys
from grcs.builder import run_builder

def main():
    parser = argparse.ArgumentParser(
        prog="grcs.build",
        description="GRCS Build — Analyze labeled data and build a .grcs map",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to labeled data JSONL"
    )
    parser.add_argument(
        "--output", type=str, default="maps/v1.grcs", help="Output .grcs map path"
    )
    parser.add_argument(
        "--k", type=int, default=50, help="Maximum number of centroids to store"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Contrastive scoring alpha (default: 0.1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/embedding-gemma-300m",
        help="Embedding model from HF",
    )
    parser.add_argument(
        "--type", type=str, default="html", help="Expected content type (default: html)"
    )

    args = parser.parse_args()
    
    run_builder(
        input_path=args.data,
        output_path=args.output,
        model_name=args.model,
        n_clusters=args.k,
        alpha=args.alpha,
        expected_type=args.type,
    )
    print(f"Success! Map saved to {args.output}")


if __name__ == "__main__":
    main()
