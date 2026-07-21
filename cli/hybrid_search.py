#!/usr/bin/env python3

import argparse

from lib.hybrid_search import normalized_scores, weighted_search, rrf_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normal_score_parser = subparsers.add_parser("normal_score", help = "Normalization of scores")
    normal_score_parser.add_argument("scores", type=float, nargs="+",help="list of scores")

    weighted_score_parser = subparsers.add_parser("weighted_search", help = "Weighted Search")
    weighted_score_parser.add_argument("query", type=str, help="Text to be searched")
    weighted_score_parser.add_argument("--alpha", type=float, nargs='?', default=0.5, help="wighted percentage")
    weighted_score_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit in response")

    rrf_score_parser = subparsers.add_parser("rrf_search", help = "RRF Search")
    rrf_score_parser.add_argument("query", type=str, help="Text to be searched")
    rrf_score_parser.add_argument("--k", type=float, nargs='?', default=60, help="wighted percentage")
    rrf_score_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit in response")
    rrf_score_parser.add_argument("--enhance", type=str, choices=["spell","rewrite"], help="enhance using LLM")
    rrf_score_parser.add_argument("--rerank", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank using LLM")
    rrf_score_parser.add_argument("--evaluate", action="store_true", help="Evaluate using LLM")


    args = parser.parse_args()

    match args.command:
        case "normal_score":
            norm_scores = normalized_scores(args.scores)
            for score in norm_scores:
                print(f"* {score: .4f}")
        case "weighted_search":
            results = weighted_search(args.query, args.alpha, args.limit)
        case "rrf_search":

            results = rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank, args.evaluate)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()