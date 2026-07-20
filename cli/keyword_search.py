#!/usr/bin/env python3

import argparse
from lib.search_utils import BM25_K1, BM25_B, BM25_LIMIT
from lib.keyword_search import (search_command, 
                                build_command, 
                                tf_command, 
                                idf_command, 
                                tf_idf_command, 
                                bmidf25_command, 
                                bm25_tf_command, 
                                bm25_search_command,
                                bm25_command)


def movie_search() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search for a keyword")
    search_parser.add_argument("query", type=str, help="The keyword to search for")

    search_parser = subparsers.add_parser("build", help="Build Cache")

    search_parser = subparsers.add_parser("tf", help="Term Frequencies the inverted index")
    search_parser.add_argument("doc_id", type=int, help="Search Doc Id")
    search_parser.add_argument("term", type=str, help="Search term")


    search_parser = subparsers.add_parser("idf", help="Calcuate Inverse Document Frequency")
    search_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("tf-idf", help="Term Frequencies the inverted index")
    search_parser.add_argument("doc_id", type=int, help="Search Doc Id")
    search_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("bmidf", help="Calcuate Inverse Document Frequency With BM25")
    search_parser.add_argument("term", type=str, help="Search term")


    search_parser = subparsers.add_parser("bmtf", help="Term Frequencies the inverted index")
    search_parser.add_argument("doc_id", type=int, help="Search Doc Id")
    search_parser.add_argument("term", type=str, help="Search term")
    search_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="K1 value default 1.5")
    search_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="b value default 0.75")

    search_parser = subparsers.add_parser("bm25", help="BM25 for Term")
    search_parser.add_argument("doc_id", type=int, help="Search Doc Id")
    search_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("bm25_search", help="BM25 Search for Term")
    search_parser.add_argument("term", type=str, help="Search term")
    search_parser.add_argument("limit", type=int, nargs='?', default=BM25_LIMIT, help="limit to 5")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Performing keyword search for: {args.query}")
            results = search_command(args.query, n_results=5)
            for i,result in enumerate(results):
                print(f"{i}: {result['title']}")
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tf-idf":
            tf_idf_command(args.doc_id, args.term)
        case "bmidf":
            bmidf25_command(args.term)
        case "bmtf":
            bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
        case "bm25":
            bm25_search_command(args.term, args.limit)
        case "bm25_search":
            bm25_search_command(args.term, args.limit)
        case "bm25":
            bm25_command(args.doc_id, args.term)
        case _:
            parser.print_help()

    # Here you would implement the logic to perform the keyword search
    if hasattr(args, 'query'):
        print(f"Searching for keyword: {args.query}")

if __name__ == "__main__":
    movie_search()    