#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, search, chunk_text, chunk_text_semantic, embed_chunks, embed_chunk_search
from lib.search_utils import LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help ="Verify Embedding Model")

    embed_parser = subparsers.add_parser("embed_text", help="Encode Text with embedding model")
    embed_parser.add_argument("text", type=str, help="Text to be encoded")

    verify_parser = subparsers.add_parser("verify_embed", help="Verify embedded model")

    search_parser = subparsers.add_parser("search", help="Search query in embeddings using model")
    search_parser.add_argument("query", type=str, help="Text to be encoded")
    search_parser.add_argument("--limit", type=int, nargs='?', default=LIMIT, help="Limits in response")

    fixed_chunk_parser = subparsers.add_parser("fixed-chunk", help="Search query for chunking using fixed size")
    fixed_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    fixed_chunk_parser.add_argument("--chunk_size", type=int, nargs='?', default=200, help="Limits in response")
    fixed_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Limits in response")

    semantic_chunk_parser = subparsers.add_parser("semantic-chunk", help="Search query for chunking using semantic size")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk_parser.add_argument("--max_chunk_size", type=int, nargs='?', default=4, help="Limits in chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Limits in response")

    embed_chunk_parser = subparsers.add_parser("embed-chunks", help="Embed the semantic chunks")

    search_embed_parser = subparsers.add_parser("search-chunks", help="Search query in embeddings")
    search_embed_parser.add_argument("query", type=str, help="Text to be encoded")
    search_embed_parser.add_argument("--limit", type=int, nargs='?', default=LIMIT, help="Limits in response")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embed":
            verify_embeddings()
        case "search":
            search(args.query, args.limit)
        case "fixed-chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic-chunk":
            chunk_text_semantic(args.text, args.max_chunk_size, args.overlap)
        case "embed-chunks":
            embed_chunks()
        case "search-chunks":
            embed_chunk_search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()