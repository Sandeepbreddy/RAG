import argparse

from lib.rag import rag

def main():
    parser = argparse.ArgumentParser(description="SRAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("rag_search", help="RAG using OPENAI LLM")
    search_parser.add_argument("query", type=str, help="Text to be encoded")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit in response")
    search_parser.add_argument("--type", type=str, choices=["movieinfo", "summarize", "citation"], help="Rerank using LLM")

    args = parser.parse_args()

    match args.command:
        case "rag_search":
            rag(args.query, args.type)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()