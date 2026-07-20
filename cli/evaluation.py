import argparse

from lib.evaluation import evaluate

def main():
    parser = argparse.ArgumentParser(description="Search for precision")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (K for precision)")

    args = parser.parse_args()

    limit = args.limit
    evaluate(limit)


if __name__ == "__main__":
    main()