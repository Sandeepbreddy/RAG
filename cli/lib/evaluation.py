
import json

from lib.search_utils import PROJECT_ROOT, load_movies
from lib.hybrid_search import HybridSearch


def load_test_cases():
    with open(PROJECT_ROOT/'data'/'golden_dataset.json') as f:
        test_cases = json.load(f)['test_cases']
    return test_cases

def evaluate(limit):
    print(f"k={limit}")
    test_cases = load_test_cases()

    movies = load_movies()

    hs = HybridSearch(movies)

    for test in test_cases:
        query = test['query']
        exp = test['relevant_docs']


        rrf_results = hs.rrf_search(query, k=60, limit = limit)

        print(len(rrf_results))

        relevant_count = 0

        for result in rrf_results:
            relevant_count += result['title'] in exp
        precision = relevant_count/limit
        retrieved = ", ".join([r['title'] for r in rrf_results])

        recall = relevant_count/len(exp)
        f1 = (2 * ((precision * recall) / (precision + recall))) if (precision + recall) > 0 else 0.0

        print(f"Query: {query}")
        print(f"- Precision@{limit}: {precision:.4f}")
        print(f"- Recall@{limit}: {recall:.4f}")
        print(f"- F1@{limit}: {f1:.4f}")


        print(f"- Retrieved: {retrieved}")
        print(f"- Relevant: {', '.join(exp)}")






         