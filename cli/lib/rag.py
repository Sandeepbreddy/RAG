from lib.llm import llm_answer_question, llm_movie_summarization, llm_movie_summary_citation
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def rag(query, type):
    match type:
        case "movieinfo":
            print("Movie Info Results")
            movies = load_movies()
            movie_info_from_llm(query,movies)
        case "summarize":
            print("Summarize Results")
            movies = load_movies()
            summary_from_llm(query,movies)
        case "citation":
            print("Summary with citation Results")
            movies = load_movies()
            summary_with_citation_from_llm(query,movies)
        case _:
            pass

def movie_info_from_llm(query, documents):
    hs = HybridSearch(documents)
    rrf_results = hs.rrf_search( query, k=60, limit=5)
    print("Search Results: ")
    for res in rrf_results:
        print(f"- {res['title']}")
    rag_results = llm_answer_question(query, rrf_results)
    print("RAG results")
    print(rag_results)

def summary_from_llm(query, documents, limit=5):
    hs = HybridSearch(documents)
    rrf_results = hs.rrf_search( query, k=60, limit=limit)
    print("Search Results: ")
    for res in rrf_results:
        print(f"- {res['title']}")
    rag_results = llm_movie_summarization(query, rrf_results)
    print("LLM Summary")
    print(rag_results)

def summary_with_citation_from_llm(query, documents, limit=5):
    hs = HybridSearch(documents)
    rrf_results = hs.rrf_search( query, k=60, limit=limit)
    print("Search Results: ")
    for res in rrf_results:
        print(f"- {res['title']}")
    rag_results = llm_movie_summary_citation(query, rrf_results)
    print("LLM Summary With Citation")
    print(rag_results)