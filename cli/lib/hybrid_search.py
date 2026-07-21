import os

from lib.search_utils import load_movies
from lib.llm import correct_spelling, rewrite_query, llm_judge
from lib.rerank import individual_rerank, batch_rerank, cross_encoder_rerank
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self,documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query, alpha, limit = 5):
        bm25_result = self._bm25_search(query, limit*500)
        sem_result = self.semantic_search.search_chunks(query, limit*500)

        combined_results = combine_search_results(bm25_result, sem_result, alpha)

        return combined_results[:limit]

    def rrf_search(self, query, k, limit = 10):
        bm25_result = self._bm25_search(query, limit*500)
        sem_result = self.semantic_search.search_chunks(query, limit*500)

        combined_results = rff_combine_search_results(bm25_result, sem_result, k)

        return combined_results[:limit]

def hybrid_score(bm25_score, sem_score, alpha = 0.5):
    return (alpha * bm25_score) + ((1-alpha)*sem_score)

def normalize_search_results(results):
    scores = [r['score'] for r in results]

    norm_scores = normalized_scores(scores)

    for idx, result in enumerate(results):
        result['normalized_score'] = norm_scores[idx]
    return results

def combine_search_results(bm25_results, sem_results, alpha):
    bm25_norm = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results(sem_results)

    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm['doc_id']
        combined_norm[doc_id] = {
            'doc_id': doc_id,
            'bm25_score': norm['normalized_score'],
            'sem_score': 0.,
            'title': norm['title'],
            'description': norm['description']
        }
    for norm in sem_norm:
        doc_id = norm['id']
        if doc_id not in combined_norm:
            combined_norm[doc_id]= {
                'doc_id': doc_id,
                'bm25_score': 0.,
                'sem_score': 0.,
                'title': norm['title'],
                'description': norm['description']                
            }
        combined_norm[doc_id]['sem_score']=norm['normalized_score']

    for k,v in combined_norm.items():
        combined_norm[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['sem_score'], alpha)

    results = sorted(list(combined_norm.values()), key=lambda x: x['hybrid_score'], reverse=True)
    return results


def rrf_score(rank,k):
    return 1/(rank +k)

def rrf_final_score(r1, r2, k):
    if r1 and r2:
        return rrf_score(r1, k) + rrf_score(r2, k)
    return 0.

def rff_combine_search_results(bm25_results, sem_results, k):
    scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result['doc_id']

        scores[doc_id]= {
                'doc_id': doc_id,
                'bm25_rank': rank,
                'bm25_score': rrf_score(rank, k),
                'sem_rank': None,
                'sem_score': None,
                'title': result['title'],
                'description': result['description']                  
        }

    for rank, result in enumerate(sem_results, start=1):
        doc_id=result['id']
        if doc_id not in scores:
            scores[doc_id] = {
                'doc_id': doc_id,
                'bm25_rank': None,
                'bm25_score': None,
                'sem_rank': None,
                'sem_score': None,
                'title': result['title'],
                'description': result['description']                     
            }
        scores[doc_id]['sem_rank'] = rank
        scores[doc_id]['sem_score'] = rrf_score(rank, k)
    for doc_id in scores.keys():
        scores[doc_id]['rrf_score'] = rrf_final_score(
            scores[doc_id]['bm25_rank'],
            scores[doc_id]['sem_rank'],
            k
        )
    results = sorted(list(scores.values()), key=lambda x: x['rrf_score'], reverse=True)
    return results


def normalized_scores(scores):
    if not scores: return []
    min_score=min(scores)
    max_score=max(scores)

    if min_score == max_score:
        return [1.]*len(scores)
    
    score_range = max_score - min_score

    return [(score - min_score)/score_range for score in scores]

def weighted_search(query, alpha = 0.5, limit = 5):
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.weighted_search(query, alpha, limit)
    for idx,result in enumerate(results[:limit]):
        print(f"{idx+1} {result['title']}")
        print(f"Hybrid score: {result['hybrid_score']}")
        print(f"BM25 score: {result['bm25_score']} and Semantic score: {result['sem_score']}")
        print(f"Description: {result['description'][:100]}")
    

def rrf_search(query, k = 60, limit = 5, enhance=None, rerank=None, evaluate=None):
    movies = load_movies()
    hs = HybridSearch(movies)
    match enhance:
        case "spell":
            new_query = correct_spelling(query)
            print(f"Enhanced query (spell): '{query}' -> '{new_query}'\n")
            query = new_query
        case "rewrite":
            new_query = rewrite_query(query)
            print(f"Rewritten query (rewrite): '{query}' -> '{new_query}'\n")
            query = new_query
    
    rrf_limit = limit * 5 if rerank else 5

    results = hs.rrf_search(query, k, rrf_limit)

    print(f"results of rrf search - {len(results)}")


    match rerank:
        case "individual":
            print("Started Rerank")
            results = individual_rerank(query, results)
            print(f"Reranking top {limit} results using Individual Method")
        case "batch":
            print("Started Rerank")
            results = batch_rerank(query, results)
            print(f"Reranking top {limit} results using Batch Method")
        case "cross_encoder":
            print("Started Rerank")
            results = cross_encoder_rerank(query, results)
            print(f"Reranking top {limit} results using Cross Encoder Method")
        case _:
            pass
    if evaluate: formatted_results = []

    for idx,result in enumerate(results[:limit], start=1):
        print(f"{idx} {result['title']}")
        print(f"RRF score: {result['rrf_score']}")
        print(f"BM25 rank: {result['bm25_rank']} and Semantic rank: {result['sem_rank']}")
        # print(f"LLM Rerank: {result['rerank_response']}")
        print(f"Description: {result['description'][:100]}")
        print('-'*25)
        if evaluate:
            formatted_results.append(f"<result id={idx}>{result['title']}: {result['description'][:100]}")
    if evaluate:
        llm_results = llm_judge(query, "\n".join(formatted_results))
        for idx, r in enumerate(results[:limit], start=1):
            print(f"{idx} {r['title']}: {llm_results[idx-1]}/3")