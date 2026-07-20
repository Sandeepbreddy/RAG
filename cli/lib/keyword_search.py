import math
import pickle

from lib.search_utils import CACHE_PATH, load_movies, load_stopwords, BM25_K1, BM25_B, BM25_LIMIT
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import os
import string


stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index =defaultdict(set)
        self.term_frequencies = defaultdict(Counter)
        self.docmap = {} #map document id

        self.doc_lengths = {}

        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH/'term_frequencies.pkl'

        self.doc_length_path = CACHE_PATH/'doc_lengths.pkl'

    def __add_documents(self, doc_id, text):
        tokens = tokenized_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        lenghts = list(self.doc_lengths.values())
        if len(lenghts) == 0:
            return 0.0
        total = 0
        for l in lenghts:
            total +=l
        return total / len(lenghts)

    def get_documents(self, term):
        return sorted(list(self.index[term]))
    
    def get_term_frequencies(self, doc_id, term):
        token = tokenized_text(term)
        if len(token) != 1:
            raise ValueError("can only have 1 tokens")
        return self.term_frequencies[doc_id][token[0]]

    def get_idf(self, term):
        token = tokenized_text(term)
        if len(token) !=1:
            raise ValueError("Can only have 1 token")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])


        return math.log((doc_count +1)/(term_doc_count +1))
    
    def get_tf_idf(self, doc_id, term):
        tf = self.get_term_frequencies(doc_id, term)
        print(tf)
        idf = self.get_idf(term)
        print(idf)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenized_text(term)
        if len(token) != 1:
            raise ValueError("can only have 1 tokens")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5 ) / (term_doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1 = BM25_K1, b = BM25_B):
        tf = self.get_term_frequencies(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        return (tf * (k1 +1)) / (tf + k1 * length_norm)
    
    def get_bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf
    
    def bm25_search(self, query, limit = BM25_LIMIT):
        tokens = tokenized_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0
            for token in tokens:
                score += self.get_bm25(doc_id, token)
            scores[doc_id] = score
        

        sorted_scores = sorted(scores.items(),
                                key = lambda x: x[1],
                                reverse= True)
        
        results = sorted_scores[:limit]
        format_results = []
        for doc_id, score in results:
            title = self.docmap[doc_id]['title']
            description = self.docmap[doc_id]['description']

            format_results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "score": score,
                    "description": description
                }
            )
        return format_results
    
    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path,"rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_length_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_documents(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_length_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)
            

def search_command(query, n_results):
    movies  = load_movies()
    idx = InvertedIndex()
    idx.load()
    seen, res = set(), []
    query_tokens = tokenized_text(query)

    for query_token in query_tokens:
        matching_doc_ids =idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            matching_doc = idx.docmap[doc_id]
            res.append(matching_doc)

            if len(res) >= n_results:
                return res
    return res

    # for movie in movies:
    #     movie_tokens = tokenized_text(movie['title'])
    #     if has_matching_token(query_tokens, movie_tokens):
    #         res.append(movie)
    #     if len(res) == n_results:
    #         break
    # return res

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    
    print(idx.get_term_frequencies(doc_id, term))


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    return text

def tokenized_text(text):
    text = clean_text(text)
    res = []
    for tok in text.split():
        if filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)
    return res
    
def has_matching_token(query_tokens, movie_tokens):
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False

def filter(token):
    stopwords = load_stopwords()
    if token and token not in stopwords:
        return True
    return False



def build_command():
    docs = InvertedIndex()
    docs.build()
    docs.save()
    
    #removed for now, as it was just for testing
    # results = docs.get_documents("merida")
    # print(f"First Document for token 'merida' = {results[0]} ")

def idf_command(term):
    idx = InvertedIndex()
    idx.load()

    idf = idx.get_idf(term)

    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tf_idf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()

    tf_idf = idx.get_tf_idf(doc_id, term)

    print(f"TF-IDF score of '{term}' in document '{doc_id}' : {tf_idf:.2f}")

def bmidf25_command(term):
    idx = InvertedIndex()
    idx.load()

    bmidf = idx.get_bm25_idf(term)

    print(f"With BM25 Inverse document frequency of '{term}': {bmidf:.2f}")

def bm25_tf_command(doc_id,term, k1 = BM25_K1, b = BM25_B):
    idx = InvertedIndex()
    idx.load()

    bm25_tf = idx.get_bm25_tf(doc_id, term, k1, b)

    print(f"BM25 with Term frequency of '{term}' in document '{doc_id}': {bm25_tf:.2f}")

def bm25_command(doc_id,term):
    idx = InvertedIndex()
    idx.load()

    bm25 = idx.get_bm25(doc_id, term)

    print(f"BM25 of '{term}' in document '{doc_id}': {bm25:.2f}")

def bm25_search_command(term, limit = BM25_LIMIT):
    idx = InvertedIndex()
    idx.load()

    bm25_search_results = idx.bm25_search(term, limit)

    for index, result in enumerate(bm25_search_results):
        print(f"{index}. {result['doc_id']} {result['title']} - score {result['score']:.2f}")
