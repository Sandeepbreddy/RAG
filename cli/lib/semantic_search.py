
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from lib.search_utils import CACHE_PATH, load_movies, LIMIT
import re
import json


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embeddings_path = CACHE_PATH/'movie_embeddings.npy'

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Must have text to create embedding")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embedding_docs(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        
        return self.build_embeddings(documents)
    
    def search(self, query, limit = LIMIT):
        if self.embeddings is None:
            ValueError("No embedding loaded")
        
        qry_emb  = self.generate_embedding(query)

        similarities = []

        for doc_emb, doc in zip(self.embeddings,self.documents):
            _similarity = cosine_similarity(qry_emb, doc_emb)

            similarities.append((_similarity, doc))
        
        similarities.sort(key = lambda x: x[0], reverse=True)
        res = []

        for sc,doc in similarities[:limit]:
            res.append({'score': sc, 'title': doc['title'], 'desc' : doc['description']})  
        return res

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.document_map = {}


        self.chunk_embeddings_path = CACHE_PATH/'chunk_embeddings.npy'
        self.chunk_metadata_path = CACHE_PATH/'chunk_metadata.json'


    def build_chunk_embeddings(self, documents):
        self.documents = documents

        # for doc in documents:
        self.document_map = {idx:doc for idx,doc in enumerate(documents)}

        all_chunks= []
        chunk_metadata = []
        
        for midx,doc in enumerate(documents):
            if doc["description"].strip()=="":
                continue

            _chunks = semantic_chunk(doc['description'], max_chunk_size = 4, overlap=1)

            all_chunks+=_chunks
            
            for cidx in range(len(_chunks)):
                chunk_metadata.append({
                    "movie_idx": midx,
                    "chunk_idx": cidx,
                    "total_chunks": len(_chunks)
                })
        
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = {
            "chunks": chunk_metadata,
            "total_chunks": len(all_chunks)
        }


        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {idx:doc for idx,doc in enumerate(documents)}

        # for doc in documents:
        #     self.document_map = {doc['id']:doc for doc in documents}

        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, text, limit: int = LIMIT):
        query_emb = self.generate_embedding(text)
        chunk_scores = []
        movie_scores = defaultdict(lambda: 0)

        for idx in range(len(self.chunk_embeddings)):
            chunk_embedding = self.chunk_embeddings[idx]

            metadata = self.chunk_metadata['chunks'][idx]
            midx, cidx = metadata['movie_idx'], metadata['chunk_idx']

            sim = cosine_similarity(query_emb, chunk_embedding)



            # chunk_scores.append({
            #     'movies_idx': midx,
            #     'chunk_idx': cidx,
            #     'score': sim
            # })

            movie_scores[midx] = max(movie_scores[midx], sim)
            
        movie_scores_sorted = sorted(movie_scores.items(), key=lambda x:x[1], reverse=True)

        res = []

        for midx, score in movie_scores_sorted[:limit]:
            doc = self.document_map[midx]
            res.append({
                "id": doc['id'],
                "title": doc['title'],
                "document": doc['description'][:100],
                "description": doc['description'][:100],
                "score": round(score, 4),
                "metadata":  {}
                })
        return res

def embed_chunk_search(query, limit=LIMIT):
    css = ChunkedSemanticSearch()
    movies = load_movies()
    _ = css.load_or_create_chunk_embeddings(movies)
    result_chunk = css.search_chunks(query, limit)

    for idx, res in enumerate(result_chunk):
        print(f"\n{idx+1}. {res['title']} (score: {res['score']:.4f})")
        print(f"{res['document']}")

def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max Sequence Length: {ss.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    result = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {result[:3]}")
    print(f"Dimensions: {result.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embedding_docs(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
 
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product/(norm1 * norm2)


def search(query, limit = LIMIT):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embedding_docs(documents)

    results = ss.search(query, limit)

    for idx, result in enumerate(results):
        print(f"{idx}. {result['title']} (score: {result['score']:.4f})")
        print(result['desc'][:100])

def fixed_size_chunk(text, chunk_size = 200, overlap = 0):
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap

    for i in range(0, len(words), step_size):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks

def chunk_text(text, chunk_size = 200, overlap =0):
    chunks = fixed_size_chunk(text, chunk_size, overlap)
    
    print(f"Chunking {len(text)} characters")

    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}")    


def semantic_chunk(text, max_chunk_size=4, overlap=0):
    text = text.strip()

    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if (len(sentences)==1) and sentences[0].endswith(('!','.','?')):
        pass

    chunks = []
    step_size = max_chunk_size - overlap

    sentences = [s.strip() for s in sentences if s.strip()]
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i:i+max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks

def chunk_text_semantic(text, max_chunk_size=4, overlap=0):
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Chunking {len(text)} characters")

    for i,chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

