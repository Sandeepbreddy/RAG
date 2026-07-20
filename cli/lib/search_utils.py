import json
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]

DATA_PATH=PROJECT_ROOT/'data'/'movies.json'
STOP_WORDS_PATH=PROJECT_ROOT/'data'/'stopwords.txt'

CACHE_PATH=PROJECT_ROOT/'cache'
PROMPT_PATH=PROJECT_ROOT/'cli'/'lib'/'prompts'
BM25_K1 = 1.5
BM25_B = 0.75
BM25_LIMIT = 5
LIMIT = 5

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data['movies']

def load_stopwords():
    with open(STOP_WORDS_PATH, "r") as f:
        data = f.read().splitlines()
    return data