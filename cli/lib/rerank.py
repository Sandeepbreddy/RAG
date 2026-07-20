import json
import os
from dotenv import load_dotenv

from openai import OpenAI
from lib.search_utils import PROMPT_PATH
from sentence_transformers import CrossEncoder

load_dotenv()

model = "gpt-5-mini"
# model = "gpt-5"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def individual_rerank(query, documents):
    with open(PROMPT_PATH/'individual_rerank.md', 'r') as f:
        prompt = f.read()
    results = []
    for doc in documents:
        _prompt = prompt.format(query = query,
                                title = doc["title"],
                                desciption = doc["description"])
        response = client.responses.create(
                                            model=model,
                                            input=_prompt
                                        )
        print(f"{doc['title']} has LLM rerank of {response.output_text}")
        clean_response_test = (response.output_text or "").strip()
        try:
            clean_response_test = int(clean_response_test)
        except:
            print(f"Failed to case {response.output_text} to int for {doc['title']}")
            clean_response_test = 0

        results.append({**doc, 'rerank_response': int(response.output_text)})
    results = sorted(results, key=lambda x:x['rerank_response'], reverse=True)
    return results

def batch_rerank(query, documents):
    with open(PROMPT_PATH/'batch_rerank.md', 'r') as f:
        prompt = f.read()
    results = []
    doc_list = ''
    _movietmp = '''<movie id={idx}> title={title}:\n{desc}\n</movie>\n'''
    for idx,doc in enumerate(documents):
        doc_list += _movietmp.format(idx=idx, title=doc['title'], desc=doc['description'])
    
    _prompt = prompt.format(query = query,doc_list = doc_list)
    response = client.responses.create(
                                    model=model,
                                    input=_prompt
                                    )
    print(response.output_text)
    response_parsed = json.loads(response.output_text)

    for idx, doc in enumerate(documents):
        results.append({**doc, 'rerank_response':response_parsed.index(idx)})
    results = sorted(results, key=lambda x: x['rerank_response'], reverse=False)

    return results

def cross_encoder_rerank(query, documents):
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title','')} - {doc.get('document','')}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


    scores = cross_encoder.predict(pairs)

    results = []
    for idx, doc in enumerate(documents):
        results.append({**doc, 'rerank_response':scores[idx]})
    
    results = sorted(results, key=lambda x: x['rerank_response'], reverse=True)

    return results



