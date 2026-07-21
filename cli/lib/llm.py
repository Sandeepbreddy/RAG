import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from lib.search_utils import PROMPT_PATH
load_dotenv()

model = "gpt-5-mini"
# model = "gpt-5"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_content(prompt, query, **kwargs):
    prompt = prompt.format(query=query, **kwargs)
    response = client.responses.create(
        model=model,
        input=prompt
    )
    return response.output_text

def correct_spelling(query):
    with open(PROMPT_PATH/'spelling.md', 'r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

def rewrite_query(query):
    with open(PROMPT_PATH/'rewrite.md', 'r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

def llm_judge(query, formatted_results):
    with open(PROMPT_PATH/'llm_evaluate.md', 'r') as f:
        prompt = f.read()
    results = generate_content(prompt, query, formatted_results = formatted_results)

    results = json.loads(results)

    return results

def llm_answer_question(query, document):
    with open(PROMPT_PATH/'answer_question.md', 'r') as f:
        prompt = f.read()
    results = generate_content(prompt, query=query, docs=document)

    return results

def llm_movie_summarization(query, document):
    with open(PROMPT_PATH/'summarization.md', 'r') as f:
        prompt = f.read()
    results = generate_content(prompt, query=query, docs=document)

    return results

def llm_movie_summary_citation(query, document):
    with open(PROMPT_PATH/'answer_with_citation.md', 'r') as f:
        prompt = f.read()
    results = generate_content(prompt, query=query, docs=document)

    return results