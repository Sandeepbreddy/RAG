import os
from dotenv import load_dotenv

from openai import OpenAI
from lib.search_utils import PROMPT_PATH
load_dotenv()

model = "gpt-5-mini"
# model = "gpt-5"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_content(prompt, query):
    prompt = prompt.format(query=query)
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