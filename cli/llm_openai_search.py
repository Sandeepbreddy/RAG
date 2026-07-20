import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

model = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_content(prompt):
    response = client.responses.create(
        model=model,
        input="Hello AI. Greet me back as this is my introduction to you"
    )

    print("Response from OPEN AI LLM:", response.output_text)
    print("Total Token Count", response.usage.total_tokens)
    print("Response Token Count", response.usage.output_tokens)

if __name__ == "__main__":
    generate_content()