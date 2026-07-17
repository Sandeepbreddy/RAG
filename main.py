from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain")
graph_version = version("langgraph")

from langchain_openai import ChatOpenAI

print(f"LangChain Core Version: {core_version}")
print(f"LangGraph Version: {graph_version}")
print(f"LangChain OpenAI Version: {ChatOpenAI}")

def main():
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word and in all caps.") 
    print("Response from OPEN AI LLM:", response)

    print("Setup Complete")

if __name__ == "__main__":
    main()
