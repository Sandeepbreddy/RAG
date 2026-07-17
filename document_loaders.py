import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader, PyPDFLoader)

from dotenv import load_dotenv
load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write("This is a sample text file for testing.")
        temp_file.flush()
        temp_file_path = temp_file.name

        try:
            loader = TextLoader(temp_file_path)
            documents = loader.load()

            for doc in documents:
                print(doc)
                print(doc.metadata)
                print("Document content:", doc.page_content)
        finally:
            os.remove(temp_file_path)


def pdf_loader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from {pdf_path}")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1} Content Preview: {doc.page_content[:1]}...")
        print("Metadata:", doc.metadata)

if __name__ == "__main__":
    # load_text_file()
    pdf_loader("./docs/Psoriasis-ExistingTherapies.pdf")