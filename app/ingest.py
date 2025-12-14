import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

from .config import DATA_DIR, VECTORSTORE_DIR, OPENAI_API_KEY, EMBEDDING_MODEL_NAME


def load_documents(data_dir: str):
    docs = []
    for file in Path(data_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata["source"] = file.name
        docs.extend(pdf_docs)
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL_NAME
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    vectorstore.persist()
    return vectorstore


def main():
    print(f"Loading documents from {DATA_DIR}...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents")

    print("Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Creating vectorstore...")
    create_vectorstore(chunks)
    print(f"Vectorstore saved to {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
