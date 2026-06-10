'''
Document Ingestion (ingest.py)

This script:

loads documents

chunks them

generates embeddings

builds FAISS index
'''
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_PATH = "docs"
VECTOR_PATH = "vectorstore"


def load_documents():

    documents = []

    for file in os.listdir(DOCS_PATH):

        path = os.path.join(DOCS_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

    return documents


def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(docs)


def build_vectorstore(chunks):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(VECTOR_PATH)

    print("Vector store created successfully.")


if __name__ == "__main__":

    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)