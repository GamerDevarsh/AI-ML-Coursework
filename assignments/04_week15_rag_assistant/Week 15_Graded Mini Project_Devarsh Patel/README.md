# Domain Support Assistant -- Technology Documentation RAG Chatbot

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG)
conversational assistant for the Technology Documentation domain. The
assistant answers questions about programming frameworks, APIs, and
development tools by retrieving relevant information from publicly
available documentation.

The system ensures responses remain grounded in the provided documents
and prevents hallucinations. It also supports follow-up questions using
conversation history.

------------------------------------------------------------------------

# Domain Explanation

The chosen domain is **Technology Documentation and Developer Guides**.

Developers rely heavily on documentation when learning frameworks, APIs,
or tools. However, documentation can be long and difficult to search
manually.

This assistant helps by: - Retrieving relevant sections from technical
documentation - Answering questions using those sections - Supporting
follow‑up questions using conversation history

Example queries: - What is FastAPI? - How do you create an API
endpoint? - Does FastAPI support asynchronous programming?

------------------------------------------------------------------------

# Public Data Sources

FastAPI Documentation\
https://fastapi.tiangolo.com/

PDF:\
https://www.tutorialspoint.com/fastapi/fastapi_tutorial.pdf

Python Programming Guide\
https://docs.python.org/3/tutorial/

PDF:\
https://www.halvorsen.blog/documents/programming/python/resources/Python%20Programming.pdf

Google API Design Guide\
https://cloud.google.com/apis/design

PDF:\
https://cloud.google.com/apis/design/resources/api_design_guide.pdf

------------------------------------------------------------------------

# System Architecture

Workflow:

User Question\
↓\
Query Validation (Guardrails)\
↓\
History‑aware Question Rewriting\
↓\
Vector Search (FAISS)\
↓\
Retrieve Relevant Document Chunks\
↓\
LLM Generates Grounded Answer\
↓\
Return Answer with Document Citations

Core Components: - OpenAI Embeddings - FAISS Vector Database -
LangChain - ConversationBufferMemory - Streamlit Interface

------------------------------------------------------------------------

# Project Structure

week15_rag_assistant/

docs/\
 fastapi_tutorial.pdf\
 python_programming.pdf\
 api_design_guide.pdf

vectorstore/\
 index.faiss\
 index.pkl

chatbot.py\
ingest.py\
guardrails.py\
app.py\
requirements.txt\
README.md

------------------------------------------------------------------------

# ⚠ Python Version Requirement

This project **must be run using Python 3.12**.

Current LangChain, Pydantic, and related packages **do not fully support
Python 3.14 yet**, which will cause dependency and runtime errors.

If you have multiple Python versions installed, explicitly create the
environment using **Python 3.12**.

------------------------------------------------------------------------

# Setup Instructions

## 1. Clone the repository

git clone `<repository>`{=html}\
cd week15_rag_assistant

------------------------------------------------------------------------

## 2. Create Python 3.12 virtual environment

Use the following command to ensure Python 3.12 is used:

py -3.12 -m venv .venv_stable

Activate the environment:

Windows:

..venv_stable`\Scripts`{=tex}`\activate`{=tex}

Mac/Linux:

source .venv_stable/bin/activate

------------------------------------------------------------------------

## 3. Install dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

## 4. Configure OpenAI API Key

Create a `.env` file in the project root:

OPENAI_API_KEY=your_openai_key_here

------------------------------------------------------------------------

# Document Ingestion

Before running the chatbot, documents must be indexed into the vector
database.

Run:

python ingest.py

This process will:

1.  Load documents from the `docs/` folder
2.  Split documents into semantic chunks
3.  Generate embeddings
4.  Store vectors in the FAISS database

------------------------------------------------------------------------

# Running the Chatbot

## Terminal Chatbot

Run:

python chatbot.py

Example:

User: What is FastAPI?\
Bot: FastAPI is a modern Python framework used to build APIs.

Sources:\
fastapi_tutorial.pdf

------------------------------------------------------------------------

## Streamlit Interface (Recommended)

Run:

streamlit run app.py

Features:

-   Conversational chat interface
-   Follow‑up question support
-   Document source citations
-   Retrieval‑based answers

------------------------------------------------------------------------

# Technologies Used

Python\
LangChain\
OpenAI API\
FAISS Vector Database\
Streamlit

------------------------------------------------------------------------

# Conclusion

This project demonstrates a complete **Retrieval‑Augmented Generation
(RAG) pipeline** capable of answering domain‑specific questions using
external documentation while maintaining accuracy and conversational
context.
