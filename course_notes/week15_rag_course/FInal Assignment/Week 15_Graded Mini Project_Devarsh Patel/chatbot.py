import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from guardrails import validate_query, validate_response

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
VECTOR_PATH = BASE_DIR / "vectorstore"


@dataclass
class ChatbotRuntime:
    chain: Any
    memory: ConversationBufferMemory


def load_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_path = VECTOR_PATH if VECTOR_PATH.exists() else Path("vectorstore")
    return FAISS.load_local(
        str(vector_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_chain() -> ChatbotRuntime:
    vectorstore = load_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #For each query → retrieve top 3 documents

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    '''
    This solves a common problem:

    User asks:
    What is FastAPI?
    How do I use it?

    Second question is unclear.

    The system rewrites it to:
    How do I use FastAPI?
    '''

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a question rewriting assistant. Given chat history and the latest user message, "
                "rewrite the latest message into a standalone question that preserves user intent.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    
    '''
    chat history + new question -> rewrite question ->retrieve documents
    '''
    history_aware_retriever = create_history_aware_retriever(
        llm,
        base_retriever,
        rewrite_prompt,
    )

    '''
    This prompt tells the LLM:
        Use ONLY retrieved documents.
        If answer not found → refuse.
    '''

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful documentation assistant. Answer the user's question using ONLY the "
                "retrieved context blocks below:\n\n{context}\n\n"
                "If the answer is not present in the retrieved documents, respond exactly with: "
                "'I don't have enough information in the provided documents.' Do not invent facts.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return ChatbotRuntime(chain=rag_chain, memory=memory)

'''
This function tries to extract the answer and sources from the chain's output in a flexible way,
to accommodate different possible output formats from the chain.
'''

def _extract_answer_and_sources(result: Any) -> tuple[str, list[str]]:
    if isinstance(result, dict):
        answer = (
            result.get("answer")
            or result.get("output_text")
            or result.get("result")
            or ""
        )

        sources: list[str] = []

        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                src = (
                    doc.metadata.get("source")
                    or doc.metadata.get("filename")
                    or doc.metadata.get("source_id")
                )
                if src:
                    sources.append(str(src))

        if not sources:
            for key in ("documents", "docs", "context"):
                if key in result and result[key]:
                    for doc in result[key]:
                        metadata = getattr(doc, "metadata", {}) or {}
                        src = (
                            metadata.get("source")
                            or metadata.get("filename")
                            or metadata.get("source_id")
                        )
                        if src:
                            sources.append(str(src))

        return str(answer), sources

    return str(result), []


def ask_question(runtime: ChatbotRuntime, question: str) -> str:
    query = validate_query(question)

    chat_history: list[dict[str, str]] = []
    try:
        messages = runtime.memory.chat_memory.messages #Try to extract chat history in a flexible way, to accommodate different message formats
        for message in messages:
            content = getattr(message, "content", None)
            if content is not None:
                chat_history.append(
                    {
                        "role": str(getattr(message, "type", "user")),
                        "content": str(content),
                    }
                )
            elif isinstance(message, dict):
                chat_history.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message.get("content", "")),
                    }
                )
    except Exception:
        chat_history = []

    payload = {"input": query, "chat_history": chat_history}

    try:
        result = runtime.chain.invoke(payload)
    except TypeError:
        try:
            result = runtime.chain(payload)
        except Exception:
            result = runtime.chain.invoke({"input": query})

    answer_text, sources = _extract_answer_and_sources(result)

    final_answer = answer_text.strip() or "I don't have enough information in the provided documents."

    if sources:
        unique_sources = sorted(set(sources))
        citations = "\n\nSources:\n" + "\n".join(
            f"- {os.path.basename(path)}" for path in unique_sources
        )
        final_answer += citations

    validated_answer = validate_response(final_answer)

    try:
        runtime.memory.chat_memory.add_user_message(query)
        runtime.memory.chat_memory.add_ai_message(validated_answer)
    except Exception:
        pass

    return validated_answer


if __name__ == "__main__":
    runtime = build_chain()

    print("Domain Support Assistant (history-aware retriever) ready.\nType 'exit' to quit.\n")

    while True:
        q = input("User: ").strip()

        if q.lower() in ["exit", "quit"]:
            break

        try:
            response = ask_question(runtime, q)
        except Exception as e:
            response = f"Error while processing the request: {e}"

        print("Bot:", response)


'''
# SIMPLE IMPLEMENTATION WITHOUT USING LANGCHAIN CHAINS, TO SHOW THE CORE LOGIC MORE CLEARLY
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

try:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.chains import ConversationalRetrievalChain
except ModuleNotFoundError:
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

from guardrails import validate_query, validate_response

load_dotenv()

VECTOR_PATH = "vectorstore"


def load_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def build_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain


def ask_question(chain, question):

    query = validate_query(question)

    result = chain.invoke({"question": query})

    answer = result["answer"]

    return validate_response(answer)


if __name__ == "__main__":

    chain = build_chain()

    while True:

        q = input("User: ")

        if q.lower() in ["exit", "quit"]:
            break

        response = ask_question(chain, q)

        print("Bot:", response)
'''
