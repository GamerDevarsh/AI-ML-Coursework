import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS #in-memory
# from langchain_community.vectorstores import chroma #persistent, metadsat filtering

from langchain.prompts import ChatPromptTemplate
from observability import run_observed_rag_pipeline
# from langchain.retrievers import EnsembleRetriever #hybrid (bm25=keyword+vector search)
# from langchain.retrievers import ParentDocumentRetriver #books, research
# from langchain.retrievers import ContextualCompressionRetriever

# -----------------------------------------
# ENV
# -----------------------------------------
load_dotenv()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #libomp
VECTOR_DB_PATH = "hr_faiss_index"

# -----------------------------------------
# LOAD VECTOR DB
# -----------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True #python==pkl not safe
)

retriever = vectorstore.as_retriever(search_type="similarity", 
                                     search_kwargs={"k": 4})
#mmr=Maximum marginal relevance = diverse results
#similarity_score_threshold=irrelvant information chunks
# -----------------------------------------
# LLM
# -----------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------------------
# PROMPT
# -----------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an HR Support Assistant.

Answer employee questions using ONLY the provided company documents.
If the answer is not found in the documents, say:
"I’m not sure based on current HR policies."

Be clear, professional, and policy-aligned.

HR Context:
{context}

Employee Question:
{question}
""")

# -----------------------------------------
# CHAT LOOP
# -----------------------------------------
def chat():
    print("\nHR Support Chatbot (type 'exit' to quit)\n")

    while True:
        question = input("Employee: ")
        if question.lower() == "exit":
            break

        result = run_observed_rag_pipeline(
            question=question,
            retriever=retriever,
            llm=llm,
            prompt=prompt,
            model_name="gpt-4o-mini",
            embedding_model_name="text-embedding-3-small",
        )
        print("context----------", result["context"])
        print("\nHR Bot:", result["answer"])
        print("metrics----------", result["metrics"])
        print("-" * 60)


if __name__ == "__main__":
    chat()
