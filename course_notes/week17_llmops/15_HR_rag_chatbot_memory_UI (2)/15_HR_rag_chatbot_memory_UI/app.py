import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from observability import run_observed_rag_pipeline

# -----------------------------------------
# ENV SETUP
# -----------------------------------------
load_dotenv()

st.set_page_config(
    page_title="HR Support Chatbot",
    page_icon="💼",
    layout="centered"
)

# -----------------------------------------
# CONSTANTS
# -----------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
VECTOR_DB_PATH = "hr_faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# -----------------------------------------
# LOAD VECTOR STORE (CACHED)
# -----------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------------------
# LOAD LLM (CACHED)
# -----------------------------------------
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )

llm = load_llm()

# -----------------------------------------
# PROMPT
# -----------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an HR Support Assistant for company employees.

Answer questions using ONLY the information provided in the HR documents.
If the answer is not present, say:
"I’m not sure based on current HR policies."

Be clear, professional, and concise.

HR Context:
{context}

Employee Question:
{question}
""")

# -----------------------------------------
# SESSION STATE (CHAT MEMORY)
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------
# UI
# -----------------------------------------
st.title("HR Support Chatbot")
st.markdown(
    "Ask questions about onboarding, leave policy, compensation, IT usage, "
    "security guidelines, and workplace safety."
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------
# CHAT INPUT
# -----------------------------------------
user_question = st.chat_input("Ask an HR-related question...")

if user_question:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Searching HR policies..."):
        result = run_observed_rag_pipeline(
            question=user_question,
            retriever=retriever,
            llm=llm,
            prompt=prompt,
            model_name=CHAT_MODEL,
            embedding_model_name=EMBEDDING_MODEL,
        )

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        with st.expander("Trace, Retrieval, Cost, and Evaluation"):
            metrics = result["metrics"]
            st.json(metrics)
            st.markdown("**Retrieved context preview**")
            st.code(result["context"][:3000] or "No context returned.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
