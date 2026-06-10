import streamlit as st
from chatbot import build_chain, ask_question

st.set_page_config(page_title="Technology Documentation Assistant")

st.title("Technology Documentation Assistant")

# Build chain only once
@st.cache_resource
def load_chain():
    return build_chain()

chain = load_chain()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
if prompt := st.chat_input("Ask a question about the documents"):

    # Show user message
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Get response from RAG
    response = ask_question(chain, prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )


# Sidebar controls
st.sidebar.title("Debug & Memory Tools")

# Show memory buffer
if st.sidebar.button("Show Conversation Memory"):

    st.sidebar.write("### ConversationBufferMemory")

    try:
        memory_data = chain.memory.chat_memory.messages
        for m in memory_data:
            st.sidebar.write(m)
    except:
        st.sidebar.write("Memory not available.")


# Show retrieval path
if st.sidebar.button("Show Retrieval Details"):

    st.sidebar.write("### Retrieval Path")

    try:
        retriever = chain.retriever
        docs = retriever.get_relevant_documents(
            st.session_state.messages[-1]["content"]
        )

        for i, d in enumerate(docs):
            st.sidebar.write(f"Document {i+1}")
            st.sidebar.write(d.metadata)
            st.sidebar.write(d.page_content[:300])
            st.sidebar.write("---")

    except:
        st.sidebar.write("No retrieval data available.")