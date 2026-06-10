import os
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from typing_extensions import TypedDict
from dotenv import load_dotenv

from openai import OpenAI  # openai==1.64.0
from langgraph.graph import StateGraph, START, END


# =====================================================================
# LOAD .env + OPENAI CLIENT (NON-AZURE)
# =====================================================================
load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()


if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not OPENAI_API_BASE:
    raise ValueError("Missing OPENAI_API_BASE in .env")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


def llm(messages, max_tokens=250):
    """OpenAI-compatible wrapper (Vocareum base_url)."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()


# =====================================================================
# UNIFIED STATE
# =====================================================================
class AppState(TypedDict):
    user_query: str
    intent: str
    idea: str
    outline: str
    description: str
    blurb: str
    ad_copy: str
    social_post: str
    email_pitch: str
    support_answer: str
    final_output: str


# =====================================================================
# INTENT CLASSIFIER
# =====================================================================
def classify_intent(state: AppState):
    result = llm([
        {"role": "system", "content": "Classify intent ONLY as one of: generate_product, generate_marketing_assets, support_query."},
        {"role": "user", "content": state["user_query"]},
    ])

    # robust fallback parsing
    r = result.lower()
    if "generate_product" in r or ("product" in r and "marketing" not in r):
        intent = "generate_product"
    elif "generate_marketing_assets" in r or "marketing" in r or "content" in r:
        intent = "generate_marketing_assets"
    else:
        intent = "support_query"

    return {"intent": intent, "idea": state["user_query"]}


# =====================================================================
# PROMPT CHAINING
# =====================================================================
def create_outline(state: AppState):
    return {"outline": llm([
        {"role": "system", "content": "Create a clear outline with bullet points and headings."},
        {"role": "user", "content": state["idea"]},
    ])}

def expand_description(state: AppState):
    return {"description": llm([
        {"role": "system", "content": "Expand the outline into a detailed, structured description."},
        {"role": "user", "content": state["outline"]},
    ], max_tokens=500)}

def create_blurb(state: AppState):
    return {"blurb": llm([
        {"role": "system", "content": "Write a short blurb (1–2 lines)."},
        {"role": "user", "content": state["description"]},
    ])}


# =====================================================================
# MARKETING NODES (SEQUENTIAL PATH AS IN YOUR ORIGINAL)
# =====================================================================
def generate_ad_copy(state: AppState):
    return {"ad_copy": llm([
        {"role": "system", "content": "Write punchy ad copy (headline + 2 lines + CTA)."},
        {"role": "user", "content": state["idea"]},
    ])}

def generate_social_post(state: AppState):
    return {"social_post": llm([
        {"role": "system", "content": "Write a LinkedIn post (professional tone, 6–10 lines, include 3 hashtags)."},
        {"role": "user", "content": state["idea"]},
    ], max_tokens=200)}

def generate_email_pitch(state: AppState):
    return {"email_pitch": llm([
        {"role": "system", "content": "Write a short email pitch (subject + body)."},
        {"role": "user", "content": state["idea"]},
    ], max_tokens=250)}


# =====================================================================
# SUPPORT NODE
# =====================================================================
def handle_support(state: AppState):
    return {"support_answer": llm([
        {"role": "system", "content": "You are a helpful support agent. Give a clear step-by-step answer."},
        {"role": "user", "content": state["user_query"]},
    ])}


# =====================================================================
# ROUTER
# =====================================================================
def main_router(state: AppState):
    return state["intent"]


# =====================================================================
# FINAL ASSEMBLY
# =====================================================================
def assemble_final(state: AppState):
    if state["intent"] == "generate_product":
        final = f"""PRODUCT OUTLINE:
{state['outline']}

DESCRIPTION:
{state['description']}

BLURB:
{state['blurb']}
"""
    elif state["intent"] == "generate_marketing_assets":
        final = f"""AD COPY:
{state['ad_copy']}

SOCIAL POST:
{state['social_post']}

EMAIL PITCH:
{state['email_pitch']}
"""
    else:
        final = f"SUPPORT RESPONSE:\n{state['support_answer']}"

    return {"final_output": final}


# =====================================================================
# LANGGRAPH PIPELINE
# =====================================================================
def build_graph():
    g = StateGraph(AppState)

    g.add_node("classify_intent", classify_intent)

    g.add_node("create_outline", create_outline)
    g.add_node("expand_description", expand_description)
    g.add_node("create_blurb", create_blurb)

    g.add_node("generate_ad_copy", generate_ad_copy)
    g.add_node("generate_social_post", generate_social_post)
    g.add_node("generate_email_pitch", generate_email_pitch)

    g.add_node("handle_support", handle_support)

    g.add_node("assemble_final", assemble_final)

    g.add_edge(START, "classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        main_router,
        {
            "generate_product": "create_outline",
            "generate_marketing_assets": "generate_ad_copy",
            "support_query": "handle_support",
        },
    )

    # product chain
    g.add_edge("create_outline", "expand_description")
    g.add_edge("expand_description", "create_blurb")
    g.add_edge("create_blurb", "assemble_final")

    # marketing chain (kept same as your original path)
    g.add_edge("generate_ad_copy", "generate_social_post")
    g.add_edge("generate_social_post", "generate_email_pitch")
    g.add_edge("generate_email_pitch", "assemble_final")

    # support path
    g.add_edge("handle_support", "assemble_final")

    g.add_edge("assemble_final", END)

    return g.compile()


# =====================================================================
# NETWORKX GRAPH VISUALIZER
# =====================================================================
def build_networkx():
    G = nx.DiGraph()

    edges = [
        ("START", "classify_intent"),
        ("classify_intent", "create_outline"),
        ("classify_intent", "generate_ad_copy"),
        ("classify_intent", "handle_support"),

        ("create_outline", "expand_description"),
        ("expand_description", "create_blurb"),
        ("create_blurb", "assemble_final"),

        ("generate_ad_copy", "generate_social_post"),
        ("generate_social_post", "generate_email_pitch"),
        ("generate_email_pitch", "assemble_final"),

        ("handle_support", "assemble_final"),

        ("assemble_final", "END"),
    ]

    for u, v in edges:
        G.add_edge(u, v)

    return G


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.title("🔵 Unified LangGraph Agent with Graph Visualization")
st.caption(f"Model: {MODEL} | Base URL: {OPENAI_API_BASE}")

user_input = st.text_input("Enter your query:")

if user_input:
    graph = build_graph()

    result = graph.invoke({
        "user_query": user_input,
        "intent": "",
        "idea": "",
        "outline": "",
        "description": "",
        "blurb": "",
        "ad_copy": "",
        "social_post": "",
        "email_pitch": "",
        "support_answer": "",
        "final_output": "",
    })

    st.subheader("Final Output")
    st.write(result["final_output"])

    st.subheader("LangGraph Workflow Visualization")

    G = build_networkx()
    plt.figure(figsize=(16, 9))

    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        arrows=True,
        node_size=3200,
        node_color="#90CAF9",
        font_size=10,
        font_weight="bold",
    )

    st.pyplot(plt)
