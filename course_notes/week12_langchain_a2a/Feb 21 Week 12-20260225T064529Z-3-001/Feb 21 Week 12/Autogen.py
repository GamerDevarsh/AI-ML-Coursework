import os
import streamlit as st
from dotenv import load_dotenv

import autogen
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)

# ================================
# Config (from .env)
# ================================
load_dotenv()

MODEL = "gpt-4o-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()


if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")


LLM_CONFIG = {
    "config_list": [
        {
            "model": MODEL,
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_API_BASE,   
            "api_type": "openai",          
        }
    ],
    "temperature": 0.2,
}

# ================================
# Build Agents
# ================================
def build_agents():
    """
    Build all agents for the multi-agent system:
    - HumanUser (UserProxyAgent)
    - Researcher (AssistantAgent)
    - Reviewer   (AssistantAgent)
    - GroupChat + GroupChatManager (Orchestration pipeline)
    """

    user_proxy = UserProxyAgent(
        name="HumanUser",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=False,
        system_message=(
            "You represent the human user. You pass their requests into the "
            "multi-agent system and return the final answer."
        ),
    )

    researcher = AssistantAgent(
        name="Researcher",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are a research assistant.\n"
            "- Carefully read the user request.\n"
            "- Produce a structured draft with sections:\n"
            "  1. OVERVIEW\n"
            "  2. KEY POINTS\n"
            "  3. BENEFITS / USE-CASES\n"
            "  4. RISKS / LIMITATIONS\n"
            "  5. EXAMPLES\n"
            "- Be concise but informative.\n"
            "- Assume the audience is technical but busy."
        ),
        description="Does the main research and writes a first draft.",
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are a strict reviewer.\n"
            "- Read the Researcher's draft.\n"
            "- Fix clarity, remove repetition, and ensure consistency.\n"
            "- If anything is missing, briefly add it.\n"
            "- Your final answer must be:\n"
            "  * Well-structured\n"
            "  * Easy to read\n"
            "  * Directly useful for a practitioner or trainer\n"
            "- End with a line: 'FINAL_DECISION: APPROVED'."
        ),
        description="Reviews and polishes the answer, then approves.",
    )

    group_chat = GroupChat(
        agents=[user_proxy, researcher, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        send_introductions=True,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    return user_proxy, researcher, reviewer, manager


# ================================
# Run Multi-Agent Workflow
# ================================
def run_multi_agent_workflow(user_topic: str, extra_instruction: str):
    """
    Run one end-to-end multi-agent research + review workflow.
    Returns final answer (reviewer) and full conversation history.
    """
    user_proxy, _, _, manager = build_agents()

    message = f"""
User topic / task:
{user_topic}

Additional instructions:
{extra_instruction}

Workflow:
1. Researcher: create a structured draft as per your system_message.
2. Reviewer: refine, correct, and finalize the answer.
3. Reviewer: end with 'FINAL_DECISION: APPROVED'.

Begin.
""".strip()

    _ = user_proxy.initiate_chat(
        manager,
        message=message,
    )

    msgs = manager.groupchat.messages

    final_answer = "No final answer produced."
    for m in reversed(msgs):
        if m.get("name") == "Reviewer":
            final_answer = m.get("content", "")
            break

    return final_answer, msgs


# ================================
# Streamlit UI
# ================================
def main():
    st.set_page_config(page_title="Multi-Agent Research + Review", layout="wide")
    st.title("Multi-Agent Research + Review (AutoGen + OpenAI-Compatible API)")

    st.markdown(
        """
### What this app does

This is a **multi-agent system**:

- **HumanUser (UserProxyAgent)** – represents you.
- **Researcher (AssistantAgent)** – performs structured research and drafts.
- **Reviewer (AssistantAgent)** – critiques, improves, and **approves** the final answer.
- **GroupChatManager** – orchestrates which agent speaks next.

No OCR, no document upload – it's pure **text-based research + review orchestration**.
"""
    )

    st.caption(f"Model: `{MODEL}` | Base URL: `{OPENAI_API_BASE}`")

    default_topic = "Explain Edge AI vs Cloud AI for real-time inference in autonomous vehicles."
    user_topic = st.text_area(
        "Enter your topic / task for the agents:",
        value=default_topic,
        height=120,
    )

    extra_instruction = st.text_area(
        "Any extra instructions? (optional)",
        value="Write it in a way I can use directly in a training session for engineers.",
        height=100,
    )

    if st.button("Run Multi-Agent Workflow"):
        if not user_topic.strip():
            st.error("Please enter a topic or task.")
            return

        with st.spinner("Agents are collaborating..."):
            final_answer, history = run_multi_agent_workflow(user_topic, extra_instruction)

        st.success("Multi-agent workflow completed.")

        st.subheader("Final Reviewed Answer")
        st.write(final_answer)

        st.subheader("Conversation Transcript (All Agents)")
        for msg in history:
            name = msg.get("name", "Unknown")
            content = msg.get("content", "")
            st.markdown(f"### {name}")
            st.write(content)
            st.markdown("---")


if __name__ == "__main__":
    main()
