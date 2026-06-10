
from dotenv import load_dotenv
import os
import streamlit as st
from pydantic import BaseModel
import openai


client = openai.AzureOpenAI(
    api_key="YOUR_AZURE_OPENAI_KEY",
    api_version="2023-12-01-preview",
    azure_endpoint="https://openai-api-management-gw.azure-api.net/openaiprodtest/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview"
)


class AgentResponse(BaseModel):
    sense: str
    plan: str
    act: str
    reflection: str
    final_answer: str


def calculator_tool(expression: str) -> str:
    """Safely evaluate arithmetic expressions like '5*7+2'."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


st.title("🤖 Agentic AI Workflow Demo")
st.write(
    """
### Includes:
- AI Agents
"""
)

user_input = st.text_input("Enter your question or task:")


if user_input:
    sense_prompt = f"""

User query: {user_input}
"""
    sense_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sense_prompt}],
        temperature=0.2
    )
    sense = sense_response.choices[0].message.content.strip()

    plan_prompt = f""" '{sense}'.

"""
    plan_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": plan_prompt}],
        temperature=0.3
    )
    plan = plan_response.choices[0].message.content.strip()

    if any(op in user_input for op in ["+", "-", "*", "/", "**"]):
        calc_result = calculator_tool(user_input)
        act = f"Used calculator_tool('{user_input}') → {calc_result}"
        final_answer = calc_result
    else:
        act_prompt = f"""
You are reasoning to answer the question directly.
User query: {user_input}
Provide concise answer in one paragraph.
"""
        act_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": act_prompt}],
            temperature=0.4
        )
        act = "Performed reasoning action."
        final_answer = act_response.choices[0].message.content.strip()

    reflection_prompt = f"""
 {final_answer}
"""
    reflection_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reflection_prompt}],
        temperature=0.3
    )
    reflection = reflection_response.choices[0].message.content.strip()

    structured = AgentResponse(
        sense=sense,
        plan=plan,
        act=act,
        reflection=reflection,
        final_answer=final_answer
    )

 
    st.subheader("🧠 Sense (Understand)")
    st.write(structured.sense)

    st.subheader("🧩 Plan (Decide Next Actions)")
    st.write(structured.plan)

    st.subheader("⚙️ Act (Reason / Use Tool)")
    st.write(structured.act)

    st.subheader("💭 Reflection (Self-Evaluation)")
    st.write(structured.reflection)

    st.subheader("✅ Final Answer")
    st.success(structured.final_answer)

