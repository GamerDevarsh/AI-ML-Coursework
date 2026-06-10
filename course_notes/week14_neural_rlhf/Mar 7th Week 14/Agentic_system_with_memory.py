import os, json, requests
import streamlit as st
import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import deque #5
import faiss, numpy as np
from sentence_transformers import SentenceTransformer



from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL = os.getenv("MODEL")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


st.title("易 Agentic AI System with Tools, Memory & Collaboration")

st.markdown("""
Features:
""")

user_input = st.text_input("Ask me anything (e.g., 'Weather in Delhi', '25*4+9'): ")


tools = [
   {
       "type": "function",
       "function": {
           "name": "calculator",
           "description": "Performs arithmetic operations like '23*7+4'.",
           "parameters": {
               "type": "object",

               "properties": {

                   "expression": {"type": "string", "description": "Mathematical expression to evaluate"}

               },

               "required": ["expression"]

           },

       },

   },

   {

       "type": "function",

       "function": {

           "name": "get_weather",

           "description": "Fetches current temperature and windspeed for a city using the Open-Meteo API.",

           "parameters": {

               "type": "object",

               "properties": {

                   "city": {"type": "string", "description": "City name to fetch weather for"}

               },

               "required": ["city"]

           },

       },

   },

]


def calculator(expression: str):

   try:

       result = eval(expression, {"__builtins__": {}})

       return {"result": result}

   except Exception as e:

       return {"error": str(e)}



def get_weather(city: str):

   """Simple weather lookup via Open-Meteo API."""

   try:

       url = f"https://api.open-meteo.com/v1/forecast?latitude=28.6&longitude=77.2&current_weather=true"

       data = requests.get(url).json()

       temp = data["current_weather"]["temperature"]

       wind = data["current_weather"]["windspeed"]

       return {"city": city, "temperature": temp, "windspeed": wind}

   except Exception as e:

       return {"error": str(e)}


if "short_term_memory" not in st.session_state:
    st.session_state.short_term_memory = deque(maxlen=5)


# embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  
# index = faiss.IndexFlatL2(384)

if "long_term_index" not in st.session_state:
    st.session_state.long_term_index = faiss.IndexFlatL2(384)

if "long_term_texts" not in st.session_state:
    st.session_state.long_term_texts = []

if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

memory_texts = []
def remember_long_term(text):

   emb = st.session_state.embedder.encode([text])

   st.session_state.long_term_index.add(np.array(emb, dtype="float32"))

   st.session_state.long_term_texts.append(text)



def recall_long_term(query, k=2):

   if len(st.session_state.long_term_texts) == 0:

       return []

   q_emb = st.session_state.embedder.encode([query])

   D, I = st.session_state.long_term_index.search(np.array(q_emb, dtype="float32"), k)

   return [st.session_state.long_term_texts[i] for i in I[0] if i < len(st.session_state.long_term_texts)]


def planner_agent(task):

   """Planner Agent: decides approach (reasoning/tool use)."""

   system_prompt = f"You are a Planner Agent. Decide the next step for: {task}. Choose between reasoning, calculator, or weather API."

   messages = [{"role": "system", "content": system_prompt},

               {"role": "user", "content": f"Plan how to answer: {task}"}]

   resp = client.chat.completions.create(model="gpt-4.1-mini", messages=messages, temperature=0.3)

   return resp.choices[0].message.content



def executor_agent(task, plan):

   """Executor Agent: performs reasoning or tool call via function-calling."""

   messages = [

       {"role": "system", "content": "You are the Executor Agent. Perform reasoning or invoke tools as needed."},

       {"role": "user", "content": f"User task: {task}. Plan: {plan}"}

   ]



   response = client.chat.completions.create(

       model="gpt-4.1-mini",

       messages=messages,

       tools=tools,

       tool_choice="auto",

       temperature=0.2

   )



   choice = response.choices[0].message


   if hasattr(choice, "tool_calls") and choice.tool_calls:

       fn = choice.tool_calls[0]

       fn_name = fn.function.name

       args = json.loads(fn.function.arguments)

       if fn_name == "calculator":

           return calculator(**args)

       elif fn_name == "get_weather":

           return get_weather(**args)

       else:

           return {"error": "Unknown tool"}

   else:

       return {"reasoning": choice.content}



def reflector_agent(task, result):

   prompt = f"Reflect if the answer to '{task}' is accurate, complete, and clear. Suggest improvement if needed.\nAnswer: {result}"

   resp = client.chat.completions.create(model="gpt-4.1-mini",

                                         messages=[{"role": "user", "content": prompt}],

                                         temperature=0.3)

   return resp.choices[0].message.content




if user_input:


   past_context = " ".join(list(st.session_state.short_term_memory))

   recalls = recall_long_term(user_input)

   context = f"Recent context: {past_context}\nLong-term memory: {recalls}\nUser query: {user_input}"




   plan = planner_agent(context)

   result = executor_agent(user_input, plan)

   reflection = reflector_agent(user_input, result)




   st.session_state.short_term_memory.append(f"Q: {user_input} | A: {result}")

   remember_long_term(f"User: {user_input} | Result: {result}")




   st.subheader(" Planner Agent (Sense + Plan)")

   st.write(plan)



   st.subheader("Executor Agent (ReAct + Tools)")

   st.json(result)



   st.subheader("Reflector Agent (Self-Evaluation)")

   st.write(reflection)



   st.subheader("Memory State")

   st.write(f"Short-Term Buffer: {list(st.session_state.short_term_memory)}")

   st.write(f"Long-Term Memory Count: {len(st.session_state.long_term_texts)}")



   st.success("Agentic workflow executed successfully!")




