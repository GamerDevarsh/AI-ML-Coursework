from openai import OpenAI
import os
from dotenv import load_dotenv

"""
Question
model answer
human---best answer
preferences stored
"""

# Load environment variables
load_dotenv()

MODEL = os.getenv("MODEL")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

MODEL = "gpt-4o-mini"


def generate_answers(question):

    prompt = f"""
Answer the following question in two different ways.

Question: {question}

Answer A:
Answer B:
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )

    return response.choices[0].message.content


def collect_human_feedback(question, answers):

    print("\nQUESTION:")
    print(question)

    print("\nANSWERS:")
    print(answers)

    preference = input("\nWhich answer is better? (A/B): ")

    return {
        "question": question,
        "answers": answers,
        "preferred": preference
    }


# Example question
question = "How does reinforcement learning update neural network weights without labels?"

answers = generate_answers(question)

feedback = collect_human_feedback(question, answers)

print("\nStored Feedback Data:")
print(feedback)