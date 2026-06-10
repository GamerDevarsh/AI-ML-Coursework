from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

MODEL = os.getenv("MODEL")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

DATA_FILE = "feedback_dataset.json"


# Generate multiple answers
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


# Collect human feedback
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


# Store feedback dataset
def store_feedback(data):

    try:
        with open(DATA_FILE, "r") as f:
            dataset = json.load(f)
    except:
        dataset = []

    dataset.append(data)

    with open(DATA_FILE, "w") as f:
        json.dump(dataset, f, indent=4)

    print("\nFeedback stored successfully!")


# Adaptive learning loop
def adaptive_learning():

    while True:

        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answers = generate_answers(question)

        feedback = collect_human_feedback(question, answers)

        store_feedback(feedback)


# Run system
adaptive_learning()