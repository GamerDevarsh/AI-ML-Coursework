import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from observability import run_observed_rag_pipeline

try:
    from langsmith import Client
    from langsmith.evaluation import evaluate
except ImportError as exc:
    raise ImportError(
        "LangSmith evaluation requires the 'langsmith' package. "
        "Run `pip install -r requirements.txt` in a working Python environment."
    ) from exc


VECTOR_DB_PATH = "hr_faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DATASET_NAME = os.getenv("LANGSMITH_DATASET_NAME", "hr-rag-evaluation-dataset")
EXAMPLES_PATH = Path("evaluation_examples.json")


load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


prompt = ChatPromptTemplate.from_template(
    """
You are an HR Support Assistant.

Answer employee questions using ONLY the provided company documents.
If the answer is not found in the documents, say:
"I'm not sure based on current HR policies."

Be clear, professional, and policy-aligned.

HR Context:
{context}

Employee Question:
{question}
""".strip()
)


def load_vectorstore() -> Any:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_app_components() -> tuple[Any, Any]:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return retriever, llm


def load_examples() -> list[dict[str, Any]]:
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


def ensure_dataset(client: Client) -> str:
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="HR RAG chatbot evaluation dataset",
        )

    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    if existing_examples:
        return dataset.name

    examples = load_examples()
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[example["inputs"] for example in examples],
        outputs=[example["outputs"] for example in examples],
    )
    return dataset.name


def build_target(retriever: Any, llm: Any):
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = run_observed_rag_pipeline(
            question=inputs["question"],
            retriever=retriever,
            llm=llm,
            prompt=prompt,
            model_name=CHAT_MODEL,
            embedding_model_name=EMBEDDING_MODEL,
        )
        return {
            "answer": result["answer"],
            "context": result["context"],
            "metrics": result["metrics"],
        }

    return target


def _judge_json(prompt_text: str) -> dict[str, Any]:
    evaluator_model = os.getenv("LANGSMITH_EVAL_MODEL", "gpt-4o-mini")
    judge = ChatOpenAI(model=evaluator_model, temperature=0)
    response = judge.invoke(prompt_text)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": response.content}


def answer_correctness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    prompt_text = f"""
You are grading an HR chatbot answer against a reference answer.
Return valid JSON only with this schema:
{{
  "score": 0,
  "reasoning": "short explanation"
}}

Use a 1-5 scale:
1 = incorrect
3 = partially correct
5 = correct and policy-aligned

Question: {inputs["question"]}
Reference answer: {reference_outputs["reference_answer"]}
Candidate answer: {outputs["answer"]}
""".strip()
    judgment = _judge_json(prompt_text)
    return {
        "key": "answer_correctness",
        "score": judgment.get("score", 0),
        "comment": judgment.get("reasoning", ""),
    }


def answer_groundedness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    prompt_text = f"""
You are grading whether an HR chatbot answer is grounded in the retrieved context.
Return valid JSON only with this schema:
{{
  "score": 0,
  "reasoning": "short explanation"
}}

Use a 1-5 scale:
1 = unsupported by context
3 = partially supported
5 = fully supported by context

Question: {inputs["question"]}
Retrieved context: {outputs["context"]}
Candidate answer: {outputs["answer"]}
""".strip()
    judgment = _judge_json(prompt_text)
    return {
        "key": "answer_groundedness",
        "score": judgment.get("score", 0),
        "comment": judgment.get("reasoning", ""),
    }


def fallback_behavior(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    answer = outputs["answer"].lower()
    expected = reference_outputs["reference_answer"].lower()
    score = 1 if "not sure based on current hr policies" in answer and "not sure based on current hr policies" in expected else 0
    return {
        "key": "fallback_behavior",
        "score": score,
        "comment": "Checks whether the expected fallback response appears when the reference expects it.",
    }


def main() -> None:
    client = Client()
    dataset_name = ensure_dataset(client)
    retriever, llm = build_app_components()
    target = build_target(retriever, llm)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[answer_correctness, answer_groundedness, fallback_behavior],
        experiment_prefix="hr-rag-langsmith-eval",
        description="Batch evaluation for the HR RAG chatbot using LangSmith.",
    )

    print("LangSmith evaluation launched.")
    print(f"Dataset: {dataset_name}")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
