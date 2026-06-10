import json
import os
import time
from typing import Any

from langchain_openai import ChatOpenAI

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


DEFAULT_PRICING_PER_1K_TOKENS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}

DEFAULT_LANGSMITH_ENV = {
    "LANGCHAIN_API_KEY": "YOUR_LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_PROJECT": "RAG",
    "LANGSMITH_API_KEY": "YOUR_LANGCHAIN_API_KEY",
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_PROJECT": "RAG",
    "ENABLE_LANGSMITH_EVAL": "true",
    "LANGSMITH_EVAL_MODEL": "gpt-4o-mini",
}


def configure_langsmith_env() -> None:
    for key, value in DEFAULT_LANGSMITH_ENV.items():
        os.environ[key] = value


configure_langsmith_env()


def _safe_float(value: str | None, fallback: float) -> float:
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def get_model_pricing(model_name: str) -> dict[str, float]:
    defaults = DEFAULT_PRICING_PER_1K_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})
    env_prefix = model_name.upper().replace("-", "_").replace(".", "_")
    return {
        "input": _safe_float(os.getenv(f"{env_prefix}_INPUT_COST_PER_1K"), defaults["input"]),
        "output": _safe_float(os.getenv(f"{env_prefix}_OUTPUT_COST_PER_1K"), defaults["output"]),
    }


def _get_doc_metadata(doc: Any, index: int) -> dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    return {
        "rank": index,
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "score": metadata.get("score"),
        "preview": getattr(doc, "page_content", "")[:250],
    }


def _estimate_tokens_from_text(text: str) -> int:
    return max(1, round(len(text) / 4))


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {})

    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or 0
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )
    total_tokens = usage.get("total_tokens") or token_usage.get("total_tokens") or (input_tokens + output_tokens)

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def _estimate_cost(model_name: str, usage: dict[str, int]) -> dict[str, float]:
    pricing = get_model_pricing(model_name)
    input_cost = (usage["input_tokens"] / 1000) * pricing["input"]
    output_cost = (usage["output_tokens"] / 1000) * pricing["output"]
    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(input_cost + output_cost, 8),
    }


def _fallback_response(answer: str) -> bool:
    normalized = answer.lower()
    return "not sure based on current hr policies" in normalized


@traceable(name="retrieve_documents")
def retrieve_documents(retriever: Any, question: str, embedding_model_name: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    docs = retriever.invoke(question)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    estimated_query_tokens = _estimate_tokens_from_text(question)
    embedding_cost = (
        _estimate_cost(
            embedding_model_name,
            {
                "input_tokens": estimated_query_tokens,
                "output_tokens": 0,
                "total_tokens": estimated_query_tokens,
            },
        )
        if embedding_model_name
        else None
    )

    return {
        "docs": docs,
        "context": "\n\n".join(doc.page_content for doc in docs),
        "metrics": {
            "retrieval_latency_ms": latency_ms,
            "documents_retrieved": len(docs),
            "estimated_query_tokens": estimated_query_tokens,
            "embedding_model_name": embedding_model_name,
            "estimated_embedding_cost_usd": embedding_cost,
            "documents": [_get_doc_metadata(doc, index + 1) for index, doc in enumerate(docs)],
        },
    }


@traceable(name="generate_answer")
def generate_answer(llm: Any, prompt_messages: list[Any], model_name: str) -> dict[str, Any]:
    start = time.perf_counter()
    response = llm.invoke(prompt_messages)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    usage = _extract_usage(response)
    cost = _estimate_cost(model_name, usage)

    return {
        "response": response,
        "answer": response.content,
        "metrics": {
            "generation_latency_ms": latency_ms,
            "usage": usage,
            "estimated_cost_usd": cost,
            "model_name": model_name,
            "finish_reason": (getattr(response, "response_metadata", {}) or {}).get("finish_reason"),
        },
    }


@traceable(name="evaluate_answer")
def evaluate_answer(question: str, context: str, answer: str) -> dict[str, Any]:
    context_present = bool(context.strip())
    fallback_used = _fallback_response(answer)

    return {
        "context_present": context_present,
        "fallback_used": fallback_used,
        "answer_length": len(answer),
        "context_length": len(context),
        "question_length": len(question),
    }


@traceable(name="judge_answer")
def judge_answer(question: str, context: str, answer: str) -> dict[str, Any] | None:
    if os.getenv("ENABLE_LANGSMITH_EVAL", "false").lower() != "true":
        return None

    evaluator_model = os.getenv("LANGSMITH_EVAL_MODEL", "gpt-4o-mini")
    judge_llm = ChatOpenAI(model=evaluator_model, temperature=0)
    judge_prompt = f"""
You are evaluating an HR RAG chatbot response.
Score each field from 1 to 5 and respond with valid JSON only.

Evaluation rubric:
- retrieval_relevance: Is the supplied context relevant to the user question?
- groundedness: Is the answer supported by the supplied context?
- helpfulness: Is the answer clear and useful for the employee?

Return this JSON schema:
{{
  "retrieval_relevance": 0,
  "groundedness": 0,
  "helpfulness": 0,
  "notes": "short explanation"
}}

Question: {question}

Context:
{context}

Answer:
{answer}
""".strip()

    result = judge_llm.invoke(judge_prompt)
    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        return {"raw_evaluation": result.content}


@traceable(name="rag_pipeline")
def run_observed_rag_pipeline(
    question: str,
    retriever: Any,
    llm: Any,
    prompt: Any,
    chat_history: str = "",
    model_name: str = "gpt-4o-mini",
    embedding_model_name: str | None = None,
) -> dict[str, Any]:
    pipeline_start = time.perf_counter()

    retrieval = retrieve_documents(retriever, question, embedding_model_name=embedding_model_name)
    prompt_kwargs = {
        "context": retrieval["context"],
        "question": question,
    }
    if chat_history:
        prompt_kwargs["chat_history"] = chat_history

    generation = generate_answer(
        llm=llm,
        prompt_messages=prompt.format_messages(**prompt_kwargs),
        model_name=model_name,
    )

    heuristic_evaluation = evaluate_answer(
        question=question,
        context=retrieval["context"],
        answer=generation["answer"],
    )
    judge_evaluation = judge_answer(
        question=question,
        context=retrieval["context"],
        answer=generation["answer"],
    )

    total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    return {
        "answer": generation["answer"],
        "response": generation["response"],
        "context": retrieval["context"],
        "documents": retrieval["docs"],
        "metrics": {
            **retrieval["metrics"],
            **generation["metrics"],
            "total_latency_ms": total_latency_ms,
            "estimated_end_to_end_cost_usd": round(
                generation["metrics"]["estimated_cost_usd"]["total_cost_usd"]
                + (
                    retrieval["metrics"]["estimated_embedding_cost_usd"]["total_cost_usd"]
                    if retrieval["metrics"]["estimated_embedding_cost_usd"]
                    else 0.0
                ),
                8,
            ),
            "evaluation": {
                "heuristic": heuristic_evaluation,
                "judge": judge_evaluation,
            },
        },
    }
