'''
Guardrails & Validation (guardrails.py)

These ensure safety and prevent hallucination.
'''
def validate_query(query):

    if len(query.strip()) < 3:
        raise ValueError("Query too short.")

    if any(x in query.lower() for x in ["hack", "exploit"]):
        raise ValueError("Unsafe query detected.")

    return query


def validate_response(answer):

    if answer is None or answer.strip() == "":
        return "I don’t have enough information in the provided documents."

    return answer