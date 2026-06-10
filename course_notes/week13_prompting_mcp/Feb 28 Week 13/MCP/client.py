import requests

BASE_URL = "http://127.0.0.1:8000"

def decide(question):
    r = requests.post(
        f"{BASE_URL}/mcp/decide",
        json={"question": question},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def call_tool(tool, input_dict):
    r = requests.post(
        f"{BASE_URL}/mcp/call",
        json={"tool": tool, "input": input_dict},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    question = input("Ask something: ")

    # Step 1: Host decides tool
    decision = decide(question)

    print("\n Tool Decision:")
    print("Tool chosen:", decision["decided_tool"])
    print("Reason:", decision["reason"])

    # Step 2: Execute chosen tool
    result = call_tool(
        decision["decided_tool"],
        decision["suggested_tool_input"]
    )

    print("\n Tool Result:")
    print(result)