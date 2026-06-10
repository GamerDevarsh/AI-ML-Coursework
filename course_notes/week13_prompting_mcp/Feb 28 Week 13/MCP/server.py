import os
import ast
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="Demo MCP Server", version="0.3")

# -----------------------------
# Models
# -----------------------------
class ToolCall(BaseModel):
    tool: str
    input: Dict[str, Any]

class ToolResult(BaseModel):
    tool: str
    output: Dict[str, Any]
    error: Optional[str] = None

class DecideRequest(BaseModel):
    question: str

class DecideResponse(BaseModel):
    question: str
    decided_tool: str
    reason: str
    suggested_tool_input: Dict[str, Any]
    available_tools: List[Dict[str, Any]]


# -----------------------------
# Tool registry (single source of truth)
# -----------------------------
TOOL_SPECS = [
    {
        "name": "llm_complete",
        "when_to_use": "General questions, explanations, writing, summarization, reasoning.",
        "user_should_ask_like": [
            "Explain MCP in simple terms",
            "Write an email draft",
            "Summarize this text"
        ],
        "input_schema": {"prompt": "string"},
    },
    {
        "name": "web_search",
        "when_to_use": "Search on the web (stub in this demo).",
        "user_should_ask_like": [
            "Search latest AI trends",
            "Find info about X"
        ],
        "requires": ["GOOGLE_API_KEY"],
        "input_schema": {"query": "string"},
    },
    {
        "name": "calculator",
        "when_to_use": "Arithmetic / numeric expression evaluation.",
        "user_should_ask_like": [
            "Calculate (10+2)*3",
            "What is 2**10?"
        ],
        "input_schema": {"expression": "string"},
    },
    {
        "name": "read_file",
        "when_to_use": "Read a local text file from allowed directory (sandbox).",
        "user_should_ask_like": [
            "Read demo.txt",
            "Show contents of notes.md"
        ],
        "input_schema": {"path": "string", "max_chars": "int (optional)"},
    },
]

@app.get("/mcp/tools")
async def list_tools():
    return {"tools": TOOL_SPECS}


# -----------------------------
# ✅ Tool: Calculator (safe)
# -----------------------------
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Constant,
    ast.Load,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
)

def _safe_eval_math(expr: str) -> float:
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression.")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed element: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")
    return eval(compile(tree, filename="<calculator>", mode="eval"), {"__builtins__": {}}, {})

async def calculator(expression: str) -> Dict[str, Any]:
    result = _safe_eval_math(expression)
    return {"expression": expression, "result": result}


# -----------------------------
# ✅ Tool: File Reader (safe path sandbox)
# -----------------------------
ALLOWED_DIR = Path(os.getenv("MCP_ALLOWED_DIR", str(Path.cwd() / "allowed_files"))).resolve()

def _resolve_safe_path(user_path: str) -> Path:
    if not user_path or user_path.strip() == "":
        raise ValueError("path is required")
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = (ALLOWED_DIR / candidate)
    candidate = candidate.resolve()
    if ALLOWED_DIR not in candidate.parents and candidate != ALLOWED_DIR:
        raise ValueError(f"Access denied. Path must be inside: {ALLOWED_DIR}")
    return candidate

async def read_file(path: str, max_chars: int = 20000) -> Dict[str, Any]:
    safe_path = _resolve_safe_path(path)
    if not safe_path.exists():
        raise ValueError(f"File not found: {safe_path}")
    if not safe_path.is_file():
        raise ValueError(f"Not a file: {safe_path}")
    data = safe_path.read_text(encoding="utf-8", errors="replace")
    if max_chars and max_chars > 0:
        data = data[: int(max_chars)]
    return {"path": str(safe_path), "allowed_dir": str(ALLOWED_DIR), "content": data}


# -----------------------------
# LLM tool (optional for your demo)
# -----------------------------
async def llm_complete(prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("MODEL", "gpt-4o-mini")
    if not api_key or not base_url:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY / OPENAI_API_BASE not set.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return {"text": resp.choices[0].message.content}


async def web_search(query: str) -> Dict[str, Any]:
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY not set; web_search tool disabled.")
    return {"query": query, "note": "Implement actual Google search call here."}


# -----------------------------
# Tool decision logic
# -----------------------------
def decide_tool(question: str) -> DecideResponse:
    q = (question or "").strip()
    ql = q.lower()

    # Heuristics (simple + predictable)
    # 1) File reading intent
    if any(k in ql for k in ["read file", "open file", "show file", "read ", "open "]) and any(
        ql.endswith(ext) for ext in [".txt", ".md", ".log", ".json", ".csv"]
    ):
        decided = "read_file"
        reason = "Question looks like a request to open/read a local file."
        suggested = {"path": q.split()[-1], "max_chars": 20000}
        return DecideResponse(
            question=q, decided_tool=decided, reason=reason,
            suggested_tool_input=suggested, available_tools=TOOL_SPECS
        )

    # 2) Calculator intent: contains digits/operators and typical math symbols
    math_chars = set("0123456789+-*/().%^")
    if any(ch in math_chars for ch in q) and any(op in q for op in ["+", "-", "*", "/", "**", "%", "(" , ")"]):
        decided = "calculator"
        reason = "Question contains a likely arithmetic expression."
        suggested = {"expression": q}
        return DecideResponse(
            question=q, decided_tool=decided, reason=reason,
            suggested_tool_input=suggested, available_tools=TOOL_SPECS
        )

    # 3) Web search intent words
    if any(k in ql for k in ["search", "find latest", "latest news", "look up", "google"]):
        decided = "web_search"
        reason = "Question asks to search / look up information."
        suggested = {"query": q}
        return DecideResponse(
            question=q, decided_tool=decided, reason=reason,
            suggested_tool_input=suggested, available_tools=TOOL_SPECS
        )

    # 4) Default: LLM
    decided = "llm_complete"
    reason = "Defaulting to LLM for general language tasks."
    suggested = {"prompt": q}
    return DecideResponse(
        question=q, decided_tool=decided, reason=reason,
        suggested_tool_input=suggested, available_tools=TOOL_SPECS
    )


@app.post("/mcp/decide", response_model=DecideResponse)
async def decide(req: DecideRequest):
    decision = decide_tool(req.question)
    # Print decision on server console
    print(f"[DECIDE] Q={decision.question!r} -> tool={decision.decided_tool} reason={decision.reason}")
    return decision


# -----------------------------
# /mcp/call (executes tool)
# -----------------------------
@app.post("/mcp/call", response_model=ToolResult)
async def call_tool(req: ToolCall):
    try:
        # Print tool chosen (execution)
        print(f"[CALL] tool={req.tool} input={req.input}")

        if req.tool == "llm_complete":
            out = await llm_complete(prompt=str(req.input.get("prompt", "")))
            return ToolResult(tool=req.tool, output=out)

        if req.tool == "web_search":
            out = await web_search(query=str(req.input.get("query", "")))
            return ToolResult(tool=req.tool, output=out)

        if req.tool == "calculator":
            out = await calculator(expression=str(req.input.get("expression", "")))
            return ToolResult(tool=req.tool, output=out)

        if req.tool == "read_file":
            max_chars = req.input.get("max_chars", 20000)
            out = await read_file(path=str(req.input.get("path", "")), max_chars=int(max_chars))
            return ToolResult(tool=req.tool, output=out)

        raise HTTPException(status_code=404, detail=f"Unknown tool: {req.tool}")

    except Exception as e:
        return ToolResult(tool=req.tool, output={}, error=str(e))