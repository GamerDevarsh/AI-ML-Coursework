"""
Agentic AI Concepts Demo (Pure Python, No external libs)

Covers:
1) Single-agent vs Multi-agent
2) MCP-like tool interface (standardized tool calling)
3) A2A (Agent-to-Agent) messaging + message schema
4) Speaker selection strategies: round_robin, auto
5) Interaction patterns: sequential, parallel (simulated), hierarchical
6) Planner -> Worker -> Reviewer -> Composer protocol flow
7) Coordination: turn-taking + routing
8) Safety: loop prevention, timeouts (max steps), confidence escalation

Run:
  python agent_system_demo.py
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable, Tuple
import time
import hashlib
import json
import random


# -----------------------------
# A2A Message Schema
# -----------------------------
@dataclass
class A2AMessage:
    task_id: str
    sender: str
    receiver: str
    intent: str              # e.g. "plan", "execute", "review", "compose", "tool_call"
    payload: Dict[str, Any]  # data for the receiver
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# -----------------------------
# MCP-like Tool Layer
# (Standard interface: name + schema + callable)
# -----------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolServer:
    """MCP-like tool server exposing tools via a standard call interface."""

    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec):
        self.tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"ok": False, "error": f"Unknown tool: {tool_name}", "result": None}

        tool = self.tools[tool_name]
        # lightweight schema check
        required = tool.input_schema.get("required", [])
        for k in required:
            if k not in args:
                return {"ok": False, "error": f"Missing required arg: {k}", "result": None}

        try:
            out = tool.fn(args)
            return {"ok": True, "error": None, "result": out}
        except Exception as e:
            return {"ok": False, "error": str(e), "result": None}


# -----------------------------
# Simple Tools (for demo)
# -----------------------------
def tool_search(args: Dict[str, Any]) -> Dict[str, Any]:
    q = args["query"].lower()
    # pretend "search" results
    kb = {
        "mcp": "MCP standardizes how agents call tools/APIs/data sources via structured contracts.",
        "a2a": "A2A standardizes agent-to-agent messaging, routing, and handoffs.",
        "hierarchical": "Hierarchical planning: planner decomposes goal into sub-tasks delegated to workers.",
    }
    hits = [v for k, v in kb.items() if k in q]
    if not hits:
        hits = ["No strong match found; returning generic notes."]
    return {"hits": hits, "top_hit": hits[0]}


def tool_calculate_discount(args: Dict[str, Any]) -> Dict[str, Any]:
    actual = float(args["actual"])
    discounted = float(args["discounted"])
    if actual <= 0:
        return {"discount_pct": None, "note": "Actual price must be > 0"}
    pct = (actual - discounted) / actual
    return {"discount_pct": pct}


def tool_summarize(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args["text"]
    # naive summarizer: first N chars/sentences style
    short = text.strip()
    if len(short) > 180:
        short = short[:180].rstrip() + "..."
    return {"summary": short}


# -----------------------------
# Agent Base Class
# -----------------------------
class Agent:
    def __init__(self, name: str, role: str, system_message: str, description: str):
        self.name = name
        self.role = role
        self.system_message = system_message
        self.description = description

    def act(self, msg: A2AMessage, ctx: Dict[str, Any]) -> Tuple[Optional[A2AMessage], Dict[str, Any]]:
        """Return: (next_message, updated_ctx)"""
        raise NotImplementedError


# -----------------------------
# Single Agent (all-in-one)
# -----------------------------
class SingleAgent(Agent):
    def __init__(self, tool_server: ToolServer):
        super().__init__(
            name="SingleAgent",
            role="generalist",
            system_message="You are a single-agent assistant. You plan, execute, validate, and compose alone.",
            description="Does everything end-to-end (simple but less scalable).",
        )
        self.tools = tool_server

    def solve(self, task: str) -> Dict[str, Any]:
        # plan
        plan = ["search key terms", "draft answer", "self-check", "final response"]
        # execute (tool)
        search = self.tools.call("search", {"query": task})
        # draft
        draft = f"Task: {task}\nPlan: {plan}\nEvidence: {search['result']['top_hit']}\n"
        # validate
        ok = "MCP" in draft or "A2A" in draft or "agent" in task.lower()
        # compose
        final = draft + f"\nValidation: {'OK' if ok else 'Needs more evidence'}\n"
        return {"plan": plan, "final": final, "validated": ok}


# -----------------------------
# Multi-Agent Roles
# Planner -> Worker -> Reviewer -> Composer
# -----------------------------
class PlannerAgent(Agent):
    def act(self, msg: A2AMessage, ctx: Dict[str, Any]) -> Tuple[Optional[A2AMessage], Dict[str, Any]]:
        task = msg.payload["task"]
        # hierarchical decomposition
        steps = [
            {"id": "S1", "type": "tool_search", "query": "mcp a2a hierarchical"},
            {"id": "S2", "type": "write", "instruction": f"Explain MCP vs A2A and single vs multi-agent for: {task}"},
            {"id": "S3", "type": "review", "criteria": ["clarity", "no repetition", "correct comparisons"]},
            {"id": "S4", "type": "compose", "format": "trainer_ready"},
        ]
        ctx["plan"] = steps
        ctx["state"] = "planned"

        return (
            A2AMessage(
                task_id=msg.task_id,
                sender=self.name,
                receiver="Worker",
                intent="execute",
                payload={"steps": steps},
                timestamp=time.time(),
            ),
            ctx,
        )


class WorkerAgent(Agent):
    def __init__(self, tool_server: ToolServer):
        super().__init__(
            name="Worker",
            role="executor",
            system_message="You execute assigned steps using tools and return structured outputs.",
            description="Executes plan steps; uses MCP-like tool server.",
        )
        self.tools = tool_server

    def act(self, msg: A2AMessage, ctx: Dict[str, Any]) -> Tuple[Optional[A2AMessage], Dict[str, Any]]:
        steps = msg.payload["steps"]
        outputs = []
        for s in steps:
            if s["type"] == "tool_search":
                out = self.tools.call("search", {"query": s["query"]})
                outputs.append({"step": s["id"], "kind": "tool", "tool": "search", "out": out})
            elif s["type"] == "write":
                # use retrieved evidence if exists
                evidence = ctx.get("worker_evidence", "")
                # gather evidence from previous outputs
                evidence_hits = []
                for o in outputs:
                    if o["kind"] == "tool" and o["out"]["ok"]:
                        evidence_hits.extend(o["out"]["result"]["hits"])
                evidence = "\n".join(evidence_hits) if evidence_hits else "No evidence retrieved."
                draft = (
                    f"{s['instruction']}\n\n"
                    f"Key comparisons:\n"
                    f"- Single agent: simpler, but limited scalability.\n"
                    f"- Multi-agent: specialized roles + coordination (A2A).\n"
                    f"- MCP: agent↔tools standard interface.\n"
                    f"- A2A: agent↔agent standard messaging/routing.\n\n"
                    f"Evidence:\n{evidence}\n"
                )
                outputs.append({"step": s["id"], "kind": "draft", "text": draft, "confidence": 0.72})
            else:
                # review/compose steps handled by other agents
                continue

        ctx["worker_outputs"] = outputs
        ctx["state"] = "executed"

        return (
            A2AMessage(
                task_id=msg.task_id,
                sender=self.name,
                receiver="Reviewer",
                intent="review",
                payload={"outputs": outputs},
                timestamp=time.time(),
            ),
            ctx,
        )


class ReviewerAgent(Agent):
    def act(self, msg: A2AMessage, ctx: Dict[str, Any]) -> Tuple[Optional[A2AMessage], Dict[str, Any]]:
        outputs = msg.payload["outputs"]
        draft = next((o for o in outputs if o["kind"] == "draft"), None)
        if not draft:
            review = {"approved": False, "issues": ["No draft found"], "confidence": 0.2}
        else:
            text = draft["text"]
            issues = []
            # simple consistency checks
            if "MCP" not in text or "A2A" not in text:
                issues.append("Missing MCP/A2A mention in explanation.")
            if len(text) < 120:
                issues.append("Draft too short.")
            # pretend confidence score
            confidence = min(0.95, draft.get("confidence", 0.6) + (0.15 if not issues else -0.2))
            review = {"approved": len(issues) == 0, "issues": issues, "confidence": confidence}

        ctx["review"] = review
        ctx["state"] = "reviewed"

        # confidence escalation example:
        if review["confidence"] < 0.6:
            # route back to worker for improvements
            return (
                A2AMessage(
                    task_id=msg.task_id,
                    sender=self.name,
                    receiver="Worker",
                    intent="revise",
                    payload={"issues": review["issues"], "hint": "Add clearer MCP vs A2A separation + examples."},
                    timestamp=time.time(),
                ),
                ctx,
            )

        return (
            A2AMessage(
                task_id=msg.task_id,
                sender=self.name,
                receiver="Composer",
                intent="compose",
                payload={"outputs": outputs, "review": review},
                timestamp=time.time(),
            ),
            ctx,
        )


class ComposerAgent(Agent):
    def act(self, msg: A2AMessage, ctx: Dict[str, Any]) -> Tuple[Optional[A2AMessage], Dict[str, Any]]:
        outputs = msg.payload["outputs"]
        review = msg.payload["review"]
        draft = next((o for o in outputs if o["kind"] == "draft"), None)
        if not draft:
            final = "No draft to compose."
        else:
            final = (
                "FINAL TRAINER-READY NOTE\n"
                "========================\n"
                + draft["text"].strip()
                + "\n\nQuality Gate:\n"
                + f"- Approved: {review['approved']}\n"
                + f"- Confidence: {review['confidence']:.2f}\n"
                + ("FINAL_DECISION: APPROVED" if review["approved"] else "FINAL_DECISION: REVISE")
            )

        ctx["final"] = final
        ctx["state"] = "composed"
        return (None, ctx)


# -----------------------------
# GroupChat / Manager (Autogen-like)
# -----------------------------
class GroupChatManager:
    """
    Simulates AutoGen GroupChat:
    - speaker_selection_method: "round_robin" or "auto"
    - max_rounds: loop prevention
    - repeated-message detection
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        speaker_selection_method: str = "auto",
        max_rounds: int = 12,
    ):
        self.agents = agents
        self.speaker_selection_method = speaker_selection_method
        self.max_rounds = max_rounds
        self._rr_order = ["Planner", "Worker", "Reviewer", "Composer"]
        self._rr_idx = 0

        self.seen_hashes = set()

    def _hash_msg(self, m: A2AMessage) -> str:
        h = hashlib.sha256(m.to_json().encode("utf-8")).hexdigest()
        return h

    def _auto_route(self, m: A2AMessage) -> str:
        # minimal routing logic based on intent
        if m.intent == "plan":
            return "Planner"
        if m.intent in ("execute", "revise"):
            return "Worker"
        if m.intent == "review":
            return "Reviewer"
        if m.intent == "compose":
            return "Composer"
        # fallback
        return m.receiver if m.receiver in self.agents else "Planner"

    def _pick_speaker(self, m: A2AMessage) -> str:
        if self.speaker_selection_method == "round_robin":
            name = self._rr_order[self._rr_idx % len(self._rr_order)]
            self._rr_idx += 1
            return name
        # "auto": route based on message intent/receiver
        return self._auto_route(m)

    def run(self, task: str) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"task": task, "state": "start"}
        task_id = f"T-{int(time.time())}-{random.randint(100,999)}"

        # start message
        msg = A2AMessage(
            task_id=task_id,
            sender="User",
            receiver="Planner",
            intent="plan",
            payload={"task": task},
            timestamp=time.time(),
        )

        transcript: List[str] = []

        for round_i in range(self.max_rounds):
            msg_hash = self._hash_msg(msg)
            if msg_hash in self.seen_hashes:
                transcript.append(f"[SAFETY] Loop detected. Stopping at round {round_i}.")
                break
            self.seen_hashes.add(msg_hash)

            speaker = self._pick_speaker(msg)
            if speaker not in self.agents:
                transcript.append(f"[ERROR] Unknown speaker: {speaker}")
                break

            transcript.append(f"\n--- Round {round_i+1} | Speaker: {speaker} | intent={msg.intent} ---")
            transcript.append(f"IN: {msg.to_json()}")

            next_msg, ctx = self.agents[speaker].act(msg, ctx)

            if next_msg is None:
                transcript.append("OUT: (conversation ended)")
                break
            transcript.append(f"OUT: {next_msg.to_json()}")
            msg = next_msg

        return {"ctx": ctx, "transcript": "\n".join(transcript)}


# -----------------------------
# Demo Runner
# -----------------------------
def main():
    # Tools (MCP-like)
    tools = ToolServer()
    tools.register(
        ToolSpec(
            name="search",
            description="Searches a tiny KB and returns semantically relevant notes.",
            input_schema={"required": ["query"]},
            fn=tool_search,
        )
    )
    tools.register(
        ToolSpec(
            name="calc_discount",
            description="Computes discount percentage: (actual - discounted) / actual",
            input_schema={"required": ["actual", "discounted"]},
            fn=tool_calculate_discount,
        )
    )
    tools.register(
        ToolSpec(
            name="summarize",
            description="Naive summarizer for text.",
            input_schema={"required": ["text"]},
            fn=tool_summarize,
        )
    )

    print("\n==============================")
    print("1) SINGLE AGENT DEMO")
    print("==============================")
    single = SingleAgent(tool_server=tools)
    out1 = single.solve("Explain MCP vs A2A and single-agent vs multi-agent.")
    print(out1["final"])

    print("\n==============================")
    print("2) MULTI-AGENT DEMO (Planner->Worker->Reviewer->Composer)")
    print("==============================")
    agents: Dict[str, Agent] = {
        "Planner": PlannerAgent(
            name="Planner",
            role="planner",
            system_message="Decompose tasks into steps.",
            description="Creates hierarchical plan and delegates.",
        ),
        "Worker": WorkerAgent(tool_server=tools),
        "Reviewer": ReviewerAgent(
            name="Reviewer",
            role="reviewer",
            system_message="Validate output. Add missing bits. Remove repetition. Approve if good.",
            description="Quality control and safety checks.",
        ),
        "Composer": ComposerAgent(
            name="Composer",
            role="composer",
            system_message="Merge outputs into a single trainer-ready final note.",
            description="Final formatting + coherence.",
        ),
    }

    manager = GroupChatManager(agents=agents, speaker_selection_method="auto", max_rounds=12)
    result = manager.run("Prepare a trainer-style comparison: single vs multi-agent, MCP vs A2A, where each fits.")
    print(result["ctx"].get("final", "No final output."))
    print("\n(Transcript available in result['transcript'] if you want to print it)")
    print(result["transcript"])
    print("\n==============================")
    print("3) SPEAKER SELECTION (Round Robin) DEMO")
    print("==============================")
    rr_manager = GroupChatManager(agents=agents, speaker_selection_method="round_robin", max_rounds=8)
    rr_result = rr_manager.run("Explain hierarchical vs reactive planning with a tiny example.")
    print(rr_result["ctx"].get("final", "No final output."))

    print("\n==============================")
    print("4) MCP TOOL LIST DEMO")
    print("==============================")
    print(json.dumps(tools.list_tools(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()