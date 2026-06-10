import csv
import datetime as dt
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Credential and model setup
# Reused exactly from the demo notebook so the application can run with the
# same Vocareum-backed OpenAI-compatible endpoint when LLM phrasing is enabled.
# ---------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

USE_LLM = False
MODEL_NAME = "gpt-4o-mini"

if USE_LLM:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


# ---------------------------------------------------------------------------
# Input contracts and structured trace objects
# ---------------------------------------------------------------------------
@dataclass
class Email:
    from_addr: str
    subject: str
    body: str
    received_at: dt.datetime


@dataclass
class OrderRecord:
    order_id: str
    status: str
    carrier: str | None
    tracking_id: str | None
    eta: str | None
    items: list[dict[str, Any]]
    last_updated: str
    payment_status: str
    shipping_address_masked: str


@dataclass
class TraceEvent:
    stage: str
    status: str
    detail: str
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))


@dataclass
class AgentResult:
    order_id: str | None
    order_record: OrderRecord | None
    reply: str
    outcome: str
    trace: list[TraceEvent]
    suggested_action: str | None = None


# ---------------------------------------------------------------------------
# Mock backend data
# This mirrors the notebook scenario but carries more business detail so the
# application looks closer to a real support workflow.
# ---------------------------------------------------------------------------
MOCK_DB: dict[str, OrderRecord] = {
    "ORD-1045": OrderRecord(
        order_id="ORD-1045",
        status="Shipped",
        carrier="Delhivery",
        tracking_id="DLV123456789IN",
        eta="2025-11-14",
        items=[{"sku": "TSHIRT-RED-M", "qty": 1, "unit_price": 799}],
        last_updated="2025-11-12T14:35:00",
        payment_status="Paid",
        shipping_address_masked="Bengaluru, KA, IN",
    ),
    "ORD-2048": OrderRecord(
        order_id="ORD-2048",
        status="Processing",
        carrier=None,
        tracking_id=None,
        eta="2025-11-16",
        items=[
            {"sku": "JNS-INDIGO-32", "qty": 1, "unit_price": 1999},
            {"sku": "BELT-BLK-34", "qty": 1, "unit_price": 599},
        ],
        last_updated="2025-11-12T10:10:00",
        payment_status="Paid",
        shipping_address_masked="Chennai, TN, IN",
    ),
}

ORDER_ID_RE = re.compile(r"\bORD-\d{4,6}\b", flags=re.IGNORECASE)
LOG_PATH = Path("crm_log.csv")


def add_trace(trace: list[TraceEvent], stage: str, status: str, detail: str) -> None:
    trace.append(TraceEvent(stage=stage, status=status, detail=detail))


def validate_email(email: Email, trace: list[TraceEvent]) -> bool:
    if "@" not in email.from_addr:
        add_trace(trace, "input_validation", "failed", "Sender email address is malformed.")
        return False

    if not email.subject.strip() and not email.body.strip():
        add_trace(trace, "input_validation", "failed", "Both subject and body are empty.")
        return False

    add_trace(
        trace,
        "input_validation",
        "passed",
        f"Sender={email.from_addr}, subject_length={len(email.subject)}, body_length={len(email.body)}",
    )
    return True


def extract_order_id(text: str, trace: list[TraceEvent]) -> str | None:
    match = ORDER_ID_RE.search(text)
    if not match:
        add_trace(trace, "order_id_extraction", "failed", "No valid order ID pattern found in email text.")
        return None

    order_id = match.group(0).upper()
    add_trace(trace, "order_id_extraction", "passed", f"Detected order ID {order_id}.")
    return order_id


def get_order_status(order_id: str, trace: list[TraceEvent]) -> OrderRecord | None:
    record = MOCK_DB.get(order_id)
    if not record:
        add_trace(trace, "backend_lookup", "failed", f"{order_id} does not exist in mock order database.")
        return None

    add_trace(
        trace,
        "backend_lookup",
        "passed",
        f"Status={record.status}, carrier={record.carrier}, eta={record.eta}, payment={record.payment_status}",
    )
    return record


def summarize_items(items: list[dict[str, Any]]) -> str:
    return ", ".join(f"{item['sku']} x{item['qty']}" for item in items)


def compose_update_email(customer_email: str, record: OrderRecord, trace: list[TraceEvent]) -> str:
    if USE_LLM:
        prompt = (
            "Write a concise and professional order status update email.\n"
            f"Customer email: {customer_email}\n"
            f"Order ID: {record.order_id}\n"
            f"Status: {record.status}\n"
            f"Carrier: {record.carrier}\n"
            f"Tracking ID: {record.tracking_id}\n"
            f"ETA: {record.eta}\n"
            f"Items: {summarize_items(record.items)}\n"
            "Keep the message under 140 words, friendly, accurate, and support-oriented."
        )
        reply = llm.invoke(prompt).content
        add_trace(trace, "response_generation", "passed", f"LLM response generated using {MODEL_NAME}.")
        return reply

    lines = [
        f"Subject: Update on your order {record.order_id}",
        "",
        "Hi,",
        "",
        f"We checked your request for order {record.order_id}.",
        f"Current order status: {record.status}.",
        f"Payment status: {record.payment_status}.",
        f"Items in this order: {summarize_items(record.items)}.",
        f"Last backend update recorded at: {record.last_updated}.",
    ]

    if record.carrier and record.tracking_id:
        lines.append(f"Shipping carrier: {record.carrier}.")
        lines.append(f"Tracking ID: {record.tracking_id}.")
        lines.append(f"Tracking link: https://tracking.example.com/{record.tracking_id}")
    else:
        lines.append("A carrier and tracking link are not yet available because the shipment is still being prepared.")

    if record.eta:
        lines.append(f"Estimated delivery date: {record.eta}.")

    lines.extend(
        [
            f"Delivery region on file: {record.shipping_address_masked}.",
            "",
            "If you need invoice, return, or address-change help, reply to this email and our team will assist you.",
            "",
            "Thanks,",
            "Support Team",
        ]
    )
    add_trace(trace, "response_generation", "passed", "Deterministic support email drafted successfully.")
    return "\n".join(lines)


def log_interaction(email: Email, order_id: str | None, outcome: str, notes: str, trace: list[TraceEvent]) -> None:
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["timestamp", "from_addr", "subject", "order_id", "outcome", "notes"],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "from_addr": email.from_addr,
                "subject": email.subject,
                "order_id": order_id or "",
                "outcome": outcome,
                "notes": notes[:500],
            }
        )
    add_trace(trace, "crm_logging", "passed", f"Interaction appended to {LOG_PATH.resolve()}.")


def handle_email(email: Email) -> AgentResult:
    trace: list[TraceEvent] = []
    add_trace(trace, "email_reception", "passed", f"Email received at {email.received_at.isoformat(timespec='seconds')}.")

    if not validate_email(email, trace):
        reply = (
            "Subject: We could not process your request\n\n"
            "Hi,\n\n"
            "We were unable to process your message because the input format was invalid. "
            "Please resend your query with a valid email body and order reference.\n\n"
            "Thanks,\nSupport Team"
        )
        log_interaction(email, None, "INVALID_INPUT", "Validation failure before extraction.", trace)
        return AgentResult(
            order_id=None,
            order_record=None,
            reply=reply,
            outcome="INVALID_INPUT",
            trace=trace,
            suggested_action="Ask the customer to resend the request with valid email content and a clear order reference.",
        )

    order_id = extract_order_id(email.subject + "\n" + email.body, trace)
    if not order_id:
        reply = (
            "Subject: Could you share your Order ID?\n\n"
            "Hi,\n\n"
            "We could not locate a valid order ID in your message. "
            "Please reply using a format like ORD-1045 so we can check the status immediately.\n\n"
            "Thanks,\nSupport Team"
        )
        log_interaction(email, None, "MISSING_ORDER_ID", "Requested customer to resend valid order ID.", trace)
        return AgentResult(
            order_id=None,
            order_record=None,
            reply=reply,
            outcome="MISSING_ORDER_ID",
            trace=trace,
            suggested_action="Prompt the customer to share the order ID in the format ORD-XXXX.",
        )

    order_record = get_order_status(order_id, trace)
    if not order_record:
        add_trace(
            trace,
            "business_handling",
            "passed",
            "Unknown order ID is treated as a supported business scenario and routed for customer confirmation/manual review.",
        )
        reply = (
            f"Subject: Order {order_id} not found\n\n"
            "Hi,\n\n"
            f"We could not find {order_id} in our system. "
            "This can happen if the ID was typed incorrectly, the order was created under a different account, "
            "or the backend has not synced yet. Please confirm the order ID or share your invoice / confirmation "
            "message for a manual check.\n\n"
            "Thanks,\nSupport Team"
        )
        log_interaction(email, order_id, "ORDER_NOT_FOUND", "Order ID not found in backend.", trace)
        return AgentResult(
            order_id=order_id,
            order_record=None,
            reply=reply,
            outcome="ORDER_NOT_FOUND",
            trace=trace,
            suggested_action=(
                "Escalate to manual review, ask for invoice/confirmation details, and re-check after backend sync."
            ),
        )

    reply = compose_update_email(email.from_addr, order_record, trace)
    log_interaction(
        email,
        order_record.order_id,
        "ORDER_STATUS_SENT",
        f"Status={order_record.status}; ETA={order_record.eta}; Carrier={order_record.carrier}",
        trace,
    )
    return AgentResult(
        order_id=order_record.order_id,
        order_record=order_record,
        reply=reply,
        outcome="ORDER_STATUS_SENT",
        trace=trace,
        suggested_action="No escalation needed. Status was retrieved and shared with the customer.",
    )


def print_trace(trace: list[TraceEvent]) -> None:
    print("\n=== DETAILED EXECUTION TRACE ===")
    for index, event in enumerate(trace, start=1):
        print(f"{index}. [{event.timestamp}] {event.stage} -> {event.status}")
        print(f"   Detail: {event.detail}")


def print_result(result: AgentResult) -> None:
    print("\n=== APPLICATION OUTPUT ===")
    print(f"Outcome: {result.outcome}")
    print(f"Resolved Order ID: {result.order_id}")
    print(f"Order Found: {bool(result.order_record)}")
    print(f"Suggested Action: {result.suggested_action}")

    if result.order_record:
        print("\n--- ORDER RECORD SNAPSHOT ---")
        for key, value in asdict(result.order_record).items():
            print(f"{key}: {value}")

    print("\n--- CUSTOMER RESPONSE ---")
    print(result.reply)
    print_trace(result.trace)


def main() -> None:
    sample_email = Email(
        from_addr="jane.doe@example.com",
        subject="Order status request: ORD-1045",
        body=(
            "Hello Support,\n"
            "I placed an order last week but have not received an update yet.\n"
            "Order ID: ORD-1045. Please share the latest status and tracking link.\n"
            "Thanks,\nJane"
        ),
        received_at=dt.datetime.now(),
    )

    result = handle_email(sample_email)
    print_result(result)


if __name__ == "__main__":
    main()
