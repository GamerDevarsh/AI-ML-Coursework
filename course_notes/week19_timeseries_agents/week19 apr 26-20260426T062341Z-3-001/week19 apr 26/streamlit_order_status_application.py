import datetime as dt
from dataclasses import asdict

import pandas as pd
import streamlit as st

from detailed_order_status_application import (
    LOG_PATH,
    MODEL_NAME,
    MOCK_DB,
    USE_LLM,
    Email,
    handle_email,
)


st.set_page_config(
    page_title="Order Status Resolution Agent",
    layout="wide",
)


def load_crm_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp", "from_addr", "subject", "order_id", "outcome", "notes"])


def build_default_email(order_id: str) -> tuple[str, str, str]:
    subject = f"Order status request: {order_id}"
    body = (
        "Hello Support,\n"
        "I placed an order last week but have not received an update yet.\n"
        f"Order ID: {order_id}. Please share the latest status and tracking link.\n"
        "Thanks,\nJane"
    )
    return "jane.doe@example.com", subject, body


def render_header() -> None:
    st.title("Order Status Resolution Agent")
    st.caption("A Streamlit application version of the Week 19 demo, with detailed traceability for every workflow step.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Application Type", "Customer Support")
    col2.metric("Workflow Style", "Agentic Automation")
    col3.metric("LLM Mode", "Enabled" if USE_LLM else "Deterministic")
    col4.metric("Configured Model", MODEL_NAME if USE_LLM else "gpt-4o-mini ready")

    st.info(
        "This app reuses the same OpenAI-compatible credential configuration from the demo file. "
        "The UI does not print the secret value, but the backend code keeps the same setup."
    )


def render_sidebar() -> tuple[str, str, str, str]:
    st.sidebar.header("Demo Controls")
    selected_order = st.sidebar.selectbox("Load sample order", options=list(MOCK_DB.keys()), index=0)
    default_email, default_subject, default_body = build_default_email(selected_order)

    st.sidebar.markdown("**Known mock orders**")
    for order_id, record in MOCK_DB.items():
        st.sidebar.write(f"{order_id}: {record.status}, ETA {record.eta}")

    return selected_order, default_email, default_subject, default_body


def render_input_form(default_email: str, default_subject: str, default_body: str) -> Email | None:
    st.subheader("1. Customer Email Input")
    st.write("Enter the incoming customer request exactly as it would arrive in the support workflow.")

    with st.form("email_form", clear_on_submit=False):
        from_addr = st.text_input("Customer email", value=default_email)
        subject = st.text_input("Subject", value=default_subject)
        body = st.text_area("Email body", value=default_body, height=180)
        submitted = st.form_submit_button("Run Application")

    if not submitted:
        return None

    return Email(
        from_addr=from_addr,
        subject=subject,
        body=body,
        received_at=dt.datetime.now(),
    )


def render_result(email: Email) -> None:
    result = handle_email(email)

    st.subheader("2. Workflow Outcome")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Outcome", result.outcome)
    kpi2.metric("Resolved Order ID", result.order_id or "Not found")
    kpi3.metric("Backend Record Found", "Yes" if result.order_record else "No")

    if result.outcome == "ORDER_NOT_FOUND":
        st.warning(
            "The order ID format is valid, but the order does not exist in the backend dataset. "
            "This should be treated as a handled support case, not an application error."
        )
    elif result.outcome in {"MISSING_ORDER_ID", "INVALID_INPUT"}:
        st.info("The agent is asking the customer for corrected input before it can proceed.")
    else:
        st.success("The application completed the order status workflow successfully.")

    if result.suggested_action:
        st.markdown("**Recommended Next Action**")
        st.write(result.suggested_action)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("**Customer-Facing Response**")
        st.code(result.reply, language="text")

        st.markdown("**Minute Application Flow**")
        for index, event in enumerate(result.trace, start=1):
            with st.expander(f"Step {index}: {event.stage} [{event.status}]", expanded=True):
                st.write(f"Timestamp: `{event.timestamp}`")
                st.write(event.detail)

    with right:
        st.markdown("**Backend Record Snapshot**")
        if result.order_record:
            st.json(asdict(result.order_record), expanded=True)
        else:
            st.warning("No backend order record was returned for this request.")
            st.markdown(
                """
**How the app handles this**

- The order ID is still accepted as a valid extracted identifier.
- Backend lookup is attempted normally.
- If the record is missing, the case is marked as `ORDER_NOT_FOUND`.
- The customer is asked to confirm the ID or share invoice details.
- The interaction is still logged into CRM for audit and manual follow-up.
"""
            )

        st.markdown("**Input Email Snapshot**")
        st.json(
            {
                "from_addr": email.from_addr,
                "subject": email.subject,
                "body": email.body,
                "received_at": email.received_at.isoformat(timespec="seconds"),
            },
            expanded=True,
        )

    st.subheader("3. Data Tables")
    table_left, table_right = st.columns(2)

    with table_left:
        st.markdown("**Trace Table**")
        trace_rows = [
            {
                "timestamp": event.timestamp,
                "stage": event.stage,
                "status": event.status,
                "detail": event.detail,
            }
            for event in result.trace
        ]
        st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)

    with table_right:
        st.markdown("**CRM Log Preview**")
        st.dataframe(load_crm_log(), use_container_width=True, hide_index=True)

    st.subheader("4. Explanation of What the Application Covers")
    st.markdown(
        """
This application demonstrates a complete customer-support agent workflow:

- Email reception from a customer asking for order status.
- Input validation so malformed requests do not silently pass through the system.
- Order ID extraction using a controlled pattern.
- Backend lookup against a mock order system.
- Customer reply generation using deterministic logic or the configured LLM path.
- CRM logging for auditability, reporting, and governance.
- Full trace output so every minute stage can be explained during a live demo.
"""
    )


def main() -> None:
    render_header()
    _, default_email, default_subject, default_body = render_sidebar()
    email = render_input_form(default_email, default_subject, default_body)

    if email is None:
        st.subheader("Ready to Run")
        st.write("Use the sample order from the sidebar or customize the email, then click `Run Application`.")
        return

    render_result(email)


if __name__ == "__main__":
    main()
