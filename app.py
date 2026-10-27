import streamlit as st
import time
from src.chatbot_engine import SupportChatbotOrchestrator

# Page Configuration
st.set_page_config(
    page_title="Nova Support | Context-Aware AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .ethical-banner {
        background: linear-gradient(90deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid #2563eb;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #1e40af;
        margin-bottom: 20px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-intent { background-color: #e0e7ff; color: #3730a3; }
    .badge-confidence { background-color: #dcfce7; color: #166534; }
    .badge-escalated { background-color: #fee2e2; color: #991b1b; }
    .badge-time { background-color: #f1f5f9; color: #334155; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = SupportChatbotOrchestrator()

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = "S1001"

if 'messages' not in st.session_state:
    st.session_state.messages = {}

if st.session_state.current_session_id not in st.session_state.messages:
    st.session_state.messages[st.session_state.current_session_id] = [
        {
            "role": "assistant",
            "content": "Hello! I am **Nova**, an AI-powered customer support assistant. I can help you with billing, technical troubleshooting, product information, shipping, returns, and account questions. How may I assist you today?",
            "intent": "product_info",
            "confidence": 0.99,
            "escalated": 0,
            "latency": 250,
            "turn_id": 1
        }
    ]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/chat.png", width=110)
    st.title("Control & Telemetry")
    
    st.subheader("Active Session")
    session_options = ["S0192", "S0968", "S0625", "S1001", "S1002", "Custom New"]
    selected_session = st.selectbox("Select or Switch Session ID:", session_options, index=3)
    
    if selected_session == "Custom New":
        custom_id = st.text_input("Enter Session ID:", value=f"S{int(time.time())%10000:04d}")
        active_session = custom_id
    else:
        active_session = selected_session
        
    if active_session != st.session_state.current_session_id:
        st.session_state.current_session_id = active_session
        if active_session not in st.session_state.messages:
            st.session_state.messages[active_session] = []
        st.rerun()

    st.divider()

    st.subheader("System Telemetry")
    history = st.session_state.orchestrator.memory.get_session_history(st.session_state.current_session_id)
    turn_count = len(history)
    
    col1, col2 = st.columns(2)
    col1.metric("Turn Count", f"{turn_count}")
    
    latest_turn = history[-1] if history else None
    if latest_turn:
        col2.metric("NLU Confidence", f"{latest_turn['confidence_score']:.3f}")
        st.metric("Response Time", f"{latest_turn['response_time_ms']} ms")
        
        # Intent display
        st.write(f"**Current Intent:** `{latest_turn['intent']}`")
        
        # Escalation status
        if latest_turn['escalated'] == 1:
            st.error("🚨 **Status: Escalated to Human Agent**")
        else:
            st.success("🟢 **Status: AI Automated Resolution**")
    else:
        col2.metric("NLU Confidence", "—")
        st.info("Start a conversation to view real-time metrics.")

    st.divider()
    st.subheader("Knowledge Base Status")
    st.caption("Indexed FAQs: **22 Articles across 10 Categories**")
    with st.expander("Explore Supported Categories"):
        categories = ["billing", "technical", "returns", "shipping", "account", 
                      "product_info", "complaint", "cancellation", "security", "feedback"]
        for cat in categories:
            st.markdown(f"• `{cat}`")

# ----------------- MAIN PANEL -----------------
st.markdown('<div class="main-header">LLM-Powered Customer Support Chatbot with Memory</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industry Project 73 | Python, LangChain, RAG Vector Search & Escalation Routing</div>', unsafe_allow_html=True)

# Ethical Compliance Banner
st.markdown("""
<div class="ethical-banner">
    <b>Ethical AI Disclosure:</b> You are interacting with an AI conversational assistant (Nova). 
    This system utilizes conversational memory, automated intent routing, and retrieval-augmented generation. 
    Personal sensitive information is redacted. You may request human agent escalation at any moment by typing <i>"agent"</i> or <i>"supervisor"</i>.
</div>
""", unsafe_allow_html=True)

# Display Messages
chat_container = st.container()
with chat_container:
    curr_messages = st.session_state.messages.get(st.session_state.current_session_id, [])
    for msg in curr_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "intent" in msg:
                badges = f"""
                <div style="margin-top: 8px;">
                    <span class="metric-badge badge-intent">Intent: {msg['intent']}</span>
                    <span class="metric-badge badge-confidence">Confidence: {msg['confidence']:.3f}</span>
                    <span class="metric-badge badge-time">Latency: {msg['latency']} ms</span>
                    <span class="metric-badge {'badge-escalated' if msg['escalated'] else 'badge-confidence'}">
                        {'⚠️ Escalated' if msg['escalated'] else '✓ Resolved by AI'}
                    </span>
                </div>
                """
                st.markdown(badges, unsafe_allow_html=True)

# Chat Input
user_input = st.chat_input("Type your support inquiry here (e.g., 'How do I return my item?' or 'I want a refund')...")

if user_input:
    # Append user turn
    st.session_state.messages[st.session_state.current_session_id].append({
        "role": "user",
        "content": user_input
    })

    # Process through master orchestrator
    with st.spinner("Nova is thinking, searching knowledge base, and checking context..."):
        result = st.session_state.orchestrator.process_message(
            session_id=st.session_state.current_session_id,
            user_query=user_input
        )

    # Append assistant response
    st.session_state.messages[st.session_state.current_session_id].append({
        "role": "assistant",
        "content": result["bot_response"],
        "intent": result["intent"],
        "confidence": result["confidence_score"],
        "escalated": result["escalated"],
        "latency": result["response_time_ms"],
        "turn_id": result["turn_id"],
        "retrieved_context": result.get("retrieved_context", [])
    })

    st.rerun()

# ----------------- RAG Context Inspection & CSAT Rating -----------------
st.divider()
exp_col1, exp_col2 = st.columns([2, 1])

with exp_col1:
    with st.expander("🔍 Inspect Latest RAG Retrieved Context & Similarity Scores"):
        curr_msgs = st.session_state.messages.get(st.session_state.current_session_id, [])
        if curr_msgs and "retrieved_context" in curr_msgs[-1] and curr_msgs[-1]["retrieved_context"]:
            for doc in curr_msgs[-1]["retrieved_context"]:
                st.markdown(f"**FAQ ID:** `{doc['id']}` | **Category:** `{doc['category']}` | **Vector Similarity:** `{doc['similarity_score']:.4f}`")
                st.markdown(f"**Q:** {doc['question']}")
                st.markdown(f"**A:** {doc['answer']}")
                st.write("---")
        else:
            st.caption("No knowledge base query active in this turn.")

with exp_col2:
    with st.expander("⭐ Post-Interaction CSAT Survey"):
        st.write("Rate this conversation session:")
        rating = st.slider("Satisfaction Score (1 to 5):", min_value=1, max_value=5, value=5)
        if st.button("Submit CSAT Feedback"):
            st.session_state.orchestrator.memory.set_satisfaction_score(st.session_state.current_session_id, rating)
            st.success(f"Thank you! CSAT Score {rating}/5 recorded for session {st.session_state.current_session_id}.")
