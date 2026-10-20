import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    create_user,    
    verify_user
)

# =========================== Utilities ===========================
def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    full_thread_id = f"{st.session_state['user_id']}::{thread_id}"
    st.session_state["thread_id"] = full_thread_id
    add_thread(full_thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])

def get_thread_title(thread_id):
    """Fetches the first 20 characters of the first user message to use as the chat title."""
    try:
        messages = load_conversation(str(thread_id))
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text = str(msg.content).replace("\n", " ").strip()
                if len(text) > 20:
                    return text[:20] + "..."
                elif text:
                    return text
    except Exception:
        pass
    return "New Conversation"

# ======================= Secure Login System ===================
if "user_id" not in st.session_state:
    st.sidebar.title("🔒 Welcome")
    tab1, tab2 = st.sidebar.tabs(["Login", "Sign Up"])
    
    # --- LOGIN TAB ---
    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", use_container_width=True):
            if verify_user(login_user, login_pass):
                clean_name = "".join(c for c in login_user if c.isalnum() or c in ("_", "-")).lower()
                st.session_state["user_id"] = clean_name
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
                
    # --- SIGN UP TAB ---
    with tab2:
        new_user = st.text_input("Choose Username", key="signup_user")
        new_pass = st.text_input("Create Password", type="password", key="signup_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.button("Create Account", use_container_width=True):
            if not new_user or not new_pass:
                st.warning("Please fill all fields.")
            elif new_pass != confirm_pass:
                st.warning("Passwords do not match.")
            elif len(new_pass) < 4:
                st.warning("Password must be at least 4 characters.")
            else:
                success, msg = create_user(new_user, new_pass)
                if success:
                    st.success(msg + " Please log in.")
                else:
                    st.error(msg)
                    
    st.stop()

# If they are logged in, show their name and a logout button
st.sidebar.success(f"Logged in as: **{st.session_state['user_id']}**")
if st.sidebar.button("Logout", use_container_width=True):
    del st.session_state["user_id"]
    st.rerun()

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"{st.session_state['user_id']}::{generate_thread_id()}"

if "chat_threads" not in st.session_state:
    # Fetch all threads, but ONLY keep the ones that belong to this user
    all_db_threads = retrieve_all_threads()
    my_threads = [t for t in all_db_threads if t.startswith(f"{st.session_state['user_id']}::")]
    st.session_state["chat_threads"] = my_threads

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "pending_thread" not in st.session_state:
    st.session_state["pending_thread"] = None

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]

# ============================ Sidebar (Chat UI) ============================
st.sidebar.divider()
st.sidebar.title("LangGraph PDF Chatbot")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Your Past Conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        display_key = thread_id.split("::")[-1]
        title = get_thread_title(thread_id)
        
        if st.sidebar.button(
            title, 
            key=f"side-thread-{display_key}", 
            use_container_width=True
        ):
            st.session_state["pending_thread"] = thread_id


# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

# ================= Past conversation loader =================
if st.session_state["pending_thread"]:
    selected_thread = str(st.session_state["pending_thread"])
    st.session_state["pending_thread"] = None

    if selected_thread != thread_key:
        st.session_state["thread_id"] = selected_thread
        messages = load_conversation(selected_thread)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages
        st.session_state["ingested_docs"].setdefault(selected_thread, {})
        st.rerun()