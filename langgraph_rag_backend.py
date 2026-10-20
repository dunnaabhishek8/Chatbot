from __future__ import annotations

import os
import tempfile
import time
from typing import Annotated, Any, Dict, Optional, TypedDict

import psycopg
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun  # <-- Fixed import
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    thread_id = str(thread_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        # --- Rate Limit Handling for Gemini Free Tier ---
        batch_size = 90
        vector_store = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
                
            if i + batch_size < len(chunks):
                time.sleep(2)  # Pause to avoid 429 Too Many Requests
        # ------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return dict(_THREAD_METADATA[thread_id])
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not configured."}

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread."""
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. Database & Graph Compilation
# -------------------
DB_URI = os.getenv("SUPABASE_DB_URI")
if not DB_URI:
    raise ValueError("SUPABASE_DB_URI environment variable is not set.")

conn = psycopg.connect(DB_URI, autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helpers
# -------------------
def retrieve_all_threads() -> list[str]:
    all_threads: set[str] = set()
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config["configurable"]["thread_id"]
        if tid is not None:
            all_threads.add(str(tid))
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})