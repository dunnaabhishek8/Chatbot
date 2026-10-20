from __future__ import annotations

import hmac
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_rag_backend import (
    chatbot,
    ingest_pdf,
    thread_document_metadata,
)


# ============================================================
# Configuration
# ============================================================

APP_NAME = "LangGraph PDF Chatbot API"
API_PREFIX = "/api/v1"

MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "15"))
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
MAX_MESSAGE_LENGTH = 10_000
MAX_THREAD_ID_LENGTH = 128

API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "").strip()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("chatbot.api")

bearer_scheme = HTTPBearer(auto_error=False)

# In-process locks only. These do not coordinate separate server processes.
_thread_locks: dict[str, Lock] = {}
_thread_locks_guard = Lock()


def get_thread_lock(thread_id: str) -> Lock:
    with _thread_locks_guard:
        if thread_id not in _thread_locks:
            _thread_locks[thread_id] = Lock()
        return _thread_locks[thread_id]


# ============================================================
# Request / response schemas
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )
    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_THREAD_ID_LENGTH,
    )


class ChatResponse(BaseModel):
    thread_id: str
    answer: str


class DocumentResponse(BaseModel):
    thread_id: str
    filename: str
    pages: int
    chunks: int
    status: str


class DocumentStatusResponse(BaseModel):
    thread_id: str
    document_indexed: bool
    filename: str | None = None
    pages: int | None = None
    chunks: int | None = None


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str


# ============================================================
# Application lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_ACCESS_TOKEN:
        logger.warning(
            "API_ACCESS_TOKEN is not configured. "
            "Protected endpoints will reject requests."
        )

    logger.info("Starting %s", APP_NAME)
    yield
    logger.info("Stopping %s", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="HTTP API for the existing LangGraph PDF chatbot.",
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# Security
# ============================================================

async def require_api_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(bearer_scheme),
    ],
) -> None:
    if not API_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(
            credentials.credentials,
            API_ACCESS_TOKEN,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


Protected = Annotated[None, Depends(require_api_token)]


# ============================================================
# Validation and utility functions
# ============================================================

def validate_thread_id(thread_id: str) -> str:
    # Permit common UUIDs and simple application-generated thread IDs.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", thread_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid thread_id format.",
        )
    return thread_id


def extract_answer(messages: list[Any]) -> str:
    # Search backwards for the final AI message, skipping tool-call messages.
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content

            if isinstance(content, str):
                return content

            # Handle providers that return structured content blocks.
            if isinstance(content, list):
                text_parts = [
                    item["text"]
                    for item in content
                    if isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                ]
                if text_parts:
                    return "\n".join(text_parts)

    return "I couldn't generate a final answer. Please try again."


def chat_sync(message: str, thread_id: str) -> str:
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "fastapi_chat_turn",
    }

    # Serialize operations for the same thread within this process.
    with get_thread_lock(thread_id):
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

    return extract_answer(result.get("messages", []))


def ingest_sync(
    pdf_bytes: bytes,
    thread_id: str,
    filename: str,
) -> dict[str, Any]:
    with get_thread_lock(thread_id):
        return ingest_pdf(
            file_bytes=pdf_bytes,
            thread_id=thread_id,
            filename=filename,
        )


# ============================================================
# Request logging
# ============================================================

@app.middleware("http")
async def request_logging(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "Unhandled request error method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


# ============================================================
# Health endpoints
# ============================================================

@app.get("/health/live", response_model=HealthResponse, tags=["Health"])
async def liveness():
    return HealthResponse(status="alive")


@app.get("/health/ready", response_model=HealthResponse, tags=["Health"])
async def readiness():
    # This checks application configuration, not external provider health.
    if not API_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    return HealthResponse(status="ready")


# ============================================================
# Chat endpoint
# ============================================================

@app.post(
    f"{API_PREFIX}/chat",
    response_model=ChatResponse,
    responses={401: {"model": ErrorResponse}},
    tags=["Chat"],
)
async def chat(
    payload: ChatRequest,
    _: Protected,
):
    thread_id = validate_thread_id(payload.thread_id)
    message = payload.message.strip()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be blank.",
        )

    try:
        answer = await run_in_threadpool(
            chat_sync,
            message,
            thread_id,
        )

        return ChatResponse(
            thread_id=thread_id,
            answer=answer,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat processing failed for thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat processing failed. Please try again.",
        )


# ============================================================
# PDF upload endpoint
# ============================================================

@app.post(
    f"{API_PREFIX}/documents",
    response_model=DocumentResponse,
    tags=["Documents"],
)
async def upload_document(
    _: Protected,
    thread_id: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
):
    thread_id = validate_thread_id(thread_id)

    filename = (file.filename or "").replace("\\", "/").split("/")[-1]

    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    if file.content_type not in (None, "", "application/pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/pdf.",
        )

    pdf_bytes = bytearray()

    try:
        # Read incrementally to enforce an application-level size limit.
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break

            pdf_bytes.extend(chunk)

            if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"PDF exceeds the {MAX_PDF_SIZE_MB} MB limit.",
                )

        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF is empty.",
            )

        # Basic PDF signature check; the PDF parser must still validate it.
        if not bytes(pdf_bytes).lstrip().startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file does not appear to be a valid PDF.",
            )

        metadata = await run_in_threadpool(
            ingest_sync,
            bytes(pdf_bytes),
            thread_id,
            filename,
        )

        return DocumentResponse(
            thread_id=thread_id,
            filename=metadata["filename"],
            pages=metadata["documents"],
            chunks=metadata["chunks"],
            status="indexed",
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "PDF ingestion failed for thread_id=%s filename=%s",
            thread_id,
            filename,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PDF ingestion failed. Please check the file and retry.",
        )
    finally:
        await file.close()


# ============================================================
# Document metadata endpoint
# ============================================================

@app.get(
    f"{API_PREFIX}/documents/{{thread_id}}",
    response_model=DocumentStatusResponse,
    tags=["Documents"],
)
async def get_document_status(
    thread_id: str,
    _: Protected,
):
    thread_id = validate_thread_id(thread_id)

    try:
        metadata = await run_in_threadpool(
            thread_document_metadata,
            thread_id,
        )

        if not metadata:
            return DocumentStatusResponse(
                thread_id=thread_id,
                document_indexed=False,
            )

        return DocumentStatusResponse(
            thread_id=thread_id,
            document_indexed=True,
            filename=metadata.get("filename"),
            pages=metadata.get("documents"),
            chunks=metadata.get("chunks"),
        )

    except Exception:
        logger.exception(
            "Document metadata lookup failed for thread_id=%s",
            thread_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve document status.",
        )