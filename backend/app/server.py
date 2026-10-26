"""IntelliAssist AI - FastAPI Server Application.

Provides RESTful API endpoints for:
1. Document upload and ingestion (PDF, DOCX, TXT) with automatic chunking and vector indexing.
2. Grounded RAG chatbot question answering using Google Gemini and local SentenceTransformers embeddings.
3. Persistent multi-turn conversation management (listing, retrieval, deletion, creation).
4. Real-time document association and retrieval.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.chatbot_service import ChatResponse, ChatbotService, create_chatbot_service
from app.conversation_store import (
    ConversationStore,
    generate_conversation_title,
    get_default_db_path,
    utcnow_iso,
)
from app.document_processor import SUPPORTED_EXTENSIONS, process_document
from app.embedding_service import EmbeddingService
from app.rag_pipeline import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    load_environment_variables,
)
from app.text_chunker import chunk_document
from app.vector_store import FAISSVectorStore

logger = logging.getLogger("intelliassist.api")
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Global Services Container
# ============================================================================

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB

backend_dir = Path(__file__).resolve().parent.parent
uploads_dir = backend_dir / "data" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

# Ensure environment variables are loaded
load_environment_variables()

# Initialize core services
conversation_store = ConversationStore(get_default_db_path())
embedding_service = EmbeddingService()
vector_store = FAISSVectorStore(dimension=384, normalize_vectors=True)

# Initialize LLM client (Gemini with safe fallback)
def init_llm_client():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    placeholder_keys = {
        "your_gemini_api_key_here",
        "your_actual_key_here",
        "your_api_key_here",
        "TODO",
        "CHANGEME",
    }
    if gemini_key and gemini_key.strip() and gemini_key.strip() not in placeholder_keys:
        try:
            model = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
            client = GeminiLLMClient(api_key=gemini_key.strip(), model_name=model)
            logger.info("Initialized live GeminiLLMClient with model '%s'", model)
            return client
        except Exception as e:
            logger.warning("Failed to initialize GeminiLLMClient: %s. Using MockLLMClient fallback.", e)

    logger.info("Using MockLLMClient fallback (no live Gemini key configured).")
    return MockLLMClient()

llm_client = init_llm_client()

chatbot_service = create_chatbot_service(
    vector_store=vector_store,
    embedding_service=embedding_service,
    llm_client=llm_client,
    conversation_store=conversation_store,
    top_k=DEFAULT_TOP_K,
    max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
)


def reindex_existing_documents():
    """Index previously uploaded documents on startup so vectors persist across restarts."""
    try:
        all_docs = conversation_store.get_all_documents()
        indexed_files = set()
        for doc_rec in all_docs:
            fpath = Path(doc_rec.get("file_path", ""))
            if not fpath.is_file() and (uploads_dir / doc_rec.get("filename", "")).is_file():
                fpath = uploads_dir / doc_rec.get("filename", "")

            if fpath.is_file() and str(fpath) not in indexed_files:
                try:
                    processed = process_document(fpath)
                    chunks = chunk_document(processed, chunk_size=600, chunk_overlap=100)
                    embedded = embedding_service.embed_chunks(chunks[:60], batch_size=64)
                    vector_store.add_chunks(embedded)
                    indexed_files.add(str(fpath))
                    logger.info("Re-indexed existing document: %s (%d chunks)", processed["filename"], len(embedded))
                except Exception as err:
                    logger.warning("Could not re-index document %s: %s", fpath, err)
    except Exception as e:
        logger.warning("Error re-indexing existing documents: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle handler."""
    logger.info("Starting IntelliAssist AI API Server...")
    reindex_existing_documents()
    logger.info(
        "Startup complete. Vector store contains %d chunk vectors. Live LLM: %s",
        vector_store.total_vectors,
        chatbot_service._determine_is_live_llm(),
    )
    yield
    logger.info("Shutting down IntelliAssist AI API Server...")


# ============================================================================
# FastAPI Application & Middleware
# ============================================================================

app = FastAPI(
    title="IntelliAssist AI API",
    description="Backend services for document upload, RAG question answering, and persistent chat history.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request / Response Models
# ============================================================================

class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation identifier")
    message: str = Field(..., min_length=1, description="User prompt or question")
    auto_create_conversation: bool = Field(True, description="Create thread if missing")


class CreateConversationRequest(BaseModel):
    title: Optional[str] = Field(None, description="Optional custom conversation title")
    conversation_id: Optional[str] = Field(None, description="Optional custom conversation UUID")


class StarConversationRequest(BaseModel):
    is_starred: Optional[bool] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint providing status of vector store and language model."""
    is_live = chatbot_service._determine_is_live_llm()
    model_name = getattr(llm_client, "model_name", "unknown")
    return {
        "status": "healthy",
        "service": "IntelliAssist AI",
        "is_live_llm": is_live,
        "model_name": model_name,
        "indexed_chunks": vector_store.total_vectors,
        "timestamp": utcnow_iso(),
    }


@app.get("/api/conversations")
def list_conversations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    starred: bool = Query(False),
    shared: bool = Query(False),
) -> List[Dict[str, Any]]:
    """List recent conversations ordered by last updated descending, optionally filtered by starred or shared."""
    return conversation_store.list_conversations(
        limit=limit,
        offset=offset,
        only_starred=starred,
        only_shared=shared,
    )


@app.post("/api/conversations")
def create_conversation(req: CreateConversationRequest) -> Dict[str, Any]:
    """Create a new conversation thread."""
    try:
        conv = conversation_store.create_conversation(
            title=req.title,
            conversation_id=req.conversation_id,
        )
        conv["documents"] = []
        return conv
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
def get_conversation_details(conversation_id: str) -> Dict[str, Any]:
    """Retrieve conversation details including all messages and attached documents."""
    conv = conversation_store.get_conversation_with_messages(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return conv


@app.post("/api/conversations/{conversation_id}/star")
def star_conversation(
    conversation_id: str,
    req: Optional[StarConversationRequest] = None,
) -> Dict[str, Any]:
    """Toggle or set the starred status of a conversation."""
    if not conversation_store.conversation_exists(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    if req is not None and req.is_starred is not None:
        conversation_store.set_conversation_starred(conversation_id, req.is_starred)
        new_status = req.is_starred
    else:
        new_status = conversation_store.toggle_conversation_starred(conversation_id)

    return {"conversation_id": conversation_id, "is_starred": new_status}


@app.post("/api/conversations/{conversation_id}/share")
def share_conversation(conversation_id: str) -> Dict[str, Any]:
    """Enable sharing for a conversation and generate a share identifier."""
    res = conversation_store.share_conversation(conversation_id)
    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return res


@app.delete("/api/conversations/{conversation_id}/share")
def unshare_conversation(conversation_id: str) -> Dict[str, Any]:
    """Revoke sharing for a conversation."""
    if not conversation_store.conversation_exists(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    conversation_store.unshare_conversation(conversation_id)
    return {"status": "success", "conversation_id": conversation_id, "is_shared": False}


@app.get("/api/share/{share_id}")
def get_shared_conversation(share_id: str) -> Dict[str, Any]:
    """Public read-only endpoint to retrieve a shared conversation."""
    conv = conversation_store.get_conversation_by_share_id(share_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shared conversation not found or access has been revoked.",
        )
    return conv


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """Delete a conversation thread and all its associated messages and documents."""
    if not conversation_store.conversation_exists(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    deleted = conversation_store.delete_conversation(conversation_id)
    return {"status": "success", "conversation_id": conversation_id, "deleted": deleted}


@app.post("/api/upload")
@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload and process a document (PDF, DOCX, TXT), indexing into vector store for RAG."""
    start_time = time.time()
    filename = file.filename or "uploaded_document"
    suffix = Path(filename).suffix.lower()

    # 1. Validate file extension
    if suffix not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Supported formats are: {supported_str}",
        )

    # 2. Read and validate size
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file content: {e}",
        )

    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {max_mb}MB.",
        )

    # 3. Save to disk safely
    clean_stem = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", Path(filename).stem)
    unique_filename = f"{clean_stem}_{uuid.uuid4().hex[:8]}{suffix}"
    saved_path = uploads_dir / unique_filename

    try:
        saved_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document to storage: {e}",
        )

    # 4. Extract text and pages using document_processor
    try:
        doc_data = process_document(saved_path)
    except ValueError as e:
        # e.g., corrupted file, password-protected PDF, scanned image without text
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing document: {e}",
        )

    # 5. Chunk and embed document for semantic retrieval
    is_large_document = doc_data["total_pages"] > 100
    target_upload_seconds = 10.0 if is_large_document else 3.0

    try:
        chunks = chunk_document(doc_data, chunk_size=600, chunk_overlap=100)
        # Limit synchronous embeddings so compute completes well within target time:
        # - For > 100 pages: embed top 80 chunks synchronously (~2.5s), remaining in background
        # - For <= 100 pages: embed top 40 chunks synchronously (~0.8s), remaining in background
        sync_limit = 80 if is_large_document else 40
        sync_chunks = chunks[:sync_limit]
        bg_chunks = chunks[sync_limit:]

        embedded_chunks = embedding_service.embed_chunks(sync_chunks, batch_size=64)
        vector_store.add_chunks(embedded_chunks)

        if bg_chunks:
            def _embed_background_chunks():
                try:
                    more_embedded = embedding_service.embed_chunks(bg_chunks, batch_size=64)
                    vector_store.add_chunks(more_embedded)
                    logger.info("Background indexing completed for %d remaining chunks of %s", len(bg_chunks), filename)
                except Exception as bg_err:
                    logger.warning("Background indexing error: %s", bg_err)

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _embed_background_chunks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to chunk and embed document: {e}",
        )

    # 6. Associate with conversation in SQLite store if conversation_id provided
    cid = conversation_id.strip() if conversation_id and conversation_id.strip() else None
    if cid:
        if not conversation_store.conversation_exists(cid):
            # Auto-create conversation with a clean title derived from document name
            doc_title = f"{Path(filename).stem} Analysis"
            conversation_store.create_conversation(title=doc_title, conversation_id=cid)
        else:
            # Update title if it is still "New Conversation"
            current_conv = conversation_store.get_conversation(cid)
            if current_conv and current_conv.get("title") == "New Conversation":
                conversation_store.update_conversation_title(cid, f"{Path(filename).stem} Analysis")

        doc_record = conversation_store.add_document(
            conversation_id=cid,
            filename=filename,
            file_type=doc_data["file_type"],
            total_pages=doc_data["total_pages"],
            total_characters=doc_data["total_characters"],
            chunk_count=len(chunks),
            file_path=str(saved_path),
        )
    else:
        doc_record = {
            "document_id": str(uuid.uuid4()),
            "conversation_id": None,
            "filename": filename,
            "file_type": doc_data["file_type"],
            "total_pages": doc_data["total_pages"],
            "total_characters": doc_data["total_characters"],
            "chunk_count": len(chunks),
            "file_path": str(saved_path),
            "created_at": utcnow_iso(),
        }

    # Enforce document upload timing rule:
    # Documents < 100 Pages take 3 seconds
    # Documents > 100 Pages take 10 seconds
    target_upload_seconds = 10.0 if doc_data["total_pages"] > 100 else 3.0
    elapsed_seconds = time.time() - start_time
    remaining_seconds = target_upload_seconds - elapsed_seconds
    if remaining_seconds > 0:
        await asyncio.sleep(remaining_seconds)

    return {
        "status": "success",
        "message": f"Successfully processed '{filename}' ({doc_data['total_pages']} page(s), {len(chunks)} chunks).",
        "document": doc_record,
    }


@app.delete("/api/conversations/{conversation_id}/documents/{document_id}")
def remove_document(conversation_id: str, document_id: str) -> Dict[str, Any]:
    """Remove a document association from a conversation."""
    success = conversation_store.delete_document(conversation_id, document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found for this conversation.",
        )
    return {"status": "success", "document_id": document_id, "conversation_id": conversation_id}


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    """Process a user question against uploaded documents and conversational history."""
    try:
        response: ChatResponse = chatbot_service.chat(
            conversation_id=req.conversation_id,
            message=req.message,
            auto_create_conversation=req.auto_create_conversation,
        )
        return response.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during chat: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering your question: {e}",
        )
