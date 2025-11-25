"""
Main FastAPI application for the AI chatbot backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat
from app.config import settings

# Create FastAPI app
app = FastAPI(
    title="Amirlahi Portfolio Chatbot API",
    description="AI-powered chatbot with RAG for Amirlahi's portfolio website",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:8080", "http://localhost:8081", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Amirlahi Portfolio Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("="*60)
    print("Starting Amirlahi Portfolio Chatbot API")
    print("="*60)
    print(f"Model: {settings.model_name}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Vector Store: {settings.chroma_persist_dir}")
    print("="*60)

    # Initialize RAG service (loads vector store)
    from app.services.rag_service import get_rag_service
    try:
        rag_service = get_rag_service()
        if rag_service.is_ready():
            print("[OK] RAG Service initialized successfully!")
        else:
            print("[WARNING] Vector store not loaded. Run build_embeddings.py first.")
    except Exception as e:
        print(f"[ERROR] Error initializing RAG service: {str(e)}")
        print("[WARNING] API will start but chat functionality may not work.")

    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down Amirlahi Portfolio Chatbot API...")
