"""
Chat API endpoints.
"""

from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, SourceDocument, HealthResponse
from app.services.rag_service import get_rag_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG-powered responses.

    Args:
        request: Chat request with message and optional conversation_id

    Returns:
        Chat response with AI-generated answer and source documents
    """
    try:
        # Get RAG service
        rag_service = get_rag_service()

        # Generate response
        response, docs, conversation_id = rag_service.generate_response(
            query=request.message,
            conversation_id=request.conversation_id
        )

        # Format source documents
        sources = [
            SourceDocument(
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                metadata=doc.metadata
            )
            for doc in docs
        ]

        return ChatResponse(
            response=response,
            sources=sources,
            conversation_id=conversation_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and vector store status
    """
    try:
        rag_service = get_rag_service()
        vector_store_loaded = rag_service.is_ready()

        return HealthResponse(
            status="healthy" if vector_store_loaded else "degraded",
            message="API is running" if vector_store_loaded else "Vector store not loaded",
            vector_store_loaded=vector_store_loaded
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Error: {str(e)}",
            vector_store_loaded=False
        )


@router.post("/reset-conversation")
async def reset_conversation(conversation_id: str):
    """
    Reset/clear a conversation's history.

    Args:
        conversation_id: ID of conversation to reset

    Returns:
        Success message
    """
    try:
        rag_service = get_rag_service()

        if conversation_id in rag_service.conversations:
            del rag_service.conversations[conversation_id]

        return {"message": "Conversation reset successfully", "conversation_id": conversation_id}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting conversation: {str(e)}"
        )
