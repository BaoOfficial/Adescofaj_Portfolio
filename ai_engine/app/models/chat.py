"""
Pydantic models for chat requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User's message/question", min_length=1)
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for multi-turn conversations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are Amirlahi's Python projects?",
                "conversation_id": "conv_123"
            }
        }


class SourceDocument(BaseModel):
    """Model for source document information."""

    content: str = Field(..., description="Relevant content from source")
    metadata: dict = Field(default_factory=dict, description="Document metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Amirlahi worked as Data Scientist at PlayMode Music...",
                "metadata": {
                    "h2": "Experience",
                    "h3": "PlayMode Music - Data Scientist"
                }
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="AI assistant's response")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used to generate the response"
    )
    conversation_id: str = Field(..., description="Conversation ID")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Amirlahi has worked on several Python projects including...",
                "sources": [
                    {
                        "content": "Customer Segmentation project using K-Means clustering...",
                        "metadata": {"h2": "Projects", "h3": "Customer-Segmentation"}
                    }
                ],
                "conversation_id": "conv_123"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health message")
    vector_store_loaded: bool = Field(..., description="Whether vector store is initialized")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "API is running",
                "vector_store_loaded": True
            }
        }
