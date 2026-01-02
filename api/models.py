"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The question to ask")
    k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")
    num_variations: int = Field(default=2, ge=0, le=5, description="Number of query variations for rephrasing")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")

