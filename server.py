"""
FastAPI Server for ParcelAm using Pinecone Assistant
Replaces the previous custom RAG implementation
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ParcelAm Assistant API",
    description="Pinecone Assistant API for ParcelAm customer support",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Configure Assistant
ASSISTANT_NAME = "parcel-assistant"
ASSISTANT_HOST = "https://prod-1-data.ke.pinecone.io/assistant"  # Fixed host


# Pydantic Models
class QueryRequest(BaseModel):
    tenant_id: str = "default"  # Tenant ID for future multi-tenancy
    question: str


class AssistantResponse(BaseModel):
    answer: str
    citations: List[Dict]
    tenant_id: str


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ParcelAm Assistant API",
        "version": "2.0.0",
        "assistant": ASSISTANT_NAME,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check"""
    try:
        # Test connection to Pinecone
        pc.list_indexes()
        return {
            "service": "healthy",
            "pinecone_connected": True,
            "assistant": ASSISTANT_NAME,
            "status": "ready"
        }
    except Exception as e:
        return {
            "service": "healthy",
            "pinecone_connected": False,
            "error": str(e),
            "status": "partial"
        }


@app.post("/query")
async def query_assistant(request: QueryRequest):
    """
    Query the Pinecone Assistant

    Args:
        tenant_id: User/organization ID (currently unused, kept for future multi-tenancy)
        question: User's question
    """
    try:
        # Get assistant instance
        from pinecone_plugins.assistant.models.chat import Message

        # Initialize assistant
        assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)

        # Create message
        message = Message(role="user", content=request.question)

        # Get response from assistant
        response = assistant.chat(messages=[message])

        # Format citations
        citations = []
        if hasattr(response, 'citations') and response.citations:
            for citation in response.citations:
                references = citation.get('references', [])
                for ref in references:
                    citations.append({
                        "file_name": ref.get("file.name", "Unknown"),
                        "file_id": ref.get("file.id", ""),
                        "pages": ref.get("pages", []),
                        "position": citation.get("position", 0),
                        "metadata": ref.get("metadata", {})
                    })

        return AssistantResponse(
            answer=response.message.content,
            citations=citations,
            tenant_id=request.tenant_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant error: {str(e)}")


@app.get("/assistant/status")
async def assistant_status():
    """Get assistant status and configuration"""
    try:
        # List assistants to verify ours exists
        assistants = pc.assistant.list_assistants()

        # Find our assistant
        our_assistant = None
        for asst in assistants:
            if asst.name == ASSISTANT_NAME:
                our_assistant = asst
                break

        if not our_assistant:
            return {
                "status": "not_found",
                "assistant": ASSISTANT_NAME,
                "message": "Assistant not found in your Pinecone account"
            }

        return {
            "status": "ready",
            "assistant": ASSISTANT_NAME,
            "host": ASSISTANT_HOST,
            "region": our_assistant.region,
            "assistant_host": our_assistant.host,
            "message": "Assistant is ready for queries"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/assistant/context")
async def get_context(request: QueryRequest):
    """
    Get relevant context snippets without generating a full response
    Useful for debugging or custom RAG implementations
    """
    try:
        assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)

        response = assistant.context(
            query=request.question,
            top_k=5,
            snippet_size=1024
        )

        # Format context snippets
        snippets = []
        for snippet in response.snippets:
            snippets.append({
                "content": snippet.content,
                "score": snippet.score,
                "file_name": snippet.reference.file.name,
                "file_id": snippet.reference.file.id,
                "pages": snippet.reference.pages
            })

        return {
            "query": request.question,
            "snippets": snippets,
            "tenant_id": request.tenant_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")


# Export assistant info for client apps
@app.get("/assistant/info")
async def assistant_info():
    """Return assistant configuration for frontend integration"""
    return {
        "assistant_name": ASSISTANT_NAME,
        "host": ASSISTANT_HOST,
        "api_version": "v2",
        "features": [
            "Natural language processing",
            "Document citations",
            "Real-time tracking",
            "Delivery estimates",
            "Shipping guidance"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)