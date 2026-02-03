"""
FastAPI Server for ParcelAm RAG Service
Deploy to: Render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

from rag_service import MultiTenantRAG, SAMPLE_PARCEL_DOCUMENTS

# Initialize FastAPI
app = FastAPI(
    title="ParcelAm RAG API",
    description="Pinecone-powered semantic search for ParcelAm",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service (lazy loading)
rag = None
rag_init_error = None
rag_initialized = False


def get_rag_service():
    """Lazy initialization of RAG service"""
    global rag, rag_init_error, rag_initialized
    if not rag_initialized:
        try:
            if not rag:
                rag = MultiTenantRAG()
            rag_initialized = True
            print("✅ RAG service initialized (lazy)")
        except Exception as e:
            rag_init_error = str(e)
            print(f"❌ Error initializing RAG: {e}")
            raise
    return rag


@app.on_event("startup")
async def startup():
    """Startup handler - server is ready immediately"""
    print("✅ Server started - RAG will initialize on first request")


# ============ Pydantic Models ============

class QueryRequest(BaseModel):
    tenant_id: str
    question: str
    filter: Optional[Dict] = None


class Document(BaseModel):
    _id: str
    content: str
    title: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict] = None


class IndexRequest(BaseModel):
    tenant_id: str
    documents: List[Document]


# ============ API Endpoints ============

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ParcelAm RAG API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check - initializes RAG to keep service warm"""
    try:
        # Trigger RAG initialization to keep service warm
        get_rag_service()
    except Exception as e:
        pass  # Don't fail health check if RAG init fails

    status = {
        "service": "healthy",
        "rag_initialized": rag_initialized,
        "server": "ready"
    }

    if rag_init_error:
        status["rag_error"] = rag_init_error

    if rag_initialized and rag:
        try:
            indexes = rag.pc.list_indexes()
            status["pinecone_connected"] = True
            status["index_count"] = len(indexes.indexes)
        except Exception as e:
            status["pinecone_connected"] = False
            status["error"] = str(e)

    return status


@app.get("/status")
async def status():
    """Detailed service status"""
    return {
        "service": "ParcelAm RAG API",
        "version": "1.0.0",
        "rag_initialized": rag_initialized,
        "rag_error": rag_init_error,
        "message": "RAG service initializes on first request" if not rag_initialized else "RAG service ready"
    }


@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    Query the RAG system

    Args:
        tenant_id: User/organization ID
        question: User's question
        filter: Optional metadata filter
    """
    try:
        rag_service = get_rag_service()
        result = rag_service.query(
            tenant_id=request.tenant_id,
            question=request.question,
            filter=request.filter
        )

        return {
            "answer": result.answer,
            "sources": result.sources,
            "tenant_id": result.tenant_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_documents(request: IndexRequest):
    """
    Index documents for a tenant

    Args:
        tenant_id: User/organization ID
        documents: List of documents to index
    """
    try:
        rag_service = get_rag_service()
        docs = [doc.model_dump() for doc in request.documents]
        count = rag_service.index_bulk_documents(
            tenant_id=request.tenant_id,
            documents=docs
        )

        return {
            "success": True,
            "indexed": count,
            "tenant_id": request.tenant_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-sample")
async def index_sample_data(tenant_id: str):
    """
    Index sample parcel documents (for testing)

    Args:
        tenant_id: Tenant ID to index documents for
    """
    try:
        rag_service = get_rag_service()
        count = rag_service.index_bulk_documents(
            tenant_id=tenant_id,
            documents=SAMPLE_PARCEL_DOCUMENTS
        )

        return {
            "success": True,
            "indexed": count,
            "tenant_id": tenant_id,
            "message": "Sample documents indexed. Wait 10 seconds before querying."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/tenant/{tenant_id}")
async def delete_tenant(tenant_id: str):
    """Delete all data for a tenant"""
    try:
        rag_service = get_rag_service()
        success = rag_service.delete_tenant_data(tenant_id)
        return {
            "success": success,
            "tenant_id": tenant_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
