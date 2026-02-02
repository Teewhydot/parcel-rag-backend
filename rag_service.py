"""
Multi-Tenant RAG Service for ParcelAm
Pinecone-powered semantic search with reranking
"""

import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Configuration
INDEX_NAME = "parcel-rag-index"
RERANK_MODEL = "bge-reranker-v2-m3"
TOP_K_RESULTS = 10
FINAL_TOP_K = 5


@dataclass
class RAGResult:
    """RAG query result"""
    answer: str
    sources: List[Dict]
    tenant_id: str


class MultiTenantRAG:
    """Multi-tenant RAG system with namespace isolation"""

    def __init__(self, index_name: str = INDEX_NAME):
        self.index_name = index_name
        self.index = pc.Index(index_name)

    def index_document(
        self,
        tenant_id: str,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Index a single document for a tenant"""
        try:
            namespace = tenant_id
            record = {
                "_id": doc_id,
                "content": content,
                **(metadata or {})
            }

            self.index.upsert_records(namespace=namespace, records=[record])
            return True
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False

    def index_bulk_documents(
        self,
        tenant_id: str,
        documents: List[Dict],
        batch_size: int = 96
    ) -> int:
        """Bulk index documents for a tenant"""
        try:
            namespace = tenant_id

            records = []
            for doc in documents:
                record = {
                    "_id": doc["_id"],
                    "content": doc["content"],
                    **{k: v for k, v in doc.items() if k not in ["_id", "content"]}
                }
                records.append(record)

            # Batch upsert
            success_count = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.index.upsert_records(namespace=namespace, records=batch)
                success_count += len(batch)

            return success_count
        except Exception as e:
            print(f"Error bulk indexing: {e}")
            return 0

    def query(
        self,
        tenant_id: str,
        question: str,
        filter: Optional[Dict] = None,
        top_k: int = FINAL_TOP_K
    ) -> RAGResult:
        """
        Complete RAG query: search with reranking

        Args:
            tenant_id: Tenant to query
            question: User question
            filter: Optional metadata filter
            top_k: Number of results to return

        Returns:
            RAGResult with sources
        """
        try:
            namespace = tenant_id

            # Semantic search with integrated reranking
            results = self.index.search(
                namespace=namespace,
                query={
                    "inputs": {
                        "text": question
                    },
                    "top_k": top_k * 2,  # Fetch more for reranking
                    "filter": filter or {}
                },
                rerank={
                    "model": RERANK_MODEL,
                    "top_n": top_k,
                    "rank_fields": ["content"]
                }
            )

            hits = results.get("result", {}).get("hits", [])

            # Extract sources
            sources = []
            for hit in hits[:top_k]:
                fields = hit.get("fields", {})
                sources.append({
                    "id": hit.get("_id"),
                    "title": fields.get("title", "Untitled"),
                    "content": fields.get("content", ""),
                    "score": hit.get("_score", 0),
                    "metadata": {k: v for k, v in fields.items() if k not in ["content", "title"]}
                })

            # Build context answer (no LLM)
            answer = self._build_context_answer(sources)

            return RAGResult(
                answer=answer,
                sources=sources,
                tenant_id=tenant_id
            )

        except Exception as e:
            print(f"Query error: {e}")
            return RAGResult(answer="", sources=[], tenant_id=tenant_id)

    def _build_context_answer(self, sources: List[Dict]) -> str:
        """Build answer from context sources"""
        if not sources:
            return "I couldn't find relevant information to answer your question."

        answer = "Based on the documentation:\n\n"
        for i, source in enumerate(sources[:3], 1):
            title = source.get("title", "Document")
            content = source.get("content", "")
            score = source.get("score", 0)

            answer += f"{i}. {title} (Relevance: {score:.1%})\n"
            answer += f"{content[:300]}...\n\n"

        return answer

    def delete_tenant_data(self, tenant_id: str) -> bool:
        """Delete all data for a tenant"""
        try:
            self.index.delete(delete_all=True, namespace=tenant_id)
            return True
        except Exception as e:
            print(f"Error deleting tenant data: {e}")
            return False


# Sample documents for testing
SAMPLE_PARCEL_DOCUMENTS = [
    {
        "_id": "doc1",
        "content": "Parcel tracking allows customers to monitor their shipment in real-time. Enter your tracking number on the dashboard to see current location, delivery status, and estimated arrival time. Tracking updates are provided at each checkpoint including pickup, transit, and delivery.",
        "category": "tracking",
        "title": "How to Track Your Parcel"
    },
    {
        "_id": "doc2",
        "content": "Package delivery times vary by service level: Standard (5-7 business days), Express (2-3 business days), and Same-Day (for local deliveries within the same city). Delivery times exclude weekends and holidays. Remote areas may require additional time.",
        "category": "shipping",
        "title": "Delivery Time Estimates"
    },
    {
        "_id": "doc3",
        "content": "To create a shipping label, log into your account and navigate to 'Create Shipment'. Enter recipient address, package dimensions, weight, and select service level. Payment is processed automatically using your saved payment method. The label can be printed or downloaded as PDF.",
        "category": "shipping",
        "title": "Creating Shipping Labels"
    },
    {
        "_id": "doc4",
        "content": "If your package shows 'delivered' but you haven't received it, wait 24 hours as it may have been left with a neighbor or in a safe location. Check your delivery confirmation email for specific delivery instructions. If still not found, contact support within 7 days.",
        "category": "support",
        "title": "Missing Package Resolution"
    },
    {
        "_id": "doc5",
        "content": "International shipping requires customs declaration forms. Prohibited items include firearms, hazardous materials, perishable goods, and certain electronics. Duties and taxes may apply and are typically the recipient's responsibility. Check country-specific restrictions before shipping.",
        "category": "international",
        "title": "International Shipping Guide"
    },
    {
        "_id": "doc6",
        "content": "Package insurance is available for valuable items. Standard coverage includes up to $100 for Express shipments and $50 for Standard. Additional insurance can be purchased at checkout. To file a claim, provide proof of value and photos of damaged items within 14 days of delivery.",
        "category": "insurance",
        "title": "Package Insurance and Claims"
    },
    {
        "_id": "doc7",
        "content": "Business accounts offer bulk shipping discounts, API integration, monthly billing, and dedicated account management. Features include address book management, shipping analytics, and multi-user support. Apply online with business documentation for approval.",
        "category": "business",
        "title": "Business Account Benefits"
    },
    {
        "_id": "doc8",
        "content": "Parcel pickup can be scheduled through the app or website. Standard pickup is free for shipments over $50. Same-day pickup requires scheduling before 2 PM. Couriers will collect from your specified address during the chosen time window. Have packages ready and labeled.",
        "category": "pickup",
        "title": "Scheduling Package Pickup"
    }
]
