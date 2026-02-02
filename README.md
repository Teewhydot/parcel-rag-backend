# ParcelAm RAG Backend

Pinecone-powered semantic search service for ParcelAm AI chat.

## Features

- Multi-tenant semantic search
- Pinecone vector database with reranking
- FastAPI REST API
- Deploy to Render.com

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY

# Run server
uvicorn server:app --reload
```

### Deploy to Render

1. Create GitHub repo and push code
2. Go to render.com → New Web Service
3. Connect repo
4. Configure:
   - Runtime: Python 3
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Add env var: `PINECONE_API_KEY=your_key`
6. Deploy!

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check
- `POST /query` - Query RAG system
- `POST /index` - Index documents
- `POST /index-sample` - Index sample data (testing)
- `DELETE /tenant/{id}` - Delete tenant data

## Example Usage

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "user123",
    "question": "How do I track my package?"
  }'
```

### Index Sample Data

```bash
curl -X POST http://localhost:8000/index-sample?tenant_id=user123
```

## Integration with Flutter

In your Flutter app, call this API instead of using keyword-based knowledge base.

Replace:
```dart
ParcelKnowledgeBase.searchContext(content)
```

With:
```dart
final response = await http.post(
  'https://your-service.onrender.com/query',
  body: jsonEncode({
    'tenant_id': userId,
    'question': content,
  }),
);
```

## Architecture

```
Flutter → Render (Python/FastAPI) → Pinecone → Results
```

## License

Part of ParcelAm project.
