# ParcelAm Assistant API

Updated to use **Pinecone Assistant** for better RAG capabilities!

## What's Changed

- ✅ **Simplified RAG**: No more custom indexing or embedding management
- ✅ **Automatic Processing**: Documents are automatically chunked and embedded
- ✅ **Cited Responses**: Every answer includes source references
- ✅ **Better Accuracy**: Improved search with Pinecone's built-in intelligence
- ✅ **Less Code**: Cleaner, more maintainable implementation

## Quick Start

### Local Development

```bash
# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
# Ensure PINECONE_API_KEY is set in .env

# Run server
python server.py
```

### Deploy to Render

1. Update your Render service to use `server.py`
2. Configure:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
3. Add env vars:
   - `PINECONE_API_KEY=your_key`
   - `PINECONE_ASSISTANT_HOST=https://prod-1-data.ke.pinecone.io/assistant`

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check
- `POST /query` - Ask questions (returns answers with citations)
- `GET /assistant/status` - Check assistant status
- `GET /assistant/context` - Get raw context snippets
- `GET /assistant/info` - Assistant configuration for frontend

## Example Usage

### Query Assistant

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "question": "How do I track my package?"
  }'
```

### Assistant Status

```bash
curl http://localhost:8000/assistant/status
```

## Integration with Flutter

In your Flutter app, the API call remains the same:

```dart
final response = await http.post(
  'https://your-service.onrender.com/query',
  body: jsonEncode({
    'tenant_id': userId,
    'question': content,
  }),
);

// Parse response with citations
final answer = response.body['answer'];
final citations = response.body['citations']; // Source references
```

## Architecture

```
Flutter → Render (Python/FastAPI) → Pinecone Assistant → Intelligent Responses
```

## Sample Documents

The assistant has been uploaded with 8 comprehensive documents covering:
- Package tracking
- Delivery times
- Shipping labels
- Missing packages
- International shipping
- Insurance coverage
- Business accounts
- Pickup scheduling

## Why Pinecone Assistant is Better

### Before (Custom Implementation):
- Manual document indexing
- Custom chunking logic
- Built-in reranking
- Multi-tenant management
- 200+ lines of RAG code

### Now (Pinecone Assistant):
- Automatic document processing
- Built-in intelligence
- Source citations
- No RAG code to maintain
- Professional LLM responses

### Key Benefits:
1. **Less Maintenance**: Pinecone handles all the heavy lifting
2. **Better Answers**: Professional language with citations
3. **Easier Updates**: Just upload new files
4. **Cost Effective**: No need to manage embeddings
5. **Reliable**: Enterprise-grade service

## Adding More Documents

### Via Web Console (Easiest):
1. Go to https://app.pinecone.io
2. Select "parcel-assistant"
3. Click "Upload files"
4. Upload your PDFs, text files, or markdown files

### Via API:
```bash
export PINECONE_API_KEY=your_key
export PINECONE_ASSISTANT_HOST=your_host
uv run /path/to/upload.py --assistant parcel-assistant --source ./docs
```

## Migration Notes

- The old RAG service (`server.py`) is no longer needed
- The assistant automatically inherits all document knowledge
- API endpoint remains the same (`/query`)
- Responses are more comprehensive and cited
- No need for manual document management

## License

Part of ParcelAm project.
