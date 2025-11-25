# AI Chatbot Backend

FastAPI backend with Hybrid RAG (Retrieval Augmented Generation) for Amirlahi's portfolio chatbot.

## Features

- **Hybrid Search**: Combines semantic search (vector similarity) + BM25 (keyword) search for optimal retrieval
- **OpenAI Integration**: Uses GPT-3.5-turbo for chat and text-embedding-3-small for embeddings
- **ChromaDB**: Persistent vector store for knowledge base
- **Conversation Memory**: Multi-turn conversation support
- **FastAPI**: Modern, fast API with automatic documentation

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your-actual-api-key-here
```

### 3. Build Vector Embeddings

**Important**: Run this once before starting the server:

```bash
python scripts/build_embeddings.py
```

This will:
- Load the knowledge base (`knowledge_base/amirlahi_portfolio.md`)
- Split into chunks based on Markdown headers
- Generate embeddings using OpenAI
- Store in ChromaDB vector store

### 4. Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

## API Endpoints

### POST `/api/chat`

Chat with the AI assistant.

**Request:**
```json
{
  "message": "What are Amirlahi's Python projects?",
  "conversation_id": "conv_123"  // optional
}
```

**Response:**
```json
{
  "response": "Amirlahi has worked on several Python projects including...",
  "sources": [
    {
      "content": "Customer Segmentation project...",
      "metadata": {"h2": "Projects", "h3": "Customer-Segmentation"}
    }
  ],
  "conversation_id": "conv_123"
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running",
  "vector_store_loaded": true
}
```

### POST `/api/reset-conversation`

Reset conversation history.

**Request:**
```json
{
  "conversation_id": "conv_123"
}
```

## Architecture

```
User Query
    ↓
Hybrid Retrieval
    ├── Semantic Search (60%) - ChromaDB vector similarity
    └── BM25 Search (40%) - Keyword matching
    ↓
Top-5 Most Relevant Documents
    ↓
Context Assembly + Conversation History
    ↓
OpenAI GPT-3.5-turbo
    ↓
Response with Sources
```

## Project Structure

```
ai_engine/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   └── chat.py          # Pydantic models
│   ├── services/
│   │   └── rag_service.py   # RAG implementation (Hybrid Search)
│   └── api/
│       └── chat.py          # Chat endpoints
├── knowledge_base/
│   └── amirlahi_portfolio.md  # Knowledge base (Markdown)
├── scripts/
│   ├── build_embeddings.py  # Build vector store
│   ├── fetch_github.py      # Fetch GitHub data
│   └── ...
├── vector_store/
│   └── chroma_db/           # ChromaDB persistent storage
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Edit `app/config.py` or set environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `MODEL_NAME`: GPT model to use (default: gpt-3.5-turbo)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 5)
- `SEMANTIC_WEIGHT`: Weight for semantic search (default: 0.6)
- `BM25_WEIGHT`: Weight for BM25 search (default: 0.4)

## Development

### Run in Development Mode

```bash
uvicorn app.main:app --reload --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/api/health

# Chat request
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are Amirlahi'\''s skills?"}'
```

### Rebuild Vector Store

If you update the knowledge base, rebuild the vector store:

```bash
python scripts/build_embeddings.py
```

## Troubleshooting

### "Vector store not loaded" error

Run `python scripts/build_embeddings.py` to initialize the vector store.

### "OpenAI API key not set" error

Make sure you've created a `.env` file with your OpenAI API key.

### Import errors

Make sure all dependencies are installed: `pip install -r requirements.txt`

## Cost Estimation

- **Embeddings**: ~$0.0003 one-time cost (for ~50KB knowledge base)
- **Chat**: ~$0.001-0.003 per conversation (with GPT-3.5-turbo)

## Next Steps

1. Connect frontend Chatbot component to this API
2. Test end-to-end functionality
3. Deploy to production (Railway, Render, etc.)
