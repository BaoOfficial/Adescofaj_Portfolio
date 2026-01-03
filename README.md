# Amirlahi Portfolio - AI-Powered Data Science Portfolio

An interactive portfolio website featuring an intelligent AI chatbot that answers questions about Amirlahi's work, projects, and professional experience using advanced RAG (Retrieval-Augmented Generation) technology.

**Live Site**: [https://amir-adescofaj.onrender.com](https://amir-adescofaj.onrender.com)

## Project Overview

This is a modern, full-stack portfolio application that combines a sleek React frontend with a powerful AI-driven backend. The standout feature is an intelligent chatbot that can answer questions about Amirlahi's skills, projects, and experience using natural language processing and semantic search.

## Key Features

### AI-Powered Chatbot
- Conversational AI assistant built with OpenAI GPT-3.5-turbo
- Hybrid search combining semantic understanding and keyword matching
- Multi-turn conversation support with context retention
- 60-second cold-start handling for seamless user experience

### Hybrid RAG System
- **Semantic Search (60%)**: Vector embeddings using ChromaDB for contextual understanding
- **Keyword Search (40%)**: BM25 algorithm for precise term matching
- Retrieves top-5 most relevant content chunks for accurate responses
- Custom-built knowledge base with structured portfolio content

### Modern User Interface
- Responsive design optimized for all devices
- Smooth scroll-based animations using Intersection Observer
- Floating chat widget with typing indicators and loading states
- Professional design system with custom Tailwind CSS theme

### Technical Architecture
- Two-service microservices architecture for scalability
- RESTful API with FastAPI backend
- Real-time conversational AI with conversation memory
- Production-deployed on Render with optimized performance

## Technology Stack

### Backend (AI Engine)
- **FastAPI** - High-performance Python web framework
- **LangChain** - AI orchestration and integration framework
- **ChromaDB** - Vector database for semantic search capabilities
- **OpenAI API** - GPT-3.5-turbo for natural language generation
- **BM25** - Statistical ranking algorithm for keyword search
- **Pydantic** - Data validation and settings management

### Frontend
- **React 18** - Modern component-based UI framework
- **TypeScript** - Type-safe JavaScript for robust development
- **Vite** - Next-generation frontend build tool
- **Tailwind CSS** - Utility-first CSS framework with custom design system
- **shadcn-ui** - Collection of accessible, customizable UI components
- **React Router** - Declarative routing for single-page application
- **TanStack Query** - Powerful data fetching and state management
- **React Hook Form + Zod** - Form handling with schema validation

## Architecture Highlights

### System Architecture & Request Flow

```mermaid
graph TD
    A[User Browser] -->|User types message| B[React Frontend<br/>Port 8080]
    B -->|POST /api/chat<br/>{message, conversation_id}| C[FastAPI Backend<br/>Port 8000]

    C -->|1. Retrieve| D[Conversation Memory<br/>Last 5 exchanges]

    C -->|2. Query| E[Hybrid Search System]

    E -->|60% weight| F[ChromaDB<br/>Semantic Search<br/>Vector Similarity]
    E -->|40% weight| G[BM25 Algorithm<br/>Keyword Search]

    F --> H[Ensemble Retriever<br/>Top-5 Results]
    G --> H

    H -->|3. Build Context| I[Context Assembly<br/>Documents + History + Query]
    D -->|Add history| I

    I -->|4. Generate| J[OpenAI API<br/>GPT-3.5-turbo]

    J -->|5. Response| K[RAG Service<br/>Store Conversation]

    K -->|{response, sources, conversation_id}| B
    B -->|Display| A

    F -.->|Persistent Storage| L[(ChromaDB<br/>Vector Store<br/>250+ Documents)]

    style A fill:#e1f5ff
    style B fill:#b3d9ff
    style C fill:#ffccbc
    style E fill:#fff9c4
    style J fill:#c8e6c9
    style L fill:#f8bbd0
```

### Request Flow Breakdown
1. **User Query** → Submitted through chat interface with conversation_id
2. **Conversation Retrieval** → Backend loads last 5 message exchanges from memory
3. **Hybrid Search** →
   - ChromaDB performs vector similarity search (60% weight)
   - BM25 performs keyword matching (40% weight)
   - Results combined using ensemble scoring
4. **Context Assembly** → Top-5 documents + conversation history + user query
5. **AI Generation** → OpenAI GPT-3.5-turbo generates contextually-aware response
6. **Response Delivery** → Answer returned with source attribution and conversation_id

### Data Strategy
- **ChromaDB**: Embedded vector database with persistent storage
- **Knowledge Base**: Structured markdown content split by headers
- **Conversation Memory**: In-memory storage for context retention (5-exchange history)
- **Embeddings**: OpenAI text-embedding-3-small for semantic understanding

## Project Structure

```
├── ai_engine/          # FastAPI backend service
│   ├── app/
│   │   ├── api/       # REST API endpoints
│   │   ├── services/  # RAG service with hybrid search
│   │   ├── models/    # Pydantic schemas
│   │   └── main.py    # Application entry point
│   ├── knowledge_base/
│   │   └── amirlahi_portfolio.md  # Portfolio content
│   └── scripts/       # Embedding generation utilities
│
└── frontend/          # React frontend application
    ├── src/
    │   ├── components/  # Reusable UI components
    │   ├── pages/       # Page-level components
    │   ├── hooks/       # Custom React hooks
    │   └── lib/         # Utility functions
    └── vite.config.ts   # Build configuration
```

## Performance & Optimization

- Vite for lightning-fast development and optimized production builds
- Persistent ChromaDB vector store (no rebuild on restart)
- Efficient conversation caching in memory
- Component lazy loading for faster initial page load
- Tailwind CSS with PurgeCSS for minimal bundle size
- Cold-start timeout handling for serverless deployment

## Design System

- **Color Palette**: Custom CSS variables for consistent theming
- **Typography**: Carefully selected fonts with proper hierarchy
- **Animations**: Custom keyframe animations (float, glow-pulse, fade-in, slide-in)
- **Responsive Design**: Mobile-first approach with breakpoint optimization
- **Accessibility**: ARIA-compliant components from shadcn-ui/Radix

## API Capabilities

The backend exposes a clean REST API with endpoints for:
- Chat message processing with conversation tracking
- Health checks with vector store status
- Conversation history management
- Comprehensive API documentation via Swagger/ReDoc

## Deployment

Both services are deployed on Render:
- Frontend: Static site deployment with environment configuration
- Backend: Web service with persistent vector storage
- Automated deployments from GitHub repository
- Environment-based configuration management

## Development Highlights

- **Type Safety**: Full TypeScript implementation in frontend
- **Code Quality**: ESLint configuration for consistent code standards
- **Modern Patterns**: React hooks, functional components, custom hooks
- **API Design**: RESTful principles with proper HTTP semantics
- **Error Handling**: Graceful degradation with user-friendly messages
- **State Management**: Efficient with TanStack Query and local state

## Browser Compatibility

Tested and optimized for:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

---

**Built with React, FastAPI, LangChain, and OpenAI**
