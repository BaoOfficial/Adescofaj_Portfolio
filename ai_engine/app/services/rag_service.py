"""
RAG (Retrieval Augmented Generation) service with Hybrid Search.
Implements semantic search + BM25 keyword search for optimal retrieval.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Tuple
from app.config import settings
import uuid


class EnsembleRetriever:
    """Custom ensemble retriever that combines multiple retrievers with weights."""

    def __init__(self, retrievers: List[BaseRetriever], weights: List[float]):
        """
        Initialize ensemble retriever.

        Args:
            retrievers: List of retriever instances
            weights: List of weights for each retriever (should sum to 1.0)
        """
        self.retrievers = retrievers
        self.weights = weights

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents from all retrievers and combine with weighted ranking.

        Args:
            query: User query string

        Returns:
            Combined and ranked list of documents
        """
        # Get results from all retrievers
        all_results = []
        for retriever, weight in zip(self.retrievers, self.weights):
            # Try both invoke() and get_relevant_documents() for compatibility
            try:
                docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(query)
            except AttributeError:
                docs = retriever.get_relevant_documents(query)

            # Add weighted score to each document
            for i, doc in enumerate(docs):
                # Higher rank (lower index) = higher score
                score = weight * (len(docs) - i) / len(docs) if docs else 0
                all_results.append((doc, score))

        # Combine documents by content (deduplicate)
        doc_scores = {}
        for doc, score in all_results:
            content_key = doc.page_content[:100]  # Use first 100 chars as key
            if content_key in doc_scores:
                doc_scores[content_key] = (doc, doc_scores[content_key][1] + score)
            else:
                doc_scores[content_key] = (doc, score)

        # Sort by score and return documents
        ranked_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs]


class RAGService:
    """Service for RAG operations with hybrid search."""

    def __init__(self):
        """Initialize RAG service with vector store and retrievers."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )

        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.3,
            openai_api_key=settings.openai_api_key
        )

        # Load vector store
        self.vectorstore = self._load_vector_store()

        # Load documents for BM25
        self.documents = self._load_documents()

        # Initialize retrievers
        self.semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.top_k_results * 2}
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=settings.top_k_results * 2
        )

        # Create ensemble retriever (hybrid search)
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.bm25_retriever],
            weights=[settings.semantic_weight, settings.bm25_weight]
        )

        # Conversation memory (simple in-memory storage)
        self.conversations: Dict[str, List[Dict]] = {}

    def _load_vector_store(self) -> Chroma:
        """Load the ChromaDB vector store."""
        try:
            vectorstore = Chroma(
                persist_directory=settings.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="amirlahi_portfolio"
            )
            print(f"[OK] Vector store loaded: {vectorstore._collection.count()} documents")
            return vectorstore
        except Exception as e:
            raise RuntimeError(f"Failed to load vector store: {str(e)}")

    def _load_documents(self) -> List[Document]:
        """Load documents from knowledge base for BM25."""
        with open(settings.knowledge_base_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into chunks
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        documents = markdown_splitter.split_text(content)
        print(f"[OK] Loaded {len(documents)} documents for BM25 retriever")
        return documents

    def retrieve_context(self, query: str) -> Tuple[List[Document], str]:
        """
        Retrieve relevant context using hybrid search.

        Args:
            query: User's query

        Returns:
            Tuple of (relevant documents, assembled context string)
        """
        # Use hybrid retriever
        docs = self.hybrid_retriever.get_relevant_documents(query)

        # Limit to top-k
        docs = docs[:settings.top_k_results]

        # Assemble context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata_str = " > ".join([
                f"{k}: {v}" for k, v in doc.metadata.items()
                if k in ["h1", "h2", "h3"]
            ])
            context_parts.append(
                f"[Source {i}] {metadata_str}\n{doc.page_content}\n"
            )

        context = "\n---\n".join(context_parts)

        return docs, context

    def generate_response(
        self,
        query: str,
        conversation_id: str = None
    ) -> Tuple[str, List[Document], str]:
        """
        Generate a response using RAG.

        Args:
            query: User's query
            conversation_id: Optional conversation ID for multi-turn

        Returns:
            Tuple of (response, source documents, conversation_id)
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        # Retrieve relevant context
        docs, context = self.retrieve_context(query)

        # Get conversation history
        history = self.conversations.get(conversation_id, [])
        history_text = self._format_history(history)

        # Create prompt
        system_prompt = """You are an AI assistant representing Amirlahi Ademola Fajingbesi, a Data Scientist with 6 years of experience in machine learning and data science.

Your role is to answer questions about Amirlahi's:
- Professional experience and achievements
- Technical skills and expertise
- Projects and portfolio work
- Educational background
- Contact information and availability

Use the context provided below to answer questions accurately. If you don't know something based on the context, politely say so and suggest contacting Amirlahi directly through the contact form.

Be professional, concise, and enthusiastic about Amirlahi's work. Focus on providing specific, relevant information from the context.

CONTEXT:
{context}

{history}

Answer the user's question based on the context above."""

        prompt = system_prompt.format(
            context=context,
            history=history_text
        )

        # Generate response
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        # Store in conversation history
        self.conversations.setdefault(conversation_id, []).append({
            "query": query,
            "response": answer
        })

        # Limit history to last 5 exchanges
        if len(self.conversations[conversation_id]) > 5:
            self.conversations[conversation_id] = self.conversations[conversation_id][-5:]

        return answer, docs, conversation_id

    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return ""

        history_parts = ["CONVERSATION HISTORY:"]
        for exchange in history[-3:]:  # Last 3 exchanges
            history_parts.append(f"User: {exchange['query']}")
            history_parts.append(f"Assistant: {exchange['response']}")

        return "\n".join(history_parts)

    def is_ready(self) -> bool:
        """Check if RAG service is ready."""
        try:
            return self.vectorstore._collection.count() > 0
        except:
            return False


# Global RAG service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
