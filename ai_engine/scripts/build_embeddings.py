"""
Script to build vector embeddings from the knowledge base and load into ChromaDB.
This script should be run once to initialize the vector store.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_knowledge_base():
    """Load the knowledge base Markdown file."""
    print(f"Loading knowledge base from: {settings.knowledge_base_path}")

    with open(settings.knowledge_base_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"[OK] Loaded {len(content)} characters from knowledge base")
    return content


def split_markdown_into_chunks(content):
    """Split Markdown content into chunks based on headers."""
    print("\nSplitting content into chunks based on headers...")

    # Define headers to split on
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    # Create splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    # Split the document
    chunks = markdown_splitter.split_text(content)

    print(f"[OK] Created {len(chunks)} chunks from knowledge base")

    # Display sample chunk info
    if chunks:
        print(f"\nSample chunk metadata:")
        print(f"  Metadata: {chunks[0].metadata}")
        print(f"  Content length: {len(chunks[0].page_content)} characters")

    return chunks


def create_vector_store(chunks):
    """Create ChromaDB vector store from chunks."""
    print(f"\nCreating vector store at: {settings.chroma_persist_dir}")
    print(f"Using embedding model: {settings.embedding_model}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model
    )

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_persist_dir,
        collection_name="amirlahi_portfolio"
    )

    print(f"[OK] Vector store created successfully!")
    print(f"[OK] Total documents in store: {vectorstore._collection.count()}")

    return vectorstore


def main():
    """Main function to build embeddings."""
    print("="*60)
    print("Building Vector Embeddings for Knowledge Base")
    print("="*60)

    # Check if OpenAI API key is set
    if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
        print("\n[ERROR] OpenAI API key not set!")
        print("Please set OPENAI_API_KEY in your .env file")
        return

    try:
        # Step 1: Load knowledge base
        content = load_knowledge_base()

        # Step 2: Split into chunks
        chunks = split_markdown_into_chunks(content)

        # Step 3: Create vector store
        vectorstore = create_vector_store(chunks)

        # Test query
        print("\n" + "="*60)
        print("Testing vector store with sample query...")
        print("="*60)

        test_query = "What are Amirlahi's machine learning skills?"
        results = vectorstore.similarity_search(test_query, k=3)

        print(f"\nQuery: {test_query}")
        print(f"Found {len(results)} relevant documents:\n")

        for i, doc in enumerate(results, 1):
            print(f"{i}. Metadata: {doc.metadata}")
            print(f"   Content preview: {doc.page_content[:150]}...")
            print()

        print("="*60)
        print("[SUCCESS] Vector store built and tested successfully!")
        print("="*60)
        print(f"\nVector store location: {settings.chroma_persist_dir}")
        print("You can now start the FastAPI server.")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
