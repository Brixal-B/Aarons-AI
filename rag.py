#!/usr/bin/env python3
"""
RAG Engine - Document loading, chunking, embedding, and semantic search.
Supports PDF, TXT, and Markdown files.
"""

import os
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# Default paths
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class RAGEngine:
    """Handles document ingestion, embedding, and retrieval."""

    def __init__(
        self,
        db_path: Path | str = CHROMA_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = None
        self.collection_name = None
        
        # Track last query sources for citation retrieval
        self._last_query_sources: list[dict] = []

    def _generate_collection_name(self, folder_path: str) -> str:
        """Generate a unique collection name based on folder path."""
        # Use hash of path to create valid collection name
        path_hash = hashlib.md5(folder_path.encode()).hexdigest()[:12]
        folder_name = Path(folder_path).name[:20].replace(" ", "_")
        return f"docs_{folder_name}_{path_hash}"

    def load_documents(self, folder_path: str) -> dict:
        """
        Load all supported documents from a folder and ingest them into the vector database.
        Supports PDF, TXT, and Markdown files.

        Args:
            folder_path: Path to folder containing document files.

        Returns:
            Dict with loading statistics.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find all supported files
        doc_files = []
        for ext in SUPPORTED_EXTENSIONS:
            doc_files.extend(folder.glob(f"*{ext}"))
        
        if not doc_files:
            raise ValueError(f"No supported files found in: {folder_path}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

        # Create or get collection
        self.collection_name = self._generate_collection_name(folder_path)

        # Delete existing collection if it exists (to re-ingest fresh)
        try:
            self.client.delete_collection(self.collection_name)
        except (ValueError, Exception):
            # Collection doesn't exist, that's fine
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "errors": [],
        }

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc_file in doc_files:
            try:
                text = self._extract_text(doc_file)
                if not text.strip():
                    stats["errors"].append(f"{doc_file.name}: No text extracted")
                    stats["files_failed"] += 1
                    continue

                chunks = self._chunk_text(text)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_file.stem}_{i}"
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "source": doc_file.name,
                        "file_type": doc_file.suffix.lower(),
                        "chunk_index": i,
                    })
                    all_ids.append(chunk_id)

                stats["files_processed"] += 1
                stats["chunks_created"] += len(chunks)

            except Exception as e:
                stats["errors"].append(f"{doc_file.name}: {str(e)}")
                stats["files_failed"] += 1

        # Generate embeddings and add to collection
        if all_chunks:
            print(f"Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = self.embedder.encode(all_chunks, show_progress_bar=True)

            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids,
            )

        return stats

    def load_pdfs(self, folder_path: str) -> dict:
        """
        Load all PDFs from a folder. Alias for load_documents for backward compatibility.
        
        Args:
            folder_path: Path to folder containing PDF files.

        Returns:
            Dict with loading statistics.
        """
        return self.load_documents(folder_path)

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from a file based on its extension."""
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            return self._extract_pdf_text(file_path)
        elif ext in {".txt", ".md"}:
            return self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(pdf_path)
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from a TXT or Markdown file."""
        # Try common encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        # Fallback: read with error handling
        return file_path.read_text(encoding="utf-8", errors="replace")

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk.
            chunk_size: Target size of each chunk in words.
            overlap: Number of words to overlap between chunks.

        Returns:
            List of text chunks.
        """
        words = text.split()
        chunks = []

        if len(words) <= chunk_size:
            return [text]

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            start = end - overlap
            if start >= len(words):
                break

        return chunks

    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Search for relevant document chunks.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of dicts with 'text', 'source', 'score', 'chunk_index', and 'file_type' keys.
        """
        if self.collection is None:
            raise ValueError("No documents loaded. Call load_documents() first.")

        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results with full citation info
        formatted = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            formatted.append({
                "text": results["documents"][0][i],
                "source": metadata["source"],
                "chunk_index": metadata.get("chunk_index", i),
                "file_type": metadata.get("file_type", ".pdf"),
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })

        # Store for later retrieval via get_last_sources()
        self._last_query_sources = formatted.copy()

        return formatted

    def get_last_sources(self) -> list[dict]:
        """
        Get the sources used in the last search query.
        
        Returns:
            List of source dicts from the most recent search.
        """
        return self._last_query_sources

    def get_context(self, query: str, k: int = 3) -> tuple[str, list[dict]]:
        """
        Get formatted context string for RAG prompt along with source citations.

        Args:
            query: The user's question.
            k: Number of chunks to retrieve.

        Returns:
            Tuple of (formatted context string, list of source citations).
        """
        results = self.search(query, k=k)

        if not results:
            return "", []

        context_parts = []
        citations = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result['source']}, chunk {result['chunk_index']}]\n{result['text']}"
            )
            citations.append({
                "citation_id": i,
                "source": result["source"],
                "chunk_index": result["chunk_index"],
                "file_type": result["file_type"],
                "score": result["score"],
                "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
            })

        return "\n\n---\n\n".join(context_parts), citations

    def is_loaded(self) -> bool:
        """Check if documents are loaded."""
        return self.collection is not None and self.collection.count() > 0

    def get_stats(self) -> dict:
        """Get statistics about loaded documents."""
        if not self.is_loaded():
            return {"loaded": False, "chunk_count": 0}

        return {
            "loaded": True,
            "chunk_count": self.collection.count(),
            "collection_name": self.collection_name,
        }


# RAG prompt template
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context. 

Instructions:
- Use the context below to answer the user's question
- If the answer is not in the context, say "I don't have enough information in the loaded documents to answer that question."
- Cite the source document when possible
- Be concise and accurate"""


def build_rag_prompt(context: str, question: str) -> list[dict]:
    """
    Build messages list for RAG-augmented chat.

    Args:
        context: Retrieved document context.
        question: User's question.

    Returns:
        List of message dicts for Ollama API.
    """
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\n---\n\nQuestion: {question}",
        },
    ]


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag.py <folder_path> [query]")
        sys.exit(1)

    folder = sys.argv[1]
    engine = RAGEngine()

    print(f"\nLoading documents from: {folder}")
    print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
    stats = engine.load_documents(folder)
    print(f"Processed: {stats['files_processed']} files, {stats['chunks_created']} chunks")

    if stats["errors"]:
        print(f"Errors: {stats['errors']}")

    if len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"\nSearching for: {query}")
        results = engine.search(query)
        for r in results:
            print(f"\n[{r['source']}] (chunk {r['chunk_index']}, score: {r['score']:.3f})")
            print(r["text"][:200] + "...")
        
        print(f"\nSources used: {len(engine.get_last_sources())}")

