# Local LLM Chat

A chat application that uses Ollama to run local LLMs. Includes both a terminal CLI and a web-based UI with RAG (Retrieval-Augmented Generation) support for document Q&A.

## Features

- **Terminal CLI** - Simple command-line chat interface with streaming responses
- **Web UI** - Browser-based chat with markdown rendering and syntax highlighting
- **RAG Support** - Load documents to chat with your content:
  - Drag-and-drop file upload (PDF, TXT, MD)
  - Folder path loading
  - URL ingestion for web pages
  - Semantic chunking for better retrieval
- **Model Switching** - Switch between available Ollama models from the web UI
- **Conversation Persistence** - Save, load, rename, and delete chat sessions
- **Export** - Download conversations as Markdown files
- **Keyboard Shortcuts** - Quick actions for common operations
- **Local Processing** - All data stays on your machine

## Prerequisites

1. **Ollama** - Install from [ollama.com](https://ollama.com)
2. **Python 3.10+**
3. **A downloaded model** - Run `ollama pull llama3.2` or `ollama pull mistral`

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Aarons-AI

# Install Python dependencies
pip install -r requirements.txt

# Verify Ollama is running and model is available
ollama list
```

## Usage

### Terminal CLI

```bash
python chat.py
```

**Options:**
```bash
python chat.py --model mistral              # Use a different model
python chat.py --system "You are helpful."  # Set a custom system prompt
python chat.py --timing                     # Show response timing
python chat.py --url http://localhost:11434 # Custom Ollama URL
```

**Commands:**
- Type your message and press Enter to chat
- `exit` or `quit` - End the conversation
- `clear` - Reset conversation history
- `Ctrl+C` - Interrupt generation

### Web UI

```bash
python web_chat.py
```

Open http://127.0.0.1:5000 in your browser.

**Options:**
```bash
python web_chat.py --port 8080              # Use different port
python web_chat.py --model mistral          # Use different model
python web_chat.py --host 0.0.0.0           # Allow external connections
python web_chat.py --preload-rag ./docs     # Preload documents on startup
```

### RAG Mode (Chat with Documents)

The RAG engine supports PDF, TXT, and Markdown files, plus web URLs.

**Via Web UI:**
1. Click the "Documents" button in the header
2. Choose your input method:
   - **Upload Files**: Drag and drop files or click to browse
   - **Folder Path**: Enter a local folder path
   - **URL**: Enter a web page URL to ingest
3. Documents are automatically processed and indexed
4. Enable "Use RAG" checkbox
5. Ask questions about your documents

**Via CLI (testing):**
```bash
python rag.py ./path/to/docs "What is the main topic?"
```

The system uses sentence-transformers (`all-MiniLM-L6-v2`) for embeddings and ChromaDB for vector storage. Documents are split using semantic chunking that respects paragraph and sentence boundaries for better context retrieval.

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line in message |
| `Ctrl+Shift+E` | Export chat |
| `Ctrl+Shift+K` | Clear chat |
| `Ctrl+Shift+D` | Toggle documents panel |
| `Ctrl+Shift+N` | New conversation |
| `Ctrl+/` | Show shortcuts help |
| `Escape` | Focus input field |

## Project Structure

```
Aarons-AI/
├── chat.py           # Terminal CLI chat application
├── web_chat.py       # Flask web server with chat UI
├── rag.py            # RAG engine (document loading, embedding, search)
├── model_manager.py  # Ollama model listing and switching
├── requirements.txt  # Python dependencies
├── static/
│   ├── chat.js       # Frontend JavaScript
│   └── styles.css    # UI styling
├── chroma_db/        # Vector database storage (auto-created)
├── library/          # Example documents
└── test_docs/        # Test documents
```

## API Endpoints (Web UI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the chat UI |
| `/chat` | POST | Send a message (streaming response) |
| `/clear` | POST | Clear conversation history |
| `/load_documents` | POST | Load documents from folder for RAG |
| `/upload_documents` | POST | Upload files for RAG (multipart) |
| `/load_url` | POST | Load URL content for RAG |
| `/rag_status` | GET | Get RAG engine status |
| `/rag_sources` | GET/POST | Get sources from last RAG query |
| `/models` | GET | List available Ollama models |
| `/switch_model` | POST | Switch to a different model |
| `/regenerate` | POST | Regenerate the last response |
| `/conversations` | GET | List saved conversations |
| `/conversations/<id>` | GET/POST | Load or save a conversation |
| `/conversations/<id>/rename` | POST | Rename a conversation |
| `/conversations/<id>/delete` | POST | Delete a conversation |

## Models

With 8-12 GB VRAM, you can run:

| Model | Command | VRAM |
|-------|---------|------|
| Llama 3.2 (3B) | `ollama pull llama3.2` | ~2 GB |
| Llama 3.2 (8B) | `ollama pull llama3.2:8b` | ~5 GB |
| Mistral 7B | `ollama pull mistral` | ~5 GB |
| Phi-3 | `ollama pull phi3` | ~4 GB |

## Dependencies

- `flask` - Web framework
- `requests` - HTTP client for Ollama API
- `chromadb` - Vector database for RAG
- `sentence-transformers` - Text embedding model
- `pypdf` - PDF text extraction

## Troubleshooting

**"Connection refused" error:**
- Make sure Ollama is running. On Windows, it runs as a background service.
- Try running `ollama serve` in a separate terminal.

**Slow responses:**
- First response after loading a model is slower (model loading into VRAM).
- Subsequent responses should be faster.

**RAG not finding relevant content:**
- Try rephrasing your question
- Ensure documents contain the information you're asking about
- Check that documents were loaded successfully (check chunk count)

**Model not found:**
- Run `ollama list` to see available models
- Pull the model with `ollama pull <model-name>`
