#!/usr/bin/env python3
"""
Web-based Chat UI for Local LLM using Ollama with RAG support.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, render_template_string, request
import requests as http_requests

from model_manager import ModelManager
from rag import RAGEngine, build_rag_prompt

app = Flask(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral-nemo"

# Store conversation history per session (simple in-memory storage)
conversations: dict[str, list[dict]] = {}

# Store last request info per session for regeneration
last_requests: dict[str, dict] = {}

# Global RAG engine instance
rag_engine: RAGEngine | None = None

# Store last query sources per session for the /rag_sources endpoint
last_rag_sources: dict[str, list[dict]] = {}

# Global model manager instance
model_manager: ModelManager | None = None

# Current active model
current_model: str = DEFAULT_MODEL

# Conversations storage directory
CONVERSATIONS_DIR = Path(__file__).parent / "conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local LLM Chat</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- Highlight.js theme -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <!-- Application styles -->
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <!-- Sidebar for conversation history -->
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h2>Conversations</h2>
            <button class="btn btn-primary sidebar-new-btn" onclick="newConversation()">New Chat</button>
        </div>
        <div class="sidebar-search">
            <input type="text" id="sidebar-search" placeholder="Search conversations..." oninput="filterConversations()">
        </div>
        <div class="conversation-list" id="conversation-list">
            <!-- Conversations will be loaded here -->
        </div>
    </aside>

    <div class="main-content" id="main-content">
        <header class="header">
            <div class="header-left">
                <button class="btn sidebar-toggle" id="sidebar-toggle" onclick="toggleSidebar()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
                </button>
                <h1 class="conversation-title" id="conversation-title" onclick="editConversationName()">New Chat</h1>
            </div>
            <div class="header-actions">
                <select class="model-select" id="model-select" onchange="switchModel()">
                    <option value="{{ model }}">{{ model }}</option>
                </select>
                <button class="btn" id="rag-toggle-btn" onclick="toggleRagPanel()">Documents</button>
                <button class="btn" onclick="exportChat()">Export</button>
                <button class="btn" onclick="clearChat()">Clear</button>
            </div>
        </header>

        <div class="rag-panel" id="rag-panel">
            <div class="rag-input-group">
                <input 
                    type="text" 
                    class="rag-input" 
                    id="folder-path" 
                    placeholder="Enter folder path containing PDFs..."
                >
                <button class="btn btn-primary" id="load-btn" onclick="loadDocuments()">Load PDFs</button>
            </div>
            <div class="rag-status" id="rag-status">
                <span class="rag-status-dot" id="rag-status-dot"></span>
                <span id="rag-status-text">No documents loaded</span>
            </div>
            <label class="rag-toggle">
                <input type="checkbox" id="rag-enabled" onchange="updateRagMode()">
                <span>Use RAG</span>
            </label>
        </div>

        <main class="chat-container" id="chat-container">
            <div class="welcome">
                <h2>Start a conversation</h2>
                <p>Messages are processed locally using Ollama.</p>
                <p style="margin-top: 0.5rem; font-size: 0.85rem;">Click "Documents" to load PDFs for document Q&A.</p>
            </div>
        </main>

        <footer class="input-area">
            <div class="input-wrapper">
                <textarea 
                    id="message-input" 
                    placeholder="Type a message..." 
                    rows="1"
                    onkeydown="handleKeyDown(event)"
                ></textarea>
                <button id="send-btn" onclick="sendMessage()">
                    Send
                </button>
            </div>
        </footer>
    </div>

    <!-- Marked.js for markdown rendering -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js"></script>
    <!-- Highlight.js for syntax highlighting -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- Application JavaScript -->
    <script src="/static/chat.js"></script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template_string(HTML_TEMPLATE, model=current_model)


@app.route("/load_documents", methods=["POST"])
def load_documents():
    """Load PDF documents for RAG."""
    global rag_engine

    data = request.json
    folder_path = data.get("folder_path", "")

    if not folder_path:
        return {"error": "No folder path provided"}

    try:
        # Initialize RAG engine if needed
        if rag_engine is None:
            rag_engine = RAGEngine()

        stats = rag_engine.load_pdfs(folder_path)
        return {
            "files_processed": stats["files_processed"],
            "chunks_created": stats["chunks_created"],
            "errors": stats["errors"],
        }

    except FileNotFoundError as e:
        return {"error": str(e)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to load documents: {str(e)}"}


@app.route("/rag_status")
def rag_status():
    """Get RAG engine status."""
    global rag_engine

    if rag_engine is None or not rag_engine.is_loaded():
        return {"loaded": False, "chunk_count": 0}

    stats = rag_engine.get_stats()
    return stats


@app.route("/rag_sources", methods=["GET", "POST"])
def rag_sources():
    """Get the sources used in the last RAG query for a session."""
    if request.method == "POST":
        data = request.json or {}
        session_id = data.get("session_id", "default")
    else:
        session_id = request.args.get("session_id", "default")

    sources = last_rag_sources.get(session_id, [])
    return {
        "session_id": session_id,
        "sources": sources,
        "count": len(sources),
    }


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages with streaming response."""
    global rag_engine, current_model

    data = request.json
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    use_rag = data.get("use_rag", False)

    if not message:
        return Response(
            "data: {\"error\": \"No message provided\"}\n\n",
            mimetype="text/event-stream",
        )

    # Store last request for regeneration
    last_requests[session_id] = {"message": message, "use_rag": use_rag}

    # Build messages
    if use_rag and rag_engine is not None and rag_engine.is_loaded():
        # RAG mode: get context and build prompt
        context, citations = rag_engine.get_context(message, k=3)
        last_rag_sources[session_id] = citations
        messages = build_rag_prompt(context, message)
    else:
        # Regular chat mode with conversation history
        if session_id not in conversations:
            conversations[session_id] = []

        conversations[session_id].append({"role": "user", "content": message})
        messages = conversations[session_id]

    def generate():
        try:
            response = http_requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": current_model,
                    "messages": messages,
                    "stream": True,
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            full_response = ""

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                    if chunk.get("done", False):
                        break

            # Store assistant response in history (only for non-RAG mode)
            if not use_rag and session_id in conversations:
                conversations[session_id].append(
                    {"role": "assistant", "content": full_response}
                )

        except http_requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Could not connect to Ollama. Make sure it is running.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/clear", methods=["POST"])
def clear():
    """Clear conversation history for a session."""
    data = request.json
    session_id = data.get("session_id", "default")
    if session_id in conversations:
        del conversations[session_id]
    if session_id in last_requests:
        del last_requests[session_id]
    return {"status": "ok"}


@app.route("/models")
def list_models():
    """List available Ollama models."""
    global model_manager, current_model
    
    if model_manager is None:
        model_manager = ModelManager(OLLAMA_URL)
    
    result = model_manager.list_models()
    result["current_model"] = current_model
    return result


@app.route("/switch_model", methods=["POST"])
def switch_model():
    """Switch to a different model."""
    global model_manager, current_model
    
    if model_manager is None:
        model_manager = ModelManager(OLLAMA_URL)
    
    data = request.json
    new_model = data.get("model", "")
    
    if not new_model:
        return {"error": "No model specified", "current_model": current_model}
    
    # Check if model exists
    if not model_manager.model_exists(new_model):
        return {"error": f"Model '{new_model}' not found", "current_model": current_model}
    
    current_model = new_model
    return {"status": "ok", "current_model": current_model}


@app.route("/regenerate", methods=["POST"])
def regenerate():
    """Regenerate the last assistant response."""
    global rag_engine, current_model
    
    data = request.json
    session_id = data.get("session_id", "default")
    
    # Check if we have a last request for this session
    if session_id not in last_requests:
        return Response(
            "data: {\"error\": \"No previous message to regenerate\"}\n\n",
            mimetype="text/event-stream",
        )
    
    last_req = last_requests[session_id]
    message = last_req.get("message", "")
    use_rag = last_req.get("use_rag", False)
    
    if not message:
        return Response(
            "data: {\"error\": \"No previous message to regenerate\"}\n\n",
            mimetype="text/event-stream",
        )
    
    # Remove the last assistant response from history if it exists
    if session_id in conversations and len(conversations[session_id]) > 0:
        if conversations[session_id][-1].get("role") == "assistant":
            conversations[session_id].pop()
    
    # Build messages
    if use_rag and rag_engine is not None and rag_engine.is_loaded():
        from rag import build_rag_prompt
        context = rag_engine.get_context(message, k=3)
        messages = build_rag_prompt(context, message)
    else:
        if session_id not in conversations:
            return Response(
                "data: {\"error\": \"No conversation history found\"}\n\n",
                mimetype="text/event-stream",
            )
        messages = conversations[session_id]
    
    def generate():
        try:
            response = http_requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": current_model,
                    "messages": messages,
                    "stream": True,
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                    if chunk.get("done", False):
                        break
            
            # Store new assistant response in history (only for non-RAG mode)
            if not use_rag and session_id in conversations:
                conversations[session_id].append(
                    {"role": "assistant", "content": full_response}
                )
        
        except http_requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Could not connect to Ollama. Make sure it is running.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


# ============== Conversation Persistence Endpoints ==============

@app.route("/conversations", methods=["GET"])
def list_conversations():
    """List all saved conversations."""
    convos = []
    
    for file_path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                convos.append({
                    "id": data.get("id", file_path.stem),
                    "name": data.get("name", "Untitled"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "model": data.get("model", ""),
                    "message_count": len(data.get("messages", [])),
                })
        except (json.JSONDecodeError, IOError):
            continue
    
    # Sort by updated_at descending (most recent first)
    convos.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    
    return {"conversations": convos}


@app.route("/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    """Load a specific conversation."""
    file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    if not file_path.exists():
        return {"error": "Conversation not found"}, 404
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        return {"error": f"Failed to load conversation: {str(e)}"}, 500


@app.route("/conversations/<conversation_id>", methods=["POST"])
def save_conversation(conversation_id):
    """Save or update a conversation."""
    data = request.json
    
    if not data:
        return {"error": "No data provided"}, 400
    
    file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    # If file exists, preserve created_at
    created_at = datetime.now().isoformat()
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                created_at = existing.get("created_at", created_at)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Build conversation data
    conversation_data = {
        "id": conversation_id,
        "name": data.get("name", "Untitled"),
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "model": data.get("model", current_model),
        "messages": data.get("messages", []),
    }
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        return {"status": "ok", "id": conversation_id}
    except IOError as e:
        return {"error": f"Failed to save conversation: {str(e)}"}, 500


@app.route("/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id):
    """Delete a conversation."""
    file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    if not file_path.exists():
        return {"error": "Conversation not found"}, 404
    
    try:
        file_path.unlink()
        return {"status": "ok"}
    except IOError as e:
        return {"error": f"Failed to delete conversation: {str(e)}"}, 500


@app.route("/conversations/<conversation_id>/rename", methods=["POST"])
def rename_conversation(conversation_id):
    """Rename a conversation."""
    data = request.json
    new_name = data.get("name", "").strip()
    
    if not new_name:
        return {"error": "No name provided"}, 400
    
    file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    if not file_path.exists():
        return {"error": "Conversation not found"}, 404
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            conversation_data = json.load(f)
        
        conversation_data["name"] = new_name
        conversation_data["updated_at"] = datetime.now().isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return {"status": "ok", "name": new_name}
    except (json.JSONDecodeError, IOError) as e:
        return {"error": f"Failed to rename conversation: {str(e)}"}, 500


def main():
    global rag_engine, model_manager, current_model

    parser = argparse.ArgumentParser(description="Web-based Chat UI for Local LLM with RAG")
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Model to use (default: llama3.2)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--preload-rag",
        default=None,
        help="Preload PDFs from this folder on startup",
    )

    args = parser.parse_args()

    global DEFAULT_MODEL, OLLAMA_URL
    DEFAULT_MODEL = args.model
    OLLAMA_URL = args.ollama_url
    current_model = args.model

    # Initialize model manager
    model_manager = ModelManager(OLLAMA_URL)

    # Preload RAG if specified
    if args.preload_rag:
        print(f"Preloading documents from: {args.preload_rag}")
        rag_engine = RAGEngine()
        stats = rag_engine.load_pdfs(args.preload_rag)
        print(f"Loaded {stats['files_processed']} files, {stats['chunks_created']} chunks")

    print(f"Starting web chat UI...")
    print(f"Model: {current_model}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Open http://{args.host}:{args.port} in your browser")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
