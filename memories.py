#!/usr/bin/env python3
"""
Memory storage and extraction for persistent user knowledge.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


# Default storage location
MEMORIES_FILE = Path(__file__).parent / "memories.json"

# System prompt for memory extraction
EXTRACTION_PROMPT = """You are a memory extraction assistant. Your job is to identify important facts from the conversation that should be remembered for future interactions.

Extract ONLY facts that are:
- Personal information about the user (name, location, job, preferences)
- Project-specific details (tech stack, project names, conventions)
- User preferences (coding style, communication preferences)
- Recurring topics or interests

DO NOT extract:
- Temporary or session-specific information
- General knowledge or facts about the world
- The content of code being discussed
- Trivial or obvious information

For each fact, provide:
1. A short title (3-5 words)
2. The fact itself (one sentence)
3. A category: "personal", "project", "preference", or "interest"

Respond in JSON format:
{
  "memories": [
    {"title": "User's Name", "content": "The user's name is Aaron.", "category": "personal"},
    {"title": "Preferred Language", "content": "User prefers Python for backend development.", "category": "preference"}
  ]
}

If there are no facts worth remembering, respond with:
{"memories": []}

Conversation to analyze:
"""


class MemoryStore:
    """Persistent storage for user memories/facts."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or MEMORIES_FILE
        self.memories: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load memories from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memories = data.get("memories", {})
            except (json.JSONDecodeError, IOError):
                self.memories = {}

    def _save(self) -> None:
        """Save memories to disk."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "memories": self.memories,
                        "updated_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except IOError as e:
            print(f"Failed to save memories: {e}")

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a memory based on content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def add(self, title: str, content: str, category: str = "general") -> str:
        """Add a new memory. Returns the memory ID."""
        memory_id = self._generate_id(content)
        
        self.memories[memory_id] = {
            "id": memory_id,
            "title": title,
            "content": content,
            "category": category,
            "created_at": datetime.now().isoformat(),
        }
        self._save()
        return memory_id

    def update(self, memory_id: str, title: Optional[str] = None, 
               content: Optional[str] = None, category: Optional[str] = None) -> bool:
        """Update an existing memory. Returns True if successful."""
        if memory_id not in self.memories:
            return False
        
        if title is not None:
            self.memories[memory_id]["title"] = title
        if content is not None:
            self.memories[memory_id]["content"] = content
        if category is not None:
            self.memories[memory_id]["category"] = category
        
        self.memories[memory_id]["updated_at"] = datetime.now().isoformat()
        self._save()
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if successful."""
        if memory_id not in self.memories:
            return False
        
        del self.memories[memory_id]
        self._save()
        return True

    def get(self, memory_id: str) -> Optional[dict]:
        """Get a specific memory by ID."""
        return self.memories.get(memory_id)

    def get_all(self) -> list[dict]:
        """Get all memories as a list."""
        return list(self.memories.values())

    def get_by_category(self, category: str) -> list[dict]:
        """Get all memories in a category."""
        return [m for m in self.memories.values() if m.get("category") == category]

    def search(self, query: str) -> list[dict]:
        """Search memories by title or content (simple substring match)."""
        query_lower = query.lower()
        results = []
        for memory in self.memories.values():
            if (query_lower in memory.get("title", "").lower() or 
                query_lower in memory.get("content", "").lower()):
                results.append(memory)
        return results

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = {}
        self._save()

    def get_context_string(self) -> str:
        """Get all memories formatted as a context string for the LLM."""
        if not self.memories:
            return ""
        
        lines = ["Known facts about the user:"]
        
        # Group by category
        categories = {}
        for memory in self.memories.values():
            cat = memory.get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(memory)
        
        for category, memories in sorted(categories.items()):
            lines.append(f"\n[{category.title()}]")
            for memory in memories:
                lines.append(f"- {memory['content']}")
        
        return "\n".join(lines)


def extract_memories_from_conversation(
    messages: list[dict],
    ollama_url: str = "http://localhost:11434",
    model: str = "mistral-nemo",
) -> list[dict]:
    """
    Use the LLM to extract memorable facts from a conversation.
    
    Args:
        messages: List of conversation messages [{"role": "user/assistant", "content": "..."}]
        ollama_url: Ollama server URL
        model: Model to use for extraction
    
    Returns:
        List of extracted memories [{"title": "...", "content": "...", "category": "..."}]
    """
    if not messages:
        return []
    
    # Format conversation for analysis
    conversation_text = ""
    for msg in messages[-10:]:  # Only analyze last 10 messages to keep context reasonable
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation_text += f"{role.upper()}: {content}\n\n"
    
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": EXTRACTION_PROMPT + conversation_text,
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        # Parse JSON response
        try:
            data = json.loads(response_text)
            return data.get("memories", [])
        except json.JSONDecodeError:
            print(f"Failed to parse memory extraction response: {response_text}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Memory extraction request failed: {e}")
        return []


def build_prompt_with_memories(
    memories_context: str,
    user_message: str,
    conversation_history: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Build a prompt that includes memory context.
    
    Args:
        memories_context: Formatted string of user memories
        user_message: The current user message
        conversation_history: Optional previous messages in the conversation
    
    Returns:
        List of messages for the LLM
    """
    messages = []
    
    # Add system message with memories if we have any
    if memories_context:
        system_content = f"""You are a helpful AI assistant. You have the following knowledge about the user from previous conversations:

{memories_context}

Use this information naturally in your responses when relevant, but don't explicitly mention that you're using "memories" or "stored information". Just incorporate the knowledge naturally."""
        messages.append({"role": "system", "content": system_content})
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    return messages

