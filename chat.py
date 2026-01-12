#!/usr/bin/env python3
"""
Local LLM Chat CLI - A terminal chat application using Ollama.
"""

import argparse
import json
import sys
import time
from typing import Generator

import requests


class OllamaClient:
    """Client for interacting with the Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/api/chat"

    def chat_stream(
        self, messages: list[dict], model: str = "llama3.2"
    ) -> Generator[str, None, None]:
        """
        Send a chat request and stream the response.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: The model name to use.
            
        Yields:
            Response text chunks as they arrive.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        try:
            with requests.post(
                self.chat_endpoint,
                json=payload,
                stream=True,
                timeout=300,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                        if chunk.get("done", False):
                            break

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to Ollama. Make sure it's running.\n"
                "Try running 'ollama serve' in a separate terminal."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out. The model may be overloaded.")

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class ChatSession:
    """Manages a conversation session with history."""

    def __init__(
        self,
        client: OllamaClient,
        model: str = "llama3.2",
        system_prompt: str | None = None,
    ):
        self.client = client
        self.model = model
        self.messages: list[dict] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send_message(self, user_input: str, show_timing: bool = False) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_input: The user's message.
            show_timing: Whether to display response timing.
            
        Returns:
            The complete assistant response.
        """
        self.messages.append({"role": "user", "content": user_input})

        start_time = time.time()
        full_response = ""

        print("\nAssistant: ", end="", flush=True)

        try:
            for chunk in self.client.chat_stream(self.messages, self.model):
                print(chunk, end="", flush=True)
                full_response += chunk
        except KeyboardInterrupt:
            print("\n[Generation interrupted]")

        print()  # New line after response

        if show_timing:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s]")

        self.messages.append({"role": "assistant", "content": full_response})
        return full_response

    def clear_history(self):
        """Clear conversation history, keeping system prompt if present."""
        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages


def main():
    parser = argparse.ArgumentParser(
        description="Chat with a local LLM using Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py
  python chat.py --model mistral
  python chat.py --system "You are a helpful coding assistant."
  python chat.py --timing
        """,
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Model to use (default: llama3.2)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt to set the assistant's behavior",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show response timing",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )

    args = parser.parse_args()

    # Initialize client and check connection
    client = OllamaClient(base_url=args.url)

    print(f"Connecting to Ollama at {args.url}...")

    if not client.is_available():
        print(
            "Error: Could not connect to Ollama.\n"
            "Make sure Ollama is running. Try 'ollama serve' in another terminal."
        )
        sys.exit(1)

    print(f"Using model: {args.model}")
    print("Type 'exit' to quit, 'clear' to reset conversation.\n")

    # Create chat session
    session = ChatSession(
        client=client,
        model=args.model,
        system_prompt=args.system,
    )

    # Main chat loop
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                session.clear_history()
                print("Conversation cleared.\n")
                continue

            try:
                session.send_message(user_input, show_timing=args.timing)
            except ConnectionError as e:
                print(f"\nError: {e}")
            except TimeoutError as e:
                print(f"\nError: {e}")

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()

