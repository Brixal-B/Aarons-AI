#!/usr/bin/env python3
"""
Web Search Module - Search the web using DuckDuckGo and fetch page content.
No API key required.
"""

import re
import html as html_module
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


# DuckDuckGo HTML search URL
DDG_URL = "https://html.duckduckgo.com/html/"

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def search_web(query: str, num_results: int = 3, fetch_content: bool = True) -> list[dict]:
    """
    Search the web using DuckDuckGo and optionally fetch page content.
    
    Args:
        query: The search query.
        num_results: Maximum number of results to return.
        fetch_content: Whether to fetch and extract content from result pages.
        
    Returns:
        List of dicts with 'title', 'url', 'snippet', and optionally 'content' keys.
    """
    try:
        # Make the search request
        response = requests.post(
            DDG_URL,
            data={"q": query, "b": ""},
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        html = response.text
        
        # Parse results from HTML
        results = _parse_ddg_results(html, num_results)
        
        # Fetch actual content from pages
        if fetch_content and results:
            results = _fetch_page_contents(results)
        
        return results
        
    except requests.RequestException as e:
        print(f"Search error: {e}")
        return []


def _parse_ddg_results(html: str, max_results: int) -> list[dict]:
    """Parse DuckDuckGo HTML results."""
    results = []
    
    # Title and URL pattern
    link_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
    # Snippet pattern
    snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*[^<]*)</a>'
    
    # Find all matches
    links = re.findall(link_pattern, html)
    snippets = re.findall(snippet_pattern, html)
    
    for i, (url, title) in enumerate(links[:max_results + 2]):  # Get a few extra in case some fail
        snippet = ""
        if i < len(snippets):
            # Clean HTML from snippet
            snippet = re.sub(r'<[^>]+>', '', snippets[i])
            snippet = snippet.strip()
        
        # Skip DuckDuckGo internal links
        if "duckduckgo.com" in url:
            continue
            
        results.append({
            "title": title.strip(),
            "url": url,
            "snippet": snippet,
            "content": "",
        })
        
        if len(results) >= max_results:
            break
    
    return results


def _fetch_page_contents(results: list[dict], max_content_length: int = 2000) -> list[dict]:
    """Fetch and extract text content from result URLs in parallel."""
    
    def fetch_one(result):
        try:
            response = requests.get(
                result["url"],
                headers=HEADERS,
                timeout=8,
                allow_redirects=True,
            )
            response.raise_for_status()
            
            # Extract text content
            content = _extract_article_text(response.text)
            
            # Truncate to max length
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            result["content"] = content
            return result
            
        except Exception as e:
            print(f"Failed to fetch {result['url']}: {e}")
            result["content"] = result["snippet"]  # Fall back to snippet
            return result
    
    # Fetch pages in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_one, r): r for r in results}
        fetched_results = []
        
        for future in as_completed(futures, timeout=15):
            try:
                fetched_results.append(future.result())
            except Exception:
                fetched_results.append(futures[future])
    
    return fetched_results


def _extract_article_text(html: str) -> str:
    """Extract main article text from HTML, focusing on content-rich elements."""
    
    # Remove script, style, nav, header, footer, aside
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<form[^>]*>.*?</form>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    # Try to find article or main content
    article_match = re.search(r'<article[^>]*>(.*?)</article>', html, re.IGNORECASE | re.DOTALL)
    if article_match:
        html = article_match.group(1)
    else:
        main_match = re.search(r'<main[^>]*>(.*?)</main>', html, re.IGNORECASE | re.DOTALL)
        if main_match:
            html = main_match.group(1)
    
    # Extract text from paragraphs for cleaner content
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.IGNORECASE | re.DOTALL)
    
    if paragraphs:
        # Clean each paragraph
        clean_paragraphs = []
        for p in paragraphs:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', p)
            # Decode HTML entities
            text = html_module.unescape(text)
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 30:  # Skip very short paragraphs (likely navigation)
                clean_paragraphs.append(text)
        
        return "\n\n".join(clean_paragraphs)
    
    # Fallback: just strip all HTML
    text = re.sub(r'<[^>]+>', ' ', html)
    text = html_module.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def format_search_context(results: list[dict]) -> str:
    """
    Format search results as context for the LLM.
    
    Args:
        results: List of search result dicts.
        
    Returns:
        Formatted string for injection into prompt.
    """
    if not results:
        return "No search results found."
    
    parts = ["Here are the web search results:\n"]
    
    for i, result in enumerate(results, 1):
        parts.append(f"--- Source {i}: {result['title']} ---")
        parts.append(f"URL: {result['url']}")
        
        # Use content if available, otherwise snippet
        content = result.get('content', '') or result.get('snippet', '')
        if content:
            parts.append(f"\n{content}\n")
        parts.append("")
    
    return "\n".join(parts)


# Search prompt template
SEARCH_SYSTEM_PROMPT = """You are a helpful assistant with access to web search results.

Instructions:
- Use the web search results provided to answer the user's question
- Cite sources by mentioning the website name or URL when relevant
- If the search results don't contain enough information, say so
- Provide accurate, up-to-date information based on the search results
- Be concise but thorough"""


def build_search_prompt(
    search_context: str, 
    question: str, 
    memories_context: str = "",
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """
    Build messages list for search-augmented chat.
    
    Args:
        search_context: Formatted search results.
        question: User's question.
        memories_context: Optional user memory context.
        conversation_history: Optional previous messages for multi-turn search.
        
    Returns:
        List of message dicts for Ollama API.
    """
    system_content = SEARCH_SYSTEM_PROMPT
    if memories_context:
        system_content += f"\n\n{memories_context}"
    
    messages = [{"role": "system", "content": system_content}]
    
    # Add conversation history for multi-turn context
    if conversation_history:
        # Include recent history (limit to last 6 messages to avoid context overflow)
        recent_history = conversation_history[-6:]
        messages.extend(recent_history)
    
    # Add current question with search context
    messages.append({
        "role": "user",
        "content": f"{search_context}\n---\n\nQuestion: {question}",
    })
    
    return messages


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python web_search.py <query>")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}\n")
    
    results = search_web(query)
    
    if not results:
        print("No results found.")
    else:
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['title']}")
            print(f"    {r['url']}")
            print(f"    {r['snippet'][:100]}..." if len(r['snippet']) > 100 else f"    {r['snippet']}")
            print()
        
        print("\n--- Formatted Context ---")
        print(format_search_context(results))

