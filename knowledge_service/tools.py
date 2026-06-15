import requests
import re
import asyncio
import logging

# Set up logging for the tools module
logger = logging.getLogger(__name__)

async def parsa_tool(query: str) -> str:
    """
    Searches the internal semantic search API for religious content from the Quran and other sources.
    THIS SEARCH TOOL WORKS BEST WHEN QUERIES ARE IN PERSIAN LANGUAGE.
    
    Args:
        query: The search query.
    """
    logger.info(f"Parsa tool called with query: {query}")
    url = "http://172.30.0.112:8181/api/user/question/search"
    params = {
        "type": "semantic",
        "query": query,
        "page_size": 10,
        "page": 1,
        "is_send_answer": "true",
    }
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        result = data.get("data", [])

        relevant_items = []
        
        search_results = result.get("result", []) if isinstance(result, dict) else []

        for res in search_results:
            highlight_text = res.get("highlight", "")
            match = re.search(r"پاسخ:\s*(.*)", highlight_text)
            answer_highlight = match.group(1).strip() if match else highlight_text

            # We include the ID and the URL in a format the LLM can parse.
            # The LLM is instructed via system prompt to use [[LINK:ID]]
            item_str = f"[ID:{res.get('id')}] Title: {res.get('content')} | Snippet: {answer_highlight[:600]} | URL: {res.get('source_link', '')}"
            relevant_items.append(item_str)

        if not relevant_items:
            return "No relevant religious content found for this query."

        return "\n\n".join(relevant_items)

    except Exception as e:
        return f"Error searching Parsa: {str(e)}"

async def haditha_tool(query: str) -> str:
    """
    Search for Ahadith from AhlulBait.
    
    Args:
        query: The search query.
    """
    return f"HADITHA RESULT: (Mock) Searching for Ahadith related to '{query}'..."

# Mapping for easy access
TOOLS = {
    "parsa": parsa_tool,
    "haditha": haditha_tool
}

def get_tool_definitions():
    return [
        {
            "name": "parsa",
            "description": "Searches for religious content from the Quran and other sources. Best used with Persian queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query in Persian or English."}
                },
                "required": ["query"]
            }
        },
        {
            "name": "haditha",
            "description": "Searches for Ahadith from AhlulBait.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    ]
