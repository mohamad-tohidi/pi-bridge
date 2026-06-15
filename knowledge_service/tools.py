import requests
import re
import asyncio

async def parsa_tool(query: str) -> str:
    """
    Searches the internal semantic search API for religious content from the Quran and other sources.
    THIS SEARCH TOOL WORKS BEST WHEN QUERIES ARE IN PERSIAN LANGUAGE.
    
    Args:
        query: The search query.
    """
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
        # Using a sync call inside the async function as the current pi-bridge 
        # wrapper handles the thread management, but we'll keep it simple.
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        result = data.get("data", [])

        relevant_items = []
        
        # Based on the user's provided logic: result["result"] seems to be the list
        # The user's code had: for res in result["result"]:
        # We need to check if 'result' is a dict containing 'result' key or the list itself.
        # The user's code: result: list[dict] = response.json().get("data", [])
        # then: for res in result["result"]: 
        # This implies 'result' is actually a dict from the 'data' key.
        
        # Let's follow the user's logic carefully:
        search_results = result.get("result", []) if isinstance(result, dict) else []

        for res in search_results:
            # Extract highlight
            highlight_text = res.get("highlight", "")
            match = re.search(r"پاسخ:\s*(.*)", highlight_text)
            answer_highlight = match.group(1).strip() if match else highlight_text

            cleaned_item = {
                "id": res.get("id"),
                "title": res.get("content"),
                "snippet": answer_highlight[:600],
                "source_link": res.get("source_link", "")
            }
            relevant_items.append(cleaned_item)

        if not relevant_items:
            return "No relevant religious content found for this query."

        # Format the output for the LLM
        formatted_results = "\n\n".join([
            f"Source: {item['title']}\nSnippet: {item['snippet']}\nLink: {item['source_link']}"
            for item in relevant_items
        ])
        return formatted_results

    except Exception as e:
        return f"Error searching Parsa: {str(e)}"

async def haditha_tool(query: str) -> str:
    """
    Search for Ahadith from AhlulBait.
    
    Args:
        query: The search query.
    """
    # Placeholder for Haditha implementation
    return f"HADITHA RESULT: (Mock) Searching for Ahadith related to '{query}'... [Mocked Content]"

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
