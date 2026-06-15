import re
from typing import Dict

def extract_link_map(tool_results: list[str]) -> Dict[str, str]:
    """
    Parses tool result strings to build a map of ID -> URL.
    Expected format in tool result: [ID:123] Title: ... | Snippet: ... | URL: http://...
    """
    link_map = {}
    for result in tool_results:
        # Find all matches of [ID:xxx] ... | URL: yyy
        # Using a regex that looks for the ID and the URL within the same line/entry
        matches = re.findall(r"\[ID:(\w+)\] .*?\| URL: (http\S+)", result)
        for link_id, url in matches:
            link_map[link_id] = url
    return link_map

def replace_tokens_with_links(text: str, link_map: Dict[str, str]) -> str:
    """
    Replaces [[LINK:ID]] tokens with actual URLs from the map.
    """
    def replacer(match):
        link_id = match.group(1)
        return link_map.get(link_id, f"[[LINK:{link_id}]]") # Return token if ID not found

    return re.sub(r"\[\[LINK:(\w+)\]\]", replacer, text)
