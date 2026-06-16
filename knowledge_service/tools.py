"""Built-in (hardcoded) tools + helpers to merge with dynamic tools."""
from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Surah name → number lookup (Arabic transliterations + common English names)
# ---------------------------------------------------------------------------
_SURAH_NAMES: dict[str, int] = {
    "al-fatihah": 1, "al fatihah": 1, "fatihah": 1, "fatiha": 1, "fatiha": 1,
    "al-baqarah": 2, "al baqarah": 2, "baqarah": 2,
    "al-imran": 3, "al imran": 3, "imran": 3,
    "an-nisa": 4, "an nisa": 4, "nisa": 4,
    "al-maidah": 5, "al maidah": 5, "maidah": 5,
    "al-anam": 6, "al anam": 6, "anam": 6,
    "al-araf": 7, "al araf": 7, "araf": 7,
    "al-anfal": 8, "al anfal": 8, "anfal": 8,
    "at-tawbah": 9, "at tawbah": 9, "tawbah": 9,
    "yunus": 10, "hud": 11, "yusuf": 12,
    "ar-rad": 13, "ibrahim": 14, "al-hijr": 15,
    "an-nahl": 16, "al-isra": 17, "al-kahf": 18,
    "maryam": 19, "ta-ha": 20, "taha": 20,
    "al-anbiya": 21, "al-hajj": 22, "al-muminun": 23,
    "an-nur": 24, "al-furqan": 25, "ash-shuara": 26,
    "an-naml": 27, "al-qasas": 28, "al-ankabut": 29,
    "ar-rum": 30, "luqman": 31, "as-sajdah": 32,
    "al-ahzab": 33, "saba": 34, "fatir": 35,
    "ya-sin": 36, "yasin": 36, "as-saffat": 37, "sad": 38,
    "az-zumar": 39, "ghafir": 40, "fussilat": 41,
    "ash-shura": 42, "az-zukhruf": 43, "ad-dukhan": 44,
    "al-jathiyah": 45, "al-ahqaf": 46, "muhammad": 47,
    "al-fath": 48, "al-hujurat": 49, "qaf": 50,
    "adh-dhariyat": 51, "at-tur": 52, "an-najm": 53,
    "al-qamar": 54, "ar-rahman": 55, "al-waqiah": 56,
    "al-hadid": 57, "al-mujadila": 58, "al-hashr": 59,
    "al-mumtahanah": 60, "as-saf": 61, "al-jumuah": 62,
    "al-munafiqun": 63, "at-taghabun": 64, "at-talaq": 65,
    "at-tahrim": 66, "al-mulk": 67, "al-qalam": 68,
    "al-haqqah": 69, "al-maarij": 70, "nuh": 71,
    "al-jinn": 72, "al-muzzammil": 73, "al-muddaththir": 74,
    "al-qiyamah": 75, "al-insan": 76, "al-mursalat": 77,
    "an-naba": 78, "an-naziat": 79, "abasa": 80,
    "at-takwir": 81, "al-infitar": 82, "al-mutaffifin": 83,
    "al-inshiqaq": 84, "al-buruj": 85, "at-tariq": 86,
    "al-ala": 87, "al-ghashiyah": 88, "al-fajr": 89,
    "al-balad": 90, "ash-shams": 91, "al-layl": 92,
    "ad-duha": 93, "ash-sharh": 94, "at-tin": 95,
    "al-alaq": 96, "al-qadr": 97, "al-bayyinah": 98,
    "az-zalzalah": 99, "al-adiyat": 100, "al-qariah": 101,
    "at-takathur": 102, "al-asr": 103, "al-humazah": 104,
    "al-fil": 105, "quraysh": 106, "al-maun": 107,
    "al-kawthar": 108, "al-kafirun": 109, "an-nasr": 110,
    "al-masad": 111, "al-ikhlas": 112, "al-falaq": 113,
    "an-nas": 114,
}


def _normalise_key(raw: str) -> str | None:
    """
    Accept inputs like:
      "2:255"  "2/255"  "Al-Baqarah:255"  "baqarah 255"
    Returns "surah:ayah" string or None if unparseable.
    """
    raw = raw.strip()
    # already numeric  e.g. "2:255" or "2/255"
    m = re.fullmatch(r"(\d+)[:/\s](\d+)", raw)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    # name:ayah  e.g. "Al-Baqarah:255" or "baqarah 255"
    m = re.fullmatch(r"([a-zA-Z\s\-]+)[:/\s]+(\d+)", raw)
    if m:
        name = m.group(1).strip().lower()
        ayah = m.group(2)
        # strip "surah" prefix if present
        name = re.sub(r"^surah\s*", "", name).strip()
        surah_num = _SURAH_NAMES.get(name)
        if surah_num:
            return f"{surah_num}:{ayah}"
    return None


async def quran_verse(verse_key: str) -> str:
    """
    Fetch a Quranic verse by its key (e.g. '2:255' for Ayat al-Kursi,
    or 'Al-Baqarah:255').  Returns Arabic text + English translation
    (Dr. Mustafa Khattab) + verse reference.
    """
    key = _normalise_key(verse_key)
    if not key:
        return (
            f"Could not parse verse key '{verse_key}'. "
            "Use format 'surah:ayah' e.g. '2:255' or 'Al-Baqarah:255'."
        )

    url = f"https://api.quran.com/api/v4/verses/by_key/{key}"
    params = {
        "language": "en",
        "words": "false",
        "translations": "131",          # Dr. Mustafa Khattab – The Clear Quran
        "fields": "text_uthmani",
    }
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 404:
            return f"Verse '{key}' not found. Check surah and ayah numbers."
        resp.raise_for_status()
        data = resp.json().get("verse", {})

        arabic   = data.get("text_uthmani", "")
        translations = data.get("translations", [])
        english  = translations[0].get("text", "") if translations else ""
        # strip HTML tags sometimes present in translation text
        english  = re.sub(r"<[^>]+>", "", english).strip()
        verse_id = data.get("verse_key", key)

        lines = [
            f"Quran {verse_id}",
            "",
            f"Arabic:  {arabic}",
            "",
            f"English: {english}",
        ]
        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return "Error: request to quran.com timed out."
    except Exception as e:
        return f"Error fetching verse: {e}"


async def parsa_tool(query: str) -> str:
    """
    Searches the internal semantic search API for religious content.
    Best used with Persian queries.
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
    try:
        response = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        search_results = data.get("data", {}).get("result", [])
        items = []
        for res in search_results:
            highlight = res.get("highlight", "")
            match = re.search(r"\u067e\u0627\u0633\u062e:\s*(.*)", highlight)
            snippet = match.group(1).strip() if match else highlight
            items.append(
                f"[ID:{res.get('id')}] {res.get('content')} | {snippet[:600]} | {res.get('source_link', '')}"
            )
        return "\n\n".join(items) if items else "No results found."
    except Exception as e:
        return f"Error: {e}"


async def haditha_tool(query: str) -> str:
    """Search for Ahadith from AhlulBait."""
    return f"HADITHA RESULT: (Mock) Searching for Ahadith related to '{query}'..."


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
BUILTIN_TOOLS = {
    "parsa":       parsa_tool,
    "haditha":     haditha_tool,
    "quran_verse": quran_verse,
}

BUILTIN_TOOL_DEFINITIONS = [
    {
        "name": "parsa",
        "description": "Searches for religious content. Best used with Persian queries.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "haditha",
        "description": "Searches for Ahadith from AhlulBait.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "quran_verse",
        "description": (
            "Fetch the exact Arabic text and English translation of a Quranic verse by its key. "
            "Use this whenever you want to cite or display an actual verse from the Quran. "
            "Accepts formats like '2:255', 'Al-Baqarah:255', or 'baqarah 255'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "verse_key": {
                    "type": "string",
                    "description": (
                        "The verse identifier. Examples: '2:255' (Ayat al-Kursi), "
                        "'1:1' (Al-Fatihah opening), 'Al-Ikhlas:1', '112:1'."
                    ),
                }
            },
            "required": ["verse_key"],
        },
    },
]


def make_sync(async_fn):
    """Wrap an async function into a sync callable."""
    def wrapper(**kwargs):
        with ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(async_fn(**kwargs))).result()
    wrapper.__name__ = getattr(async_fn, "__name__", "tool")
    return wrapper
