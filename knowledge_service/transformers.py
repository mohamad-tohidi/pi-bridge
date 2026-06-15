import re
from typing import Iterator
from pi_bridge.types import ResponseEvent, TextDeltaEvent, AgentEndEvent, TurnEndEvent, ResponseEventTransformer


class LinkEnforcementTransformer(ResponseEventTransformer):
    """
    A transformer that enforces link format [[LINK:ID]] instead of raw URLs.
    It buffers text and processes it whenever a delimiter is encountered.
    """
    def __init__(self):
        self._buffer = ""
        self._url_pattern = re.compile(r"https?://[^\s<>\"']+")
        self._delimiters = set(" \n\t\r\"'<>)]}")

    def transform(self, event_iterator: Iterator[ResponseEvent]) -> Iterator[ResponseEvent]:
        for event in event_iterator:
            if isinstance(event, TextDeltaEvent):
                self._buffer += event.delta
                
                # Look for delimiters in the new delta
                # We want to find the last delimiter in the buffer to know what we can safely yield
                last_delimiter_idx = -1
                for i, char in enumerate(self._buffer):
                    if char in self._delimiters:
                        last_delimiter_idx = i
                
                if last_delimiter_idx != -1:
                    # We can safely yield up to the last delimiter (including the delimiter)
                    # But wait, if the delimiter is part of a URL (like a parenthesis), we shouldn't yield it yet.
                    # Actually, our delimiter set includes ")".
                    
                    # Let's just yield everything up to the last delimiter
                    # that is NOT part of a URL. This is still tricky.
                    
                    # Let's simplify: yield everything up to the last delimiter, 
                    # and keep the rest in the buffer.
                    
                    to_yield = self._buffer[:last_delimiter_idx + 1]
                    self._buffer = self._buffer[last_delimiter_idx + 1:]
                    
                    # Process the text to be yielded
                    processed_text = self._url_pattern.sub(
                        lambda m: f"[[LINK:{hash(m.group(0)) % 1000}]]", 
                        to_yield
                    )
                    
                    if processed_text:
                        yield TextDeltaEvent(delta=processed_text, type="text_delta")
                
            elif isinstance(event, (AgentEndEvent, TurnEndEvent)):
                # Flush the remaining buffer
                if self._buffer:
                    processed_text = self._url_pattern.sub(
                        lambda m: f"[[LINK:{hash(m.group(0)) % 1000}]]", 
                        self._buffer
                    )
                    if processed_text:
                        yield TextDeltaEvent(delta=processed_text, type="text_delta")
                    self._buffer = ""
                yield event
            else:
                yield event
