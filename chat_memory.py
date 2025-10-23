# chat_memory.py
"""
Chat Memory Module
------------------
Implements a simple sliding window memory that stores recent user-bot pairs.
"""

from collections import deque
from typing import Deque, Tuple
import re 

class ChatMemory:
    """
    Maintains short-term memory of recent conversation turns.
    """

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self.memory: Deque[Tuple[str, str]] = deque(maxlen=max_turns)
        self.separator = "\n"

    def add_turn(self, user_text: str, bot_text: str):
        """
        Add a user-bot exchange to the memory.
        """
        self.memory.append((user_text.strip(), bot_text.strip()))

    def clear(self):
        """
        Clears memory history.
        """
        self.memory.clear()

    def get_context_text(self) -> str:
        """
        Returns concatenated dialogue context formatted for LLM input.
        """
        exchanges = []
        # Format context: User: Q\nBot: A
        for user, bot in self.memory:
            exchanges.append(f"User: {user}{self.separator}Bot: {bot}")
            
        return self.separator.join(exchanges)