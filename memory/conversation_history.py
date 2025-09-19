# memory/conversation_history.py
"""
Manages conversation history for AI chatbot sessions
"""

from collections import defaultdict
from datetime import datetime

class ConversationHistory:
    def __init__(self):
        # Store messages per user_id
        self._history = defaultdict(list)

    async def add_message(self, user_id: str, role: str, content: str, intent: str = None):
        message_record = {
            'role': role,
            'content': content,
            'intent': intent,
            'timestamp': datetime.now()
        }
        self._history[user_id].append(message_record)

    async def get_recent_messages(self, user_id: str, limit: int = 10):
        return self._history[user_id][-limit:]
