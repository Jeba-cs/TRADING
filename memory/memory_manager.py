# memory/memory_manager.py
"""
Simple memory manager for trade and conversation storage
"""

class MemoryManager:
    def __init__(self):
        self.trade_history = []
        self.chat_history = []

    def store_trade(self, trade_record):
        self.trade_history.append(trade_record)

    def store_chat_message(self, message):
        self.chat_history.append(message)

    def get_trade_history(self):
        return self.trade_history

    def get_recent_chat(self, n=10):
        return self.chat_history[-n:]
