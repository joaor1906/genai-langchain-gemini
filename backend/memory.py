import os
from typing import Dict
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import create_engine

SQLITE_URL = os.getenv("SQLITE_URL", "sqlite:///./data/memory.db")
_engine = create_engine(SQLITE_URL)

_cache: Dict[str, BaseChatMessageHistory] = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _cache:
        os.makedirs("data", exist_ok=True)
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=SQLITE_URL,
        )
        _cache[session_id] = history
    return _cache[session_id]

def build_memory(session_id: str) -> ConversationBufferMemory:
    history = get_history(session_id)
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        chat_memory=history,
    )
    return memory
