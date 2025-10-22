from typing import Dict, Any, Optional, List
import redis.asyncio as redis
import json
from loguru import logger
from datetime import datetime, timezone

from app.models.common import UserID # Assuming UserID is defined in app/models/common
# from app.schemas.context import DynamicContext # This import seems unused in the provided code, keeping for now if needed elsewhere

class ContextInjectorService: # Renamed class from ContextInjectorSerivce
    REDIS_CONTEXT_KEY_PREFIX = "user_context:"
    REDIS_CHAT_HISTORY_KEY_PREFIX = "chat_history:" # New prefix for chat history

    # Max number of messages to store in chat history per user in Redis.
    # This helps prevent the list from growing indefinitely.
    # A larger number might be needed if you rely heavily on Redis for full history.
    # For LLM context window, this will be dynamically trimmed, but this is for storage limit.
    MAX_CHAT_HISTORY_LENGTH = 100 

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    async def _get_raw_context(self, user_id: UserID) -> Dict[str, str]:
        """Retrieve the raw context hash from Redis for a given user_id."""
        context_key = f"{self.REDIS_CONTEXT_KEY_PREFIX}{user_id}"
        return await self.redis_client.hgetall(context_key)

    async def get_current_context(self, user_id: UserID) -> Optional[Dict[str, Any]]:
        """Retrieve the current context for a user, parsing JSON values."""
        raw_context = await self._get_raw_context(user_id)
        if not raw_context:
            logger.info(f"No context found for user_id:{user_id}")
            return None
        parsed_context = {}
        for key, value_bytes in raw_context.items(): # values from hgetall are bytes
            key_str = key.decode('utf-8') # Decode key bytes to string
            try:
                # Attempt to decode value bytes and then parse as JSON
                parsed_context[key_str] = json.loads(value_bytes.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If it's not JSON or not UTF-8, store as decoded string (or raw bytes if preferred)
                parsed_context[key_str] = value_bytes.decode('utf-8', errors='ignore') # Decode with error handling

        logger.info(f"Retrieved context for user_id:{user_id}")
        return parsed_context

    async def update_context(
        self, user_id: UserID, new_context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Updates or merges new context data for a user.
        Values are stored as JSON strings in a Redis hash.
        """
        context_key = f"{self.REDIS_CONTEXT_KEY_PREFIX}{user_id}"
        
        # Prepare data, serializing all values to JSON strings
        # Also ensure 'updated_at' is always set here
        data_to_store = {k: json.dumps(v) for k, v in new_context_data.items()}
        data_to_store["updated_at"] = datetime.now(timezone.utc).isoformat() 

        if data_to_store:
            await self.redis_client.hset(context_key, mapping=data_to_store)
            logger.info(f"Context updated for user_id: {user_id}")
        else:
            logger.warning(
                f"Attempted to update context for user_id: {user_id} with empty data."
            )
        return await self.get_current_context(user_id=user_id)

    async def delete_context(self, user_id: UserID) -> bool:
        """Deletes a user's entire general context and chat history."""
        context_key = f"{self.REDIS_CONTEXT_KEY_PREFIX}{user_id}"
        chat_history_key = f"{self.REDIS_CHAT_HISTORY_KEY_PREFIX}{user_id}"

        # Delete both the general context hash and the chat history list
        deleted_count = await self.redis_client.delete(context_key, chat_history_key)
        
        if deleted_count > 0:
            logger.info(f"Context and/or chat history deleted for user_id: {user_id}. Deleted {deleted_count} keys.")
            return True
        logger.info(f"No context or chat history found to delete for user_id: {user_id}")
        return False

    async def add_message_to_history(self, user_id: UserID, role: str, content: str):
        """
        Adds a new message (user or AI) to the user's chat history list in Redis.
        Maintains a maximum length for the history.
        """
        chat_history_key = f"{self.REDIS_CHAT_HISTORY_KEY_PREFIX}{user_id}"
        
        message_data = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Store message as a JSON string in the Redis list
        # LPUSH adds to the head of the list. We'll read it in reverse to get chronological order.
        await self.redis_client.lpush(chat_history_key, json.dumps(message_data))
        
        # Trim the list to MAX_CHAT_HISTORY_LENGTH, keeping only the most recent messages (from the head)
        await self.redis_client.ltrim(chat_history_key, 0, self.MAX_CHAT_HISTORY_LENGTH - 1)
        logger.debug(f"Added message to history for user_id:{user_id}. Current length trimmed to {self.MAX_CHAT_HISTORY_LENGTH}")

    async def get_chat_history(self, user_id: UserID, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieves a user's chat history from Redis, up to a specified limit.
        Messages are returned in chronological order (oldest first).
        """
        chat_history_key = f"{self.REDIS_CHAT_HISTORY_KEY_PREFIX}{user_id}"
        
        # LRANGE 0 -1 gets all elements. If limit is set, get from 0 to limit-1.
        # Note: LPUSH adds to the left, so to get chronological order (oldest first),
        # we'll read from the list and then reverse it.
        # If we need the N most recent messages, we read LRANGE 0 N-1 from the LPUSHed list
        # which means reading the HEAD. Then reverse for chronological or keep as is for reverse chronological.
        
        # Retrieve all messages (or up to limit if specified)
        # We fetch up to the limit from the 'left' (most recent) of the list
        # and then reverse it to present it chronologically.
        raw_messages = await self.redis_client.lrange(chat_history_key, 0, (limit - 1) if limit else -1)
        
        history = []
        for msg_bytes in reversed(raw_messages): # Reverse to get chronological order
            try:
                history.append(json.loads(msg_bytes.decode('utf-8')))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"Failed to decode or parse chat history message for user {user_id}: {msg_bytes}. Error: {e}")
        
        logger.info(f"Retrieved {len(history)} chat messages for user_id:{user_id}")
        return history