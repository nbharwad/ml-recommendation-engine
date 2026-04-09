"""Session memory manager for conversational context."""

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as redis
from langchain_openai import AzureChatOpenAI

from app.config import settings
from app.observability.logger import get_logger

ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class MemoryManager:
    """Session and long-term memory manager with FIFO eviction and LLM summarization."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        llm_client: Optional[AzureChatOpenAI] = None,
    ):
        """Initialize memory manager."""
        self._client = redis_client
        self._llm = llm_client
        self._max_turns = settings.session_memory_max_turns
        self._ttl = settings.session_memory_ttl
        self._lt_ttl = settings.long_term_memory_ttl
        self._max_summaries = settings.long_term_memory_max_summaries
        self._logger = get_logger(__name__)

    def _validate_id(self, id_value: str, id_type: str) -> None:
        """Validate session_id or user_id to prevent injection attacks."""
        if not id_value or not ID_PATTERN.match(id_value):
            raise ValueError(f"Invalid {id_type}: must be alphanumeric with dash/underscore only")

    def _decode_redis(self, data: Any) -> Optional[str]:
        """Consistently decode bytes from Redis responses."""
        if data is None:
            return None
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data
        return str(data)

    def _get_key(self, session_id: str) -> str:
        """Build Redis key for session."""
        self._validate_id(session_id, "session_id")
        return f"session:{session_id}"

    def _get_lt_key(self, user_id: str) -> str:
        """Build Redis key for long-term user memory."""
        self._validate_id(user_id, "user_id")
        return f"ltmem:{user_id}"

    async def set_redis_client(self, client: redis.Redis) -> None:
        """Set Redis client after initialization."""
        self._client = client

    async def get_session(self, session_id: str) -> list[dict]:
        """
        Get session memory as list of turns.

        Args:
            session_id: Session identifier

        Returns:
            List of turns with role, content, trace_id, timestamp
        """
        if not self._client:
            return []

        try:
            key = self._get_key(session_id)
            data = await self._client.get(key)

            if data is None:
                return []

            turns = json.loads(data)

            await self._client.expire(key, self._ttl)

            return turns

        except redis.RedisError as e:
            self._logger.warning(
                "Failed to get session memory",
                extra={"session_id": session_id, "error": str(e)},
            )
            return []
        except json.JSONDecodeError as e:
            self._logger.warning(
                "Corrupted session memory, resetting",
                extra={"session_id": session_id, "error": str(e)},
            )
            await self.clear_session(session_id)
            return []
        except Exception as e:
            self._logger.warning(
                "Unexpected error getting session",
                extra={"session_id": session_id, "error": str(e)},
            )
            return []

    async def get_long_term_memory(self, user_id: str, limit: int = 5) -> list[str]:
        """
        Get long-term memory summaries for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of recent summaries to return

        Returns:
            List of summary strings
        """
        if not self._client:
            return []

        try:
            key = self._get_lt_key(user_id)
            data = await self._client.lrange(key, 0, limit - 1)
            return [self._decode_redis(d) for d in data if self._decode_redis(d)]

        except redis.RedisError as e:
            self._logger.warning(
                "Failed to get long-term memory",
                extra={"user_id": user_id, "error": str(e)},
            )
            return []
        except Exception as e:
            self._logger.warning(
                "Unexpected error getting long-term memory",
                extra={"user_id": user_id, "error": str(e)},
            )
            return []

    async def save_to_long_term(self, user_id: str, summary: str) -> bool:
        """
        Save a summary to long-term memory.

        Args:
            user_id: User identifier
            summary: Concise summary string

        Returns:
            True on success, False on failure
        """
        if not self._client:
            return False

        try:
            key = self._get_lt_key(user_id)
            # Push to front of list (most recent first)
            await self._client.lpush(key, summary)
            # Set TTL (30 days)
            await self._client.expire(key, self._lt_ttl)

            # Check if we need to merge
            count = await self._client.llen(key)
            if count > self._max_summaries:
                await self._merge_old_summaries(user_id)

            return True

        except redis.RedisError as e:
            self._logger.warning(
                "Failed to save to long-term memory",
                extra={"user_id": user_id, "error": str(e)},
            )
            return False
        except Exception as e:
            self._logger.warning(
                "Unexpected error saving long-term memory",
                extra={"user_id": user_id, "error": str(e)},
            )
            return False

    async def _summarize_session(self, user_id: str, turns: list[dict]) -> Optional[str]:
        """
        Compress session turns into a concise summary using LLM.

        Args:
            user_id: User identifier
            turns: List of turns to summarize

        Returns:
            Summary string or None if failed
        """
        if not self._llm or not turns:
            return None

        try:
            # Format transcript for summarization
            transcript = []
            for t in turns:
                transcript.append(f"{t['role']}: {t['content']}")
            transcript_str = "\n".join(transcript)

            import importlib.resources

            system_prompt = (
                importlib.resources.files("app.agents.prompts")
                .joinpath("memory_summary_system.txt")
                .read_text(encoding="utf-8")
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Please summarize the following RCM session transcript:\n\n{transcript_str}",
                },
            ]

            response = await self._llm.ainvoke(messages)
            summary = response.content.strip()

            if summary:
                await self.save_to_long_term(user_id, summary)
                return summary

            return None

        except Exception as e:
            self._logger.warning(
                "Failed to summarize session",
                extra={"user_id": user_id, "error": str(e)},
            )
            return None

    async def _merge_old_summaries(self, user_id: str) -> bool:
        """
        Merge the two oldest long-term summaries if count > 20.

        Args:
            user_id: User identifier

        Returns:
            True on success, False on failure
        """
        if not self._client or not self._llm:
            return False

        try:
            key = self._get_lt_key(user_id)
            summaries = await self._client.lrange(key, -2, -1)
            if len(summaries) < 2:
                return True

            s1 = self._decode_redis(summaries[0])
            s2 = self._decode_redis(summaries[1])
            if not s1 or not s2:
                return True

            import importlib.resources

            system_prompt = (
                importlib.resources.files("app.agents.prompts")
                .joinpath("memory_merge_system.txt")
                .read_text(encoding="utf-8")
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Merge these two RCM summaries:\n\nSummary 1: {s1}\n\nSummary 2: {s2}",
                },
            ]

            response = await self._llm.ainvoke(messages)
            merged = response.content.strip()

            if merged:
                async with self._client.pipeline() as pipe:
                    pipe.rpop(key, count=2)
                    pipe.rpush(key, merged)
                    await pipe.execute()
                return True
                return True

            return False

        except Exception as e:
            self._logger.warning(
                "Failed to merge old summaries",
                extra={"user_id": user_id, "error": str(e)},
            )
            return False

    async def inject_memory(self, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Format session and long-term memory for AgentState injection.
        Recent session context is prioritized over historical context.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier for long-term memory

        Returns:
            Context string to inject into system prompt
        """
        session_context = await self.get_context_string(session_id)

        long_term_context = ""
        if user_id:
            summaries = await self.get_long_term_memory(user_id)
            if summaries:
                long_term_context = "\n".join([f"- {s}" for s in summaries])

        full_context = []

        # Session context first (more relevant - recent conversation)
        if session_context:
            full_context.append("### Recent Conversation Context (Current Session):")
            full_context.append(session_context)
            full_context.append("")

        # Long-term context second (less relevant - historical)
        if long_term_context:
            full_context.append("### Relevant Historical Context (Long-term Memory):")
            full_context.append(long_term_context)

        return "\n".join(full_context).strip()

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        trace_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Add a turn to session memory with FIFO eviction and auto-summarization.
        Uses optimistic locking to prevent race conditions.

        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
            trace_id: Trace identifier
            user_id: Optional user identifier for long-term memory

        Returns:
            True on success, False on failure
        """
        if not self._client:
            return False

        key = self._get_key(session_id)
        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with self._client.pipeline() as pipe:
                    while True:
                        try:
                            await pipe.watch(key)
                            existing_data = await self._client.get(key)
                            turns = json.loads(existing_data) if existing_data else []

                            new_turn = {
                                "role": role,
                                "content": content,
                                "trace_id": trace_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "summarized": False,
                            }

                            turns.append(new_turn)

                            if len(turns) > self._max_turns:
                                turns_to_evict = turns[: -self._max_turns]
                                if user_id and any(
                                    not t.get("summarized", False) for t in turns_to_evict
                                ):
                                    await self._summarize_session(user_id, turns_to_evict)
                                    for t in turns:
                                        t["summarized"] = True

                                turns = turns[-self._max_turns :]

                            pipe.multi()
                            pipe.setex(key, self._ttl, json.dumps(turns))
                            await pipe.execute()
                            return True

                        except redis.WatchError:
                            continue

            except redis.RedisError as e:
                if attempt == max_retries - 1:
                    self._logger.warning(
                        "Failed to add turn to session after retries",
                        extra={"session_id": session_id, "error": str(e), "attempts": max_retries},
                    )
                    return False
                continue
            except (TypeError, ValueError) as e:
                self._logger.warning(
                    "Failed to serialize session turn",
                    extra={"session_id": session_id, "error": str(e)},
                )
                return False
            except Exception as e:
                self._logger.warning(
                    "Unexpected error adding session turn",
                    extra={"session_id": session_id, "error": str(e)},
                )
                return False

        return False

    async def clear_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """
        Clear session memory.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier for long-term memory

        Returns:
            True if deleted, False otherwise
        """
        if not self._client:
            return False

        try:
            # Summarize before clearing if user_id is provided and there are unsummarized turns
            if user_id:
                turns = await self.get_session(session_id)
                if any(not t.get("summarized", False) for t in turns):
                    await self._summarize_session(user_id, turns)

            key = self._get_key(session_id)
            result = await self._client.delete(key)
            return result > 0

        except redis.RedisError as e:
            self._logger.warning(
                "Failed to clear session",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False
        except Exception as e:
            self._logger.warning(
                "Unexpected error clearing session",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False

    async def get_context_string(self, session_id: str, max_turns: Optional[int] = None) -> str:
        """
        Get session context as formatted string for LLM prompt.

        Args:
            session_id: Session identifier
            max_turns: Optional override for number of turns to include

        Returns:
            Formatted conversation context string
        """
        turns = await self.get_session(session_id)

        if not turns:
            return ""

        limit = max_turns or self._max_turns
        recent_turns = turns[-limit:]

        context_parts = []
        for turn in recent_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)


async def create_memory_manager(
    redis_client: Optional[redis.Redis] = None,
    llm_client: Optional[AzureChatOpenAI] = None,
) -> MemoryManager:
    """Factory function to create memory manager."""
    return MemoryManager(redis_client=redis_client, llm_client=llm_client)
