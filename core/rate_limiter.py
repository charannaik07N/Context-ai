import hashlib
import logging
import os
import threading
import time
import uuid
from collections import deque

try:
    from redis import Redis
except Exception:  # pragma: no cover - optional dependency
    Redis = None

logger = logging.getLogger("contexta.ratelimit")


class HybridRateLimiter:
    """Rate limiter that supports local in-memory and Redis-backed global windows."""

    _REDIS_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry = 1
    if oldest[2] then
        local oldest_ms = tonumber(oldest[2])
        local remain_ms = window_ms - (now_ms - oldest_ms)
        if remain_ms < 0 then
            remain_ms = 0
        end
        retry = math.floor((remain_ms + 999) / 1000)
        if retry < 1 then
            retry = 1
        end
    end
    redis.call('EXPIRE', key, math.floor(window_ms / 1000) + 5)
    return {0, retry}
end

redis.call('ZADD', key, now_ms, member)
redis.call('EXPIRE', key, math.floor(window_ms / 1000) + 5)
return {1, 0}
"""

    def __init__(self) -> None:
        backend = (os.getenv("RATE_LIMIT_BACKEND", "auto") or "auto").strip().lower()
        required_env = (os.getenv("RATE_LIMIT_REQUIRED", "false") or "false").strip().lower() == "true"
        redis_url = (os.getenv("REDIS_URL") or "").strip()
        key_prefix = (os.getenv("RATE_LIMIT_REDIS_KEY_PREFIX") or "contexta:ratelimit").strip()

        self.backend = backend
        self.required = required_env or backend == "redis"
        self.key_prefix = key_prefix

        self._redis = None
        self._redis_script = None

        self._local_lock = threading.Lock()
        self._local_buckets: dict[str, deque] = {}

        if backend == "local":
            return

        if not redis_url:
            if backend == "redis":
                logger.warning("RATE_LIMIT_BACKEND=redis but REDIS_URL is missing.")
            return

        if Redis is None:
            logger.warning("redis package is unavailable; using local rate limiting.")
            return

        try:
            self._redis = Redis.from_url(redis_url)
            self._redis.ping()
            self._redis_script = self._redis.register_script(self._REDIS_SLIDING_WINDOW_SCRIPT)
        except Exception as e:  # pragma: no cover - depends on external service
            logger.warning(f"Redis rate limiter init failed; using local mode: {e}")
            self._redis = None
            self._redis_script = None

    @property
    def redis_enabled(self) -> bool:
        return self._redis is not None and self._redis_script is not None

    def _redis_key(self, identity_key: str) -> str:
        digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:32]
        return f"{self.key_prefix}:{digest}"

    def check(self, *, identity_key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        cap = max(1, int(limit))
        window = max(1, int(window_seconds))

        if self.redis_enabled:
            try:
                key = self._redis_key(identity_key)
                now_ms = int(time.time() * 1000)
                result = self._redis_script(
                    keys=[key],
                    args=[str(now_ms), str(window * 1000), str(cap), f"{now_ms}:{uuid.uuid4().hex}"],
                )
                allowed = bool(int(result[0]))
                retry_after = int(result[1])
                return allowed, retry_after
            except Exception as e:  # pragma: no cover - depends on external service
                if self.required:
                    raise RuntimeError(f"Redis rate limiter unavailable: {e}")
                logger.warning(f"Redis rate-limit check failed; falling back to local mode: {e}")

        if self.required and self.backend == "redis":
            raise RuntimeError("Global Redis rate limiter is required but unavailable.")

        now = time.time()
        cutoff = now - window
        with self._local_lock:
            bucket = self._local_buckets.get(identity_key)
            if bucket is None:
                bucket = deque()
                self._local_buckets[identity_key] = bucket

            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= cap:
                retry_after = max(1, int(window - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0
