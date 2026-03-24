import os
import logging
from typing import Callable

try:
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq import Retry
except Exception:  # pragma: no cover - optional dependency
    Redis = None
    Queue = None
    Job = None
    Retry = None

logger = logging.getLogger("contexta.queue")


class OptionalRQQueue:
    """Optional Redis/RQ queue wrapper with graceful fallback when unavailable."""

    def __init__(self) -> None:
        backend = (os.getenv("TASK_QUEUE_BACKEND", "auto") or "auto").strip().lower()
        required_env = (os.getenv("TASK_QUEUE_REQUIRED", "false") or "false").strip().lower() == "true"
        redis_url = (os.getenv("REDIS_URL") or "").strip()
        queue_name = (os.getenv("TASK_QUEUE_NAME") or "contexta").strip()
        timeout = int(os.getenv("TASK_QUEUE_JOB_TIMEOUT_SECONDS", "1800"))
        retry_max = int(os.getenv("TASK_QUEUE_RETRY_MAX", "2"))
        result_ttl = int(os.getenv("TASK_QUEUE_RESULT_TTL_SECONDS", "3600"))
        failure_ttl = int(os.getenv("TASK_QUEUE_FAILURE_TTL_SECONDS", "604800"))
        job_ttl = int(os.getenv("TASK_QUEUE_JOB_TTL_SECONDS", "86400"))
        retry_intervals_raw = (os.getenv("TASK_QUEUE_RETRY_INTERVALS", "2,5") or "").strip()

        self.backend = backend
        self.required = required_env or backend == "rq"
        self.enabled = False
        self.timeout = max(30, timeout)
        self.retry_max = max(0, retry_max)
        self.result_ttl = max(60, result_ttl)
        self.failure_ttl = max(300, failure_ttl)
        self.job_ttl = max(300, job_ttl)
        self.retry_intervals = [
            int(v.strip())
            for v in retry_intervals_raw.split(",")
            if v.strip().isdigit()
        ]
        self._queue = None

        if backend == "local":
            return

        if not redis_url:
            if backend == "rq":
                logger.warning("TASK_QUEUE_BACKEND=rq but REDIS_URL is missing; falling back to local mode.")
            return

        if Redis is None or Queue is None:
            logger.warning("redis/rq packages are unavailable; falling back to local mode.")
            return

        try:
            conn = Redis.from_url(redis_url)
            conn.ping()
            self._queue = Queue(name=queue_name, connection=conn, default_timeout=self.timeout)
            self.enabled = True
        except Exception as e:  # pragma: no cover - depends on external service
            logger.warning(f"Failed to initialize RQ queue, falling back to local mode: {e}")
            self.enabled = False

    def unavailable_reason(self) -> str:
        if self.enabled:
            return ""
        if self.backend == "local":
            return "TASK_QUEUE_BACKEND=local"
        return "External queue is required/configured but not available"

    def enqueue(self, func: Callable, *, namespace: str, kind: str, **kwargs) -> str:
        if not self.enabled or self._queue is None:
            raise RuntimeError("RQ queue is not enabled")

        retry_obj = None
        if Retry is not None and self.retry_max > 0:
            retry_obj = Retry(max=self.retry_max, interval=(self.retry_intervals or [2, 5]))

        job = self._queue.enqueue(
            func,
            kwargs=kwargs,
            job_timeout=self.timeout,
            retry=retry_obj,
            result_ttl=self.result_ttl,
            failure_ttl=self.failure_ttl,
            job_ttl=self.job_ttl,
        )
        job.meta["namespace"] = namespace
        job.meta["kind"] = kind
        job.meta["max_attempts"] = max(1, self.retry_max + 1)
        job.save_meta()
        return job.id

    def get(self, job_id: str):
        if not self.enabled or self._queue is None or Job is None:
            return None
        job = Job.fetch(job_id, connection=self._queue.connection)
        status = job.get_status(refresh=True)
        error = None
        if status == "failed":
            error = (job.exc_info or "")[-4000:] if job.exc_info else "Job failed"

        attempts = 1
        if isinstance(job.meta, dict):
            attempts = int(job.meta.get("attempts", 1) or 1)
        return {
            "job_id": job.id,
            "kind": job.meta.get("kind", "unknown"),
            "namespace": job.meta.get("namespace", "unknown"),
            "status": status,
            "created_at": float(job.created_at.timestamp()) if job.created_at else None,
            "updated_at": float(job.ended_at.timestamp()) if job.ended_at else None,
            "result": job.result if status == "finished" else None,
            "error": error,
            "attempts": attempts,
            "max_attempts": int(job.meta.get("max_attempts", max(1, self.retry_max + 1))),
            "dead_letter": (status == "failed"),
        }
