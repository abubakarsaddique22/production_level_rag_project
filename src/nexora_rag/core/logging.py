"""
Structured JSON logging for Nexora RAG.

Every log line is a JSON object with a timestamp, level, logger name,
message and (when available) a request_id so logs can be correlated
across the ingestion pipeline and the API.

IMPORTANT: never log secrets, API keys, JWTs, or full user documents.
Log identifiers (doc_id, chunk_id, user_id, trace_id) instead.

Usage:
    from nexora_rag.core.logging import get_logger
    log = get_logger(__name__)
    log.info("chat_request_received", extra={"request_id": rid, "user_id": uid})
"""

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

# request_id is set per-request by API middleware and picked up automatically
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class JSONFormatter(logging.Formatter):
    RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_ctx.get()),
        }

        # include any extra fields passed via `extra={...}`
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Call once at process startup (API entrypoint, ingestion script, etc.)."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # avoid duplicate handlers on reload
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # quiet down noisy third-party loggers
    for noisy in ("httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
