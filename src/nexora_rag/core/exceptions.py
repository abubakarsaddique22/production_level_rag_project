"""
Custom exception hierarchy for Nexora RAG.

Every exception carries an HTTP-friendly `status_code` and a machine
readable `code` so the FastAPI error handler (added in Step P) can turn
any of these into a consistent JSON error response without each router
needing its own try/except block.
"""


class NexoraError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str | None = None):
        self.message = message or self.__class__.__doc__ or self.code
        super().__init__(self.message)


class ConfigError(NexoraError):
    """Invalid or missing configuration."""

    status_code = 500
    code = "config_error"


class DocumentParsingError(NexoraError):
    """A source document could not be parsed."""

    status_code = 422
    code = "document_parsing_error"


class ChunkValidationError(NexoraError):
    """A chunk failed schema validation during ingestion."""

    status_code = 422
    code = "chunk_validation_error"


class EmbeddingError(NexoraError):
    """The embedding model or embedding API call failed."""

    status_code = 502
    code = "embedding_error"


class VectorStoreError(NexoraError):
    """The vector database could not be reached or the query failed."""

    status_code = 502
    code = "vector_store_error"


class LLMError(NexoraError):
    """The LLM provider call failed after retries."""

    status_code = 502
    code = "llm_error"


class AuthenticationError(NexoraError):
    """Invalid credentials or missing/expired token."""

    status_code = 401
    code = "authentication_error"


class AuthorizationError(NexoraError):
    """The authenticated user is not allowed to access this resource."""

    status_code = 403
    code = "authorization_error"


class RateLimitExceededError(NexoraError):
    """The caller has exceeded their allotted request rate."""

    status_code = 429
    code = "rate_limit_exceeded"


class GuardrailViolationError(NexoraError):
    """Input or output failed a safety/guardrail check."""

    status_code = 400
    code = "guardrail_violation"


class NotFoundError(NexoraError):
    """The requested resource does not exist."""

    status_code = 404
    code = "not_found"
