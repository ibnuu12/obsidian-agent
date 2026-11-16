
class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMAuthenticationError(LLMError):
    """Raised when provider authentication fails."""


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits are exceeded."""


class LLMTimeoutError(LLMError):
    """Raised when a provider request times out."""


class LLMProviderError(LLMError):
    """Raised when a provider returns an unexpected error."""


class LLMInvalidResponseError(LLMError):
    """Raised when a provider response cannot be normalized."""

class LLMNotConfiguredError(LLMError):
    """Raised when no LLM provider has been configured."""