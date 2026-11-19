"""Custom exceptions used across the project."""


class RagProjectError(Exception):
    """Base class for all project errors."""


class ConfigError(RagProjectError):
    """Configuration is missing or invalid."""


class FetchError(RagProjectError):
    """A source could not be downloaded or parsed."""


class IndexingError(RagProjectError):
    """The vector index could not be built or queried."""


class ProviderError(RagProjectError):
    """An LLM provider call failed."""


class MissingCredentialError(ProviderError):
    """The API key of the selected provider is not set."""


class RateLimitError(ProviderError):
    """An LLM provider rate-limited the request."""
