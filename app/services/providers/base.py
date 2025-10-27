class ProviderError(Exception):
    pass

class RateLimitProviderError(ProviderError):
    """Wraps a RateLimitExceeded to signal the chain to try the next provider."""
    pass
