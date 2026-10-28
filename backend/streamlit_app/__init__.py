"""IntelliAssist AI - Streamlit Application Package."""

try:
    from backend.streamlit_app.api_client import (
        ApiClientError,
        ApiConnectionError,
        ApiPayloadError,
        ApiStatusError,
        IntelliAssistApiClient,
    )
except ImportError:
    from streamlit_app.api_client import (  # type: ignore
        ApiClientError,
        ApiConnectionError,
        ApiPayloadError,
        ApiStatusError,
        IntelliAssistApiClient,
    )

__all__ = [
    "ApiClientError",
    "ApiConnectionError",
    "ApiPayloadError",
    "ApiStatusError",
    "IntelliAssistApiClient",
]
