"""
MPP (Machine Payments Protocol) exception hierarchy.

All exceptions raised by the SDK inherit from :class:`MppError`, making it
easy to catch any SDK-specific error with a single ``except MppError`` clause.
"""

from __future__ import annotations


class MppError(Exception):
    """Base class for all MPP SDK errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MppFaucetError(MppError):
    """
    Raised when the devnet/testnet SOL faucet (airdrop) fails after all retries.

    Attributes
    ----------
    address:
        The base58 wallet address that was being funded.
    """

    address: str

    def __init__(self, address: str, cause: BaseException | None = None) -> None:
        super().__init__(
            f"Failed to airdrop SOL to wallet {address}. "
            "The devnet/testnet faucet may be rate-limited. "
            "Wait 30s and retry, or pass a pre-funded secret_key to skip airdrop."
        )
        self.address = address
        self.__cause__ = cause


class MppPaymentError(MppError):
    """
    Raised when a fetch request fails due to a payment-related issue.

    This covers non-402 HTTP error responses, malformed Payment-Request headers,
    and similar protocol violations.

    Attributes
    ----------
    url:
        The URL that was being fetched.
    status:
        The HTTP status code returned by the server.
    """

    url: str
    status: int

    def __init__(self, url: str, status: int, cause: BaseException | None = None) -> None:
        super().__init__(f"Payment failed for {url} (status: {status})")
        self.url = url
        self.status = status
        self.__cause__ = cause


class MppTimeoutError(MppError):
    """
    Raised when the full MPP fetch flow (wallet + payment + retry) exceeds the timeout.

    Attributes
    ----------
    url:
        The URL that was being fetched.
    timeout_ms:
        The timeout in milliseconds that was exceeded.
    """

    url: str
    timeout_ms: int

    def __init__(self, url: str, timeout_ms: int) -> None:
        super().__init__(
            f"Request to {url} timed out after {timeout_ms}ms. "
            "Increase the timeout option or check your Solana RPC connection."
        )
        self.url = url
        self.timeout_ms = timeout_ms


class MppNetworkError(MppError):
    """
    Raised when the network configuration is invalid.

    The most common cause is attempting to use mainnet without supplying a
    pre-funded ``secret_key`` (airdrop is not available on mainnet).

    Attributes
    ----------
    network:
        The network identifier that caused the error (e.g. ``"mainnet"``).
    """

    network: str

    def __init__(self, network: str, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                f'Network configuration error for "{network}". '
                "Mainnet requires a pre-funded secret_key (no airdrop available)."
            )
        )
        self.network = network
