"""
MPP test client — automatic HTTP 402 / Solana payment handling.

Usage::

    from mpp_test_sdk import create_test_client, mpp_fetch, mpp_fetch_reset

    # devnet (default) — zero config
    client = await create_test_client()
    res = await client.fetch("http://localhost:3001/api/data")

    # Or use the module-level shared client
    res = await mpp_fetch("http://localhost:3001/api/data")
    mpp_fetch_reset()   # discard shared client, next call gets a fresh wallet
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import httpx

from .errors import MppFaucetError, MppNetworkError, MppPaymentError, MppTimeoutError

# ─── Constants ────────────────────────────────────────────────────────────────

LAMPORTS_PER_SOL: int = 1_000_000_000

SolanaNetwork = Literal["devnet", "testnet", "mainnet"]

NETWORK_RPC: dict[str, str] = {
    "devnet": "https://api.devnet.solana.com",
    "testnet": "https://api.testnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
}

_AIRDROP_NETWORKS: set[str] = {"devnet", "testnet"}

# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class PaymentStep:
    """
    A single lifecycle event emitted during the MPP payment flow.

    Attributes
    ----------
    type:
        One of ``"wallet-created"``, ``"funded"``, ``"request"``,
        ``"payment"``, ``"retry"``, ``"success"``, or ``"error"``.
    message:
        Human-readable description of the step.
    data:
        Optional structured data associated with the step.
    """

    type: Literal[
        "wallet-created", "funded", "request", "payment", "retry", "success", "error"
    ]
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestClientConfig:
    """
    Configuration for :func:`create_test_client`.

    Attributes
    ----------
    network:
        Solana network. ``"devnet"`` (default) and ``"testnet"`` support free
        airdrops.  ``"mainnet"`` requires a pre-funded *secret_key*.
    secret_key:
        32- or 64-byte keypair seed/secret.  On devnet/testnet this is optional
        (airdrop is used).  On mainnet it is **required**.
    on_step:
        Callback invoked with each :class:`PaymentStep` lifecycle event.
    timeout:
        Full-flow timeout in seconds (wallet creation + payment + retry).
        Default: 30.0.
    rpc_url:
        Override the Solana JSON-RPC endpoint.  Takes precedence over
        *network*.
    """

    network: SolanaNetwork = "devnet"
    secret_key: bytes | None = None
    on_step: Callable[[PaymentStep], None] | None = None
    timeout: float = 30.0
    rpc_url: str | None = None


# ─── TestClient ───────────────────────────────────────────────────────────────


class TestClient:
    """
    An MPP-enabled async HTTP client backed by a Solana wallet.

    Do not instantiate directly; use :func:`create_test_client` instead.

    Attributes
    ----------
    address:
        Base58 public key of the wallet used for payments.
    network:
        Solana network this client is connected to.
    method:
        Always ``"solana"``.
    """

    def __init__(
        self,
        address: str,
        network: SolanaNetwork,
        *,
        keypair: Any,  # solders Keypair
        rpc_url: str,
        on_step: Callable[[PaymentStep], None],
        timeout: float,
    ) -> None:
        self.address: str = address
        self.network: SolanaNetwork = network
        self.method: Literal["solana"] = "solana"
        self._keypair = keypair
        self._rpc_url = rpc_url
        self._on_step = on_step
        self._timeout = timeout

    async def fetch(self, url: str, **kwargs: Any) -> httpx.Response:
        """
        Fetch *url*, automatically handling HTTP 402 MPP payment flows.

        On a 402 response the client:

        1. Parses the ``Payment-Request`` header.
        2. Sends a SOL transfer to the specified recipient.
        3. Retries the original request with a ``Payment-Receipt`` header
           containing the on-chain signature.

        Parameters
        ----------
        url:
            The URL to fetch.
        **kwargs:
            Additional keyword arguments forwarded to :class:`httpx.AsyncClient`
            ``get``/``request`` calls (e.g. ``method``, ``headers``, ``content``).

        Returns
        -------
        httpx.Response

        Raises
        ------
        MppPaymentError:
            On non-402 HTTP error responses or malformed payment headers.
        MppTimeoutError:
            When the full flow exceeds the configured timeout.
        """
        timeout_ms = int(self._timeout * 1000)
        try:
            return await asyncio.wait_for(
                self._fetch_inner(url, **kwargs),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise MppTimeoutError(url, timeout_ms) from exc

    async def _fetch_inner(self, url: str, **kwargs: Any) -> httpx.Response:
        """Inner fetch without the timeout wrapper."""
        emit = self._on_step
        emit(PaymentStep(type="request", message=f"→ {url}", data={"url": url}))

        http_method: str = kwargs.pop("method", "GET").upper()
        headers: dict[str, str] = dict(kwargs.pop("headers", {}))

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Step 1: initial request
            res = await client.request(
                http_method, url, headers=headers, **kwargs
            )

            # Non-402 path
            if res.status_code != 402:
                if not res.is_success:
                    emit(
                        PaymentStep(
                            type="error",
                            message=f"← {res.status_code} {res.reason_phrase}",
                            data={"status": res.status_code},
                        )
                    )
                    raise MppPaymentError(url, res.status_code)
                emit(
                    PaymentStep(
                        type="success",
                        message=f"← {res.status_code} OK",
                        data={"status": res.status_code},
                    )
                )
                return res

            # Step 2: parse Payment-Request header
            payment_request_header = res.headers.get("payment-request", "")
            if not payment_request_header:
                raise MppPaymentError(
                    url,
                    402,
                    ValueError("Server returned 402 without Payment-Request header"),
                )

            params = _parse_header_params(payment_request_header)

            if not params.get("recipient"):
                raise MppPaymentError(
                    url,
                    402,
                    ValueError("Payment-Request header missing recipient field"),
                )
            if not params.get("amount"):
                raise MppPaymentError(
                    url,
                    402,
                    ValueError("Payment-Request header missing amount field"),
                )

            try:
                amount_sol = float(params["amount"])
                if amount_sol <= 0:
                    raise ValueError(f"Invalid payment amount: {params['amount']}")
            except (ValueError, TypeError) as exc:
                raise MppPaymentError(
                    url,
                    402,
                    ValueError(f"Invalid payment amount: {params.get('amount')}"),
                ) from exc

            lamports = round(amount_sol * LAMPORTS_PER_SOL)
            recipient_str: str = params["recipient"]

            emit(
                PaymentStep(
                    type="payment",
                    message=f"Paying {amount_sol} SOL → {recipient_str[:8]}...",
                    data={"amount": amount_sol, "recipient": recipient_str},
                )
            )

            # Step 3: send SOL transfer
            signature = await _send_sol(
                self._rpc_url, self._keypair, recipient_str, lamports
            )

            emit(
                PaymentStep(
                    type="payment",
                    message=f"Confirmed: {signature[:16]}...",
                    data={"signature": signature, "amount": amount_sol},
                )
            )

            # Step 4: retry with Payment-Receipt header
            emit(
                PaymentStep(
                    type="retry",
                    message="↑ Retrying with payment proof",
                    data={"signature": signature},
                )
            )

            receipt_header = (
                f'solana; signature="{signature}"; '
                f'network="{self.network}"; '
                f'amount="{amount_sol}"'
            )
            retry_headers = {**headers, "payment-receipt": receipt_header}

            retry_res = await client.request(
                http_method, url, headers=retry_headers, **kwargs
            )

            emit(
                PaymentStep(
                    type="success" if retry_res.is_success else "error",
                    message=(
                        f"← {retry_res.status_code} "
                        f"{'OK' if retry_res.is_success else retry_res.reason_phrase}"
                    ),
                    data={"status": retry_res.status_code, "signature": signature},
                )
            )

            return retry_res


# ─── JSON-RPC helpers ─────────────────────────────────────────────────────────


async def _rpc_call(rpc_url: str, method: str, params: list[Any]) -> Any:
    """
    Execute a single Solana JSON-RPC call and return ``result``.

    Raises :class:`RuntimeError` if the response contains an ``error`` field.
    """
    async with httpx.AsyncClient(timeout=30) as h:
        r = await h.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"]["message"])
        return data["result"]


async def _get_latest_blockhash(rpc_url: str) -> str:
    """Return the latest confirmed blockhash as a base58 string."""
    result = await _rpc_call(
        rpc_url, "getLatestBlockhash", [{"commitment": "confirmed"}]
    )
    return result["value"]["blockhash"]


async def _request_airdrop(rpc_url: str, pubkey_str: str, lamports: int) -> None:
    """Request a faucet airdrop and poll until confirmed."""
    sig = await _rpc_call(rpc_url, "requestAirdrop", [pubkey_str, lamports])
    for _ in range(60):
        await asyncio.sleep(1)
        statuses = await _rpc_call(
            rpc_url, "getSignatureStatuses", [[sig]]
        )
        status = statuses["value"][0]
        if status and status.get("confirmationStatus") in ("confirmed", "finalized"):
            return
    raise RuntimeError("Airdrop confirmation timed out")


async def _send_sol(
    rpc_url: str,
    keypair: Any,  # solders Keypair
    recipient_str: str,
    lamports: int,
) -> str:
    """
    Build and submit a SOL transfer transaction.

    Returns the transaction signature string after on-chain confirmation.
    """
    from solders.hash import Hash  # noqa: PLC0415
    from solders.message import Message  # noqa: PLC0415
    from solders.pubkey import Pubkey  # noqa: PLC0415
    from solders.system_program import TransferParams, transfer  # noqa: PLC0415
    from solders.transaction import Transaction  # noqa: PLC0415

    recipient = Pubkey.from_string(recipient_str)
    blockhash_str = await _get_latest_blockhash(rpc_url)
    blockhash = Hash.from_string(blockhash_str)

    ix = transfer(TransferParams(from_pubkey=keypair.pubkey(), to_pubkey=recipient, lamports=lamports))
    msg = Message([ix], keypair.pubkey())
    tx = Transaction([keypair], msg, blockhash)

    raw_b64 = base64.b64encode(bytes(tx)).decode()
    sig = await _rpc_call(
        rpc_url,
        "sendTransaction",
        [raw_b64, {"encoding": "base64", "preflightCommitment": "confirmed"}],
    )

    # Poll for confirmation
    for _ in range(60):
        await asyncio.sleep(1)
        statuses = await _rpc_call(rpc_url, "getSignatureStatuses", [[sig]])
        status = statuses["value"][0]
        if status and status.get("confirmationStatus") in ("confirmed", "finalized"):
            if status.get("err"):
                raise RuntimeError("Transaction failed on chain")
            return sig

    raise RuntimeError("Transaction confirmation timed out")


# ─── Airdrop with retry ───────────────────────────────────────────────────────


async def _airdrop_with_retry(
    rpc_url: str,
    pubkey_str: str,
    retries: int = 3,
) -> None:
    """
    Attempt a 2 SOL airdrop with exponential back-off (1 s, 2 s, 4 s).

    Raises :class:`MppFaucetError` after *retries* consecutive failures.
    """
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            await _request_airdrop(rpc_url, pubkey_str, 2 * LAMPORTS_PER_SOL)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)  # 1s, 2s, 4s
    raise MppFaucetError(pubkey_str, last_exc)


# ─── Header parsing ───────────────────────────────────────────────────────────


def _parse_header_params(header: str) -> dict[str, str]:
    """
    Parse a structured HTTP header value of the form::

        solana; key1="val1"; key2="val2"

    Returns a dict of lowercase key → unquoted value pairs (the scheme token
    itself is skipped).
    """
    params: dict[str, str] = {}
    parts = [p.strip() for p in header.split(";")]
    for part in parts[1:]:
        eq_idx = part.find("=")
        if eq_idx > 0:
            key = part[:eq_idx].strip().lower()
            val = part[eq_idx + 1 :].strip().strip('"')
            params[key] = val
    return params


# ─── create_test_client ───────────────────────────────────────────────────────


async def create_test_client(config: TestClientConfig | None = None) -> TestClient:
    """
    Create a Solana MPP test client.

    Automatically creates a Solana wallet, funds it (via airdrop on
    devnet/testnet), and returns a :class:`TestClient` whose
    :meth:`~TestClient.fetch` method handles HTTP 402 MPP payments.

    Parameters
    ----------
    config:
        Optional :class:`TestClientConfig`.  Defaults to devnet with an
        auto-generated keypair.

    Returns
    -------
    TestClient

    Raises
    ------
    MppNetworkError:
        When ``"mainnet"`` is specified without a *secret_key*.
    MppFaucetError:
        When the devnet/testnet airdrop fails after all retries.

    Examples
    --------
    ::

        # devnet (default) — zero config
        client = await create_test_client()

        # testnet
        client = await create_test_client(TestClientConfig(network="testnet"))

        # mainnet — must provide pre-funded wallet
        client = await create_test_client(
            TestClientConfig(network="mainnet", secret_key=my_64_byte_secret)
        )

        res = await client.fetch("http://localhost:3001/api/data")
    """
    from solders.keypair import Keypair  # noqa: PLC0415

    cfg = config or TestClientConfig()
    emit: Callable[[PaymentStep], None] = cfg.on_step or (lambda _: None)
    network: SolanaNetwork = cfg.network
    rpc_url: str = cfg.rpc_url or NETWORK_RPC[network]

    # Mainnet requires a pre-funded wallet
    if network == "mainnet" and cfg.secret_key is None:
        raise MppNetworkError(
            "mainnet",
            "create_test_client: mainnet requires a pre-funded secret_key. "
            "Airdrop is not available on mainnet. "
            "Pass your keypair's secret_key in the config.",
        )

    # Build or restore keypair
    if cfg.secret_key is not None:
        keypair = Keypair.from_bytes(cfg.secret_key)
    else:
        keypair = Keypair()

    address: str = str(keypair.pubkey())

    emit(
        PaymentStep(
            type="wallet-created",
            message=f"Wallet {address}",
            data={"address": address, "network": network},
        )
    )

    # Fund via airdrop on devnet/testnet; skip on mainnet
    if network in _AIRDROP_NETWORKS:
        await _airdrop_with_retry(rpc_url, address)
        emit(
            PaymentStep(
                type="funded",
                message=f"Wallet funded via {network} airdrop (2 SOL)",
                data={"network": network, "amount": 2},
            )
        )
    else:
        emit(
            PaymentStep(
                type="funded",
                message="Using pre-funded mainnet wallet",
                data={"network": network},
            )
        )

    return TestClient(
        address=address,
        network=network,
        keypair=keypair,
        rpc_url=rpc_url,
        on_step=emit,
        timeout=cfg.timeout,
    )


# ─── mpp_fetch (shared lazy client) ──────────────────────────────────────────

_shared_client: TestClient | None = None


async def mpp_fetch(url: str, **kwargs: Any) -> httpx.Response:
    """
    Drop-in replacement for an async HTTP fetch with automatic Solana MPP payment.

    Uses a shared :class:`TestClient` lazily created on first call (devnet by
    default).  Call :func:`mpp_fetch_reset` to discard the shared instance and
    generate a new wallet on the next call.

    Parameters
    ----------
    url:
        The URL to fetch.
    **kwargs:
        Forwarded to :meth:`TestClient.fetch`.

    Returns
    -------
    httpx.Response

    Raises
    ------
    MppFaucetError:
        When the devnet airdrop fails after retries.
    MppTimeoutError:
        When the full flow exceeds the timeout.

    Examples
    --------
    ::

        from mpp_test_sdk import mpp_fetch

        res = await mpp_fetch("http://localhost:3001/api/data")
        data = res.json()
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = await create_test_client()
    return await _shared_client.fetch(url, **kwargs)


def mpp_fetch_reset() -> None:
    """
    Discard the shared client.

    The next call to :func:`mpp_fetch` will create a fresh wallet (and trigger
    a new airdrop on devnet/testnet).
    """
    global _shared_client
    _shared_client = None
