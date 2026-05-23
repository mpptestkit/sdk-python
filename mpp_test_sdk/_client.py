"""
MPP test client -automatic HTTP 402 / Solana payment handling.

Usage::

    from mpp_test_sdk import create_test_client, mpp_fetch, mpp_fetch_reset

    # devnet (default) -zero config
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

from ._chain import BaseNetwork, ChainType
from ._rpc import (
    LAMPORTS_PER_SOL,
    NETWORK_RPC,
    RpcClient,
    SolanaNetwork,
    parse_header_params,
)
from .errors import MppFaucetError, MppNetworkError, MppPaymentError, MppTimeoutError

_AIRDROP_NETWORKS: frozenset[str] = frozenset({"devnet", "testnet"})

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
    chain:
        Blockchain for payments. ``"solana"`` (default) or ``"base"``.
    network:
        Solana network. ``"devnet"`` (default) and ``"testnet"`` support free
        airdrops.  ``"mainnet"`` requires a pre-funded *secret_key*.
    secret_key:
        32- or 64-byte keypair seed/secret.  On devnet/testnet this is optional
        (airdrop is used).  On mainnet it is **required**.
    base_network:
        Base network when *chain* is ``"base"``. Default: ``"sepolia"``.
    private_key:
        Hex-encoded secp256k1 key for Base payments.
    on_step:
        Callback invoked with each :class:`PaymentStep` lifecycle event.
    timeout:
        Full-flow timeout in seconds (wallet creation + payment + retry).
        Default: 30.0.
    rpc_url:
        Override the JSON-RPC endpoint.  Takes precedence over *network* /
        *base_network*.
    """

    chain: ChainType = "solana"
    network: SolanaNetwork = "devnet"
    secret_key: bytes | None = None
    base_network: BaseNetwork | None = None
    private_key: str | None = None
    on_step: Callable[[PaymentStep], None] | None = None
    timeout: float = 30.0
    rpc_url: str | None = None


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
        keypair: Any,
        rpc: RpcClient,
        http: httpx.AsyncClient,
        on_step: Callable[[PaymentStep], None],
        timeout: float,
    ) -> None:
        self.address: str = address
        self.network: SolanaNetwork = network
        self.method: Literal["solana"] = "solana"
        self._keypair = keypair
        self._rpc = rpc
        self._http = http
        self._on_step = on_step
        self._timeout = timeout

    async def aclose(self) -> None:
        """Close underlying HTTP and RPC connections."""
        await self._http.aclose()
        await self._rpc.aclose()

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
        emit = self._on_step
        emit(PaymentStep(type="request", message=f"→ {url}", data={"url": url}))

        http_method: str = kwargs.pop("method", "GET").upper()
        headers: dict[str, str] = dict(kwargs.pop("headers", {}))

        res = await self._http.request(http_method, url, headers=headers, **kwargs)

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

        payment_request_header = res.headers.get("payment-request", "")
        if not payment_request_header:
            raise MppPaymentError(
                url,
                402,
                ValueError("Server returned 402 without Payment-Request header"),
            )

        params = parse_header_params(payment_request_header)

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

        signature = await _send_sol(self._rpc, self._keypair, recipient_str, lamports)

        emit(
            PaymentStep(
                type="payment",
                message=f"Confirmed: {signature[:16]}...",
                data={"signature": signature, "amount": amount_sol},
            )
        )

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

        retry_res = await self._http.request(
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


async def _request_airdrop(rpc: RpcClient, pubkey_str: str, lamports: int) -> None:
    """Request a faucet airdrop and wait until confirmed."""
    sig = await rpc.call("requestAirdrop", [pubkey_str, lamports])
    await rpc.wait_for_signature(sig)


async def _send_sol(
    rpc: RpcClient,
    keypair: Any,
    recipient_str: str,
    lamports: int,
) -> str:
    """Build, submit, and confirm a SOL transfer. Returns the transaction signature."""
    from solders.hash import Hash  # noqa: PLC0415
    from solders.message import Message  # noqa: PLC0415
    from solders.pubkey import Pubkey  # noqa: PLC0415
    from solders.system_program import TransferParams, transfer  # noqa: PLC0415
    from solders.transaction import Transaction  # noqa: PLC0415

    recipient = Pubkey.from_string(recipient_str)
    blockhash = Hash.from_string(await rpc.get_latest_blockhash())

    ix = transfer(
        TransferParams(
            from_pubkey=keypair.pubkey(),
            to_pubkey=recipient,
            lamports=lamports,
        )
    )
    msg = Message([ix], keypair.pubkey())
    tx = Transaction([keypair], msg, blockhash)

    raw_b64 = base64.b64encode(bytes(tx)).decode()
    sig = await rpc.call(
        "sendTransaction",
        [raw_b64, {"encoding": "base64", "preflightCommitment": "confirmed"}],
    )
    await rpc.wait_for_signature(sig)
    return sig


async def _airdrop_with_retry(
    rpc: RpcClient,
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
            await _request_airdrop(rpc, pubkey_str, 2 * LAMPORTS_PER_SOL)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)
    raise MppFaucetError(pubkey_str, last_exc)


async def create_test_client(
    config: TestClientConfig | None = None,
) -> TestClient | "BaseTestClient":
    """
    Create an MPP test client for Solana or Base.

    Automatically creates a wallet and returns a client whose
    :meth:`~TestClient.fetch` method handles HTTP 402 MPP payments.
    """
    cfg = config or TestClientConfig()
    if cfg.chain == "base":
        from ._base_client import create_base_test_client  # noqa: PLC0415

        return await create_base_test_client(cfg)

    from solders.keypair import Keypair  # noqa: PLC0415
    emit: Callable[[PaymentStep], None] = cfg.on_step or (lambda _: None)
    network: SolanaNetwork = cfg.network
    rpc_url: str = cfg.rpc_url or NETWORK_RPC[network]

    if network == "mainnet" and cfg.secret_key is None:
        raise MppNetworkError(
            "mainnet",
            "create_test_client: mainnet requires a pre-funded secret_key. "
            "Airdrop is not available on mainnet. "
            "Pass your keypair's secret_key in the config.",
        )

    if cfg.secret_key is not None:
        keypair = Keypair.from_bytes(cfg.secret_key)
    else:
        keypair = Keypair()

    address: str = str(keypair.pubkey())
    rpc = RpcClient(rpc_url, timeout=cfg.timeout)

    emit(
        PaymentStep(
            type="wallet-created",
            message=f"Wallet {address}",
            data={"address": address, "network": network},
        )
    )

    if network in _AIRDROP_NETWORKS:
        await _airdrop_with_retry(rpc, address)
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

    http = httpx.AsyncClient(timeout=cfg.timeout)
    return TestClient(
        address=address,
        network=network,
        keypair=keypair,
        rpc=rpc,
        http=http,
        on_step=emit,
        timeout=cfg.timeout,
    )


_shared_client: TestClient | None = None
_shared_client_lock = asyncio.Lock()


async def mpp_fetch(url: str, **kwargs: Any) -> httpx.Response:
    """
    Drop-in replacement for an async HTTP fetch with automatic Solana MPP payment.

    Uses a shared :class:`TestClient` lazily created on first call (devnet by
    default).  Call :func:`mpp_fetch_reset` to discard the shared instance.
    """
    global _shared_client
    async with _shared_client_lock:
        if _shared_client is None:
            _shared_client = await create_test_client()
        client = _shared_client
    return await client.fetch(url, **kwargs)


def mpp_fetch_reset() -> None:
    """Discard the shared client used by :func:`mpp_fetch`."""
    global _shared_client
    _shared_client = None
