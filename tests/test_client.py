"""
Tests for mpp_test_sdk._client — create_test_client, TestClient.fetch, mpp_fetch.

Strategy:
- Patch ``mpp_test_sdk._client._airdrop_with_retry`` so no real RPC calls are made.
- Patch the ``Keypair`` class imported inside ``create_test_client`` via
  ``mpp_test_sdk._client`` module attribute substitution.
- Use ``respx`` to mock outgoing HTTP (httpx) requests.
- Patch ``mpp_test_sdk._client._send_sol`` to avoid real blockchain I/O.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import mpp_test_sdk._client as _client_mod
from mpp_test_sdk._client import (
    PaymentStep,
    TestClient,
    TestClientConfig,
    _airdrop_with_retry,
    create_test_client,
    mpp_fetch,
    mpp_fetch_reset,
)
from mpp_test_sdk.errors import (
    MppFaucetError,
    MppNetworkError,
    MppPaymentError,
    MppTimeoutError,
)

from .conftest import FAKE_ADDRESS, FAKE_SIG, FakeKeypair

# ─── Payment-Request header used across tests ─────────────────────────────────

PAYMENT_REQUEST_HEADER = (
    f'solana; amount="0.001"; recipient="{FAKE_ADDRESS}"; network="devnet"'
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_client(
    network: str = "devnet",
    timeout: float = 30.0,
    on_step=None,
) -> TestClient:
    """Build a TestClient backed by a FakeKeypair — no solders needed."""
    kp = FakeKeypair(FAKE_ADDRESS)
    return TestClient(
        address=FAKE_ADDRESS,
        network=network,  # type: ignore[arg-type]
        keypair=kp,
        rpc_url=_client_mod.NETWORK_RPC[network],
        on_step=on_step or (lambda _: None),
        timeout=timeout,
    )


# ─── 1. create_test_client devnet — keypair generated, returns TestClient ──────


@pytest.mark.asyncio
async def test_create_client_devnet_defaults() -> None:
    """create_test_client devnet: returns TestClient with correct attributes."""
    airdrop_mock = AsyncMock()
    fake_kp = FakeKeypair(FAKE_ADDRESS)

    class FakeKeypairClass:
        def __new__(cls) -> FakeKeypair:  # type: ignore[misc]
            return fake_kp

        @staticmethod
        def from_bytes(data: bytes) -> FakeKeypair:
            return FakeKeypair(FAKE_ADDRESS)

    # Patch via solders.keypair import inside create_test_client
    with (
        patch("mpp_test_sdk._client._airdrop_with_retry", airdrop_mock),
        patch("solders.keypair.Keypair", FakeKeypairClass),
    ):
        client = await create_test_client()

    assert client.address == FAKE_ADDRESS
    assert client.network == "devnet"
    assert client.method == "solana"
    assert callable(client.fetch)
    airdrop_mock.assert_called_once()


# ─── 2. create_test_client with secret_key — uses provided key ────────────────


@pytest.mark.asyncio
async def test_create_client_with_secret_key() -> None:
    """create_test_client with secret_key still airdrops on devnet."""
    airdrop_mock = AsyncMock()
    fake_kp = FakeKeypair(FAKE_ADDRESS)

    class FakeKP:
        @staticmethod
        def from_bytes(data: bytes) -> FakeKeypair:
            return fake_kp

        def __new__(cls) -> FakeKeypair:  # type: ignore[misc]
            return fake_kp

    with (
        patch("mpp_test_sdk._client._airdrop_with_retry", airdrop_mock),
        patch("solders.keypair.Keypair", FakeKP),
    ):
        client = await create_test_client(TestClientConfig(secret_key=bytes(64)))

    assert client.address == FAKE_ADDRESS
    airdrop_mock.assert_called_once()


# ─── 3. create_test_client mainnet no secret_key → MppNetworkError ────────────


@pytest.mark.asyncio
async def test_create_client_mainnet_no_key_raises() -> None:
    """mainnet without secret_key → MppNetworkError."""
    with pytest.raises(MppNetworkError) as exc_info:
        await create_test_client(TestClientConfig(network="mainnet"))

    assert exc_info.value.network == "mainnet"


@pytest.mark.asyncio
async def test_create_client_mainnet_error_mentions_secret_key() -> None:
    """MppNetworkError message mentions mainnet and secret_key."""
    with pytest.raises(MppNetworkError) as exc_info:
        await create_test_client(TestClientConfig(network="mainnet"))

    msg = str(exc_info.value)
    assert "mainnet" in msg
    assert "secret_key" in msg


# ─── 4. create_test_client testnet — airdrop called ──────────────────────────


@pytest.mark.asyncio
async def test_create_client_testnet_airdrops() -> None:
    """create_test_client testnet: airdrop is invoked."""
    airdrop_mock = AsyncMock()
    fake_kp = FakeKeypair(FAKE_ADDRESS)

    class FakeKP:
        def __new__(cls) -> FakeKeypair:  # type: ignore[misc]
            return fake_kp

        @staticmethod
        def from_bytes(data: bytes) -> FakeKeypair:
            return fake_kp

    with (
        patch("mpp_test_sdk._client._airdrop_with_retry", airdrop_mock),
        patch("solders.keypair.Keypair", FakeKP),
    ):
        client = await create_test_client(TestClientConfig(network="testnet"))

    assert client.network == "testnet"
    airdrop_mock.assert_called_once()


# ─── 5. on_step callback receives wallet-created, funded events ───────────────


@pytest.mark.asyncio
async def test_on_step_order() -> None:
    """on_step: wallet-created then funded emitted in order."""
    steps: list[str] = []
    fake_kp = FakeKeypair(FAKE_ADDRESS)

    class FakeKP:
        def __new__(cls) -> FakeKeypair:  # type: ignore[misc]
            return fake_kp

        @staticmethod
        def from_bytes(data: bytes) -> FakeKeypair:
            return fake_kp

    with (
        patch("mpp_test_sdk._client._airdrop_with_retry", AsyncMock()),
        patch("solders.keypair.Keypair", FakeKP),
    ):
        await create_test_client(
            TestClientConfig(on_step=lambda s: steps.append(s.type))
        )

    assert steps[0] == "wallet-created"
    assert steps[1] == "funded"


@pytest.mark.asyncio
async def test_on_step_wallet_created_data() -> None:
    """wallet-created event data contains address and network."""
    events: list[PaymentStep] = []
    fake_kp = FakeKeypair(FAKE_ADDRESS)

    class FakeKP:
        def __new__(cls) -> FakeKeypair:  # type: ignore[misc]
            return fake_kp

        @staticmethod
        def from_bytes(data: bytes) -> FakeKeypair:
            return fake_kp

    with (
        patch("mpp_test_sdk._client._airdrop_with_retry", AsyncMock()),
        patch("solders.keypair.Keypair", FakeKP),
    ):
        await create_test_client(
            TestClientConfig(on_step=lambda s: events.append(s))
        )

    ev = next(e for e in events if e.type == "wallet-created")
    assert ev.data["address"] == FAKE_ADDRESS
    assert ev.data["network"] == "devnet"


# ─── 6. fetch free endpoint (200) → response + request+success events ─────────


@pytest.mark.asyncio
async def test_fetch_free_200_no_payment() -> None:
    """200 response returned without payment; request+success events emitted."""
    steps: list[str] = []
    client = make_client(on_step=lambda s: steps.append(s.type))

    with respx.mock:
        respx.get("http://localhost:3001/api/free").mock(
            return_value=httpx.Response(200, json={"data": "free"})
        )
        res = await client.fetch("http://localhost:3001/api/free")

    assert res.status_code == 200
    assert "request" in steps
    assert "success" in steps


# ─── 7. fetch paid endpoint (402 flow) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_402_full_flow() -> None:
    """402 → send SOL → retry with Payment-Receipt → 200 returned."""
    send_sol_mock = AsyncMock(return_value=FAKE_SIG)
    client = make_client()

    with respx.mock, patch("mpp_test_sdk._client._send_sol", send_sol_mock):
        route = respx.get("http://localhost:3001/api/paid")
        route.side_effect = [
            httpx.Response(
                402,
                headers={"payment-request": PAYMENT_REQUEST_HEADER},
                json={},
            ),
            httpx.Response(200, json={"data": "paid"}),
        ]
        res = await client.fetch("http://localhost:3001/api/paid")

    assert res.status_code == 200
    send_sol_mock.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_402_sends_payment_receipt_header() -> None:
    """Retry request includes a properly formatted Payment-Receipt header."""
    send_sol_mock = AsyncMock(return_value=FAKE_SIG)
    client = make_client()
    captured: dict[str, str] = {}

    def _side_effect(req: httpx.Request) -> httpx.Response:
        if "payment-receipt" not in req.headers:
            return httpx.Response(
                402,
                headers={"payment-request": PAYMENT_REQUEST_HEADER},
                json={},
            )
        captured.update(dict(req.headers))
        return httpx.Response(200, json={})

    with respx.mock, patch("mpp_test_sdk._client._send_sol", send_sol_mock):
        respx.get("http://localhost:3001/api/paid").mock(side_effect=_side_effect)
        await client.fetch("http://localhost:3001/api/paid")

    assert "payment-receipt" in captured
    receipt = captured["payment-receipt"]
    assert "solana" in receipt
    assert f'signature="{FAKE_SIG}"' in receipt
    assert 'network="devnet"' in receipt


# ─── 8. fetch non-402 error (404) → MppPaymentError ──────────────────────────


@pytest.mark.asyncio
async def test_fetch_non_402_error_raises() -> None:
    """Non-402 HTTP error raises MppPaymentError with correct url + status."""
    client = make_client()

    with respx.mock:
        respx.get("http://localhost:3001/api/private").mock(
            return_value=httpx.Response(404, json={"error": "Not Found"})
        )
        with pytest.raises(MppPaymentError) as exc_info:
            await client.fetch("http://localhost:3001/api/private")

    assert exc_info.value.status == 404
    assert exc_info.value.url == "http://localhost:3001/api/private"


# ─── 9. fetch with missing Payment-Request header on 402 ─────────────────────


@pytest.mark.asyncio
async def test_fetch_402_missing_payment_request_header() -> None:
    """402 without Payment-Request header raises MppPaymentError(status=402)."""
    client = make_client()

    with respx.mock:
        respx.get("http://localhost:3001/api/paid").mock(
            return_value=httpx.Response(402, json={})
        )
        with pytest.raises(MppPaymentError) as exc_info:
            await client.fetch("http://localhost:3001/api/paid")

    assert exc_info.value.status == 402


# ─── 10. fetch with missing recipient in header ───────────────────────────────


@pytest.mark.asyncio
async def test_fetch_402_missing_recipient_raises() -> None:
    """Payment-Request without recipient field raises MppPaymentError."""
    client = make_client()
    bad_header = 'solana; amount="0.001"; network="devnet"'

    with respx.mock:
        respx.get("http://localhost:3001/api/paid").mock(
            return_value=httpx.Response(
                402, headers={"payment-request": bad_header}, json={}
            )
        )
        with pytest.raises(MppPaymentError) as exc_info:
            await client.fetch("http://localhost:3001/api/paid")

    assert exc_info.value.status == 402


# ─── 11. fetch with invalid amount in header ──────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_402_invalid_amount_raises() -> None:
    """Non-numeric amount in Payment-Request raises MppPaymentError."""
    client = make_client()
    bad_header = (
        f'solana; amount="not-a-number"; recipient="{FAKE_ADDRESS}"; network="devnet"'
    )

    with respx.mock:
        respx.get("http://localhost:3001/api/paid").mock(
            return_value=httpx.Response(
                402, headers={"payment-request": bad_header}, json={}
            )
        )
        with pytest.raises(MppPaymentError) as exc_info:
            await client.fetch("http://localhost:3001/api/paid")

    assert exc_info.value.status == 402


# ─── 12. airdrop retry — first 2 fail, 3rd succeeds ─────────────────────────


@pytest.mark.asyncio
async def test_airdrop_retry_succeeds_on_third() -> None:
    """_airdrop_with_retry: first 2 attempts raise, 3rd succeeds."""
    call_count = 0

    async def flaky(rpc_url: str, method: str, params: list) -> Any:
        nonlocal call_count
        if method == "requestAirdrop":
            call_count += 1
            if call_count < 3:
                raise RuntimeError("rate limited")
            return "ok_sig"
        # getSignatureStatuses
        return {"value": [{"confirmationStatus": "confirmed", "err": None}]}

    # Patch sleep to be instant
    with patch("asyncio.sleep", AsyncMock()):
        with patch("mpp_test_sdk._client._rpc_call", side_effect=flaky):
            await _airdrop_with_retry("https://api.devnet.solana.com", FAKE_ADDRESS)

    assert call_count == 3


# ─── 13. airdrop all retries fail → MppFaucetError ────────────────────────────


@pytest.mark.asyncio
async def test_airdrop_all_retries_fail() -> None:
    """_airdrop_with_retry: 3 consecutive failures → MppFaucetError."""
    async def always_fail(rpc_url: str, method: str, params: list) -> Any:
        if method == "requestAirdrop":
            raise RuntimeError("rate limit")
        return {"value": [{"confirmationStatus": "confirmed", "err": None}]}

    with patch("asyncio.sleep", AsyncMock()):
        with patch("mpp_test_sdk._client._rpc_call", side_effect=always_fail):
            with pytest.raises(MppFaucetError) as exc_info:
                await _airdrop_with_retry(
                    "https://api.devnet.solana.com", FAKE_ADDRESS
                )

    assert exc_info.value.address == FAKE_ADDRESS


# ─── 14. timeout → MppTimeoutError ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_timeout_raises_mpp_timeout_error() -> None:
    """When fetch exceeds timeout, MppTimeoutError is raised."""
    client = make_client(timeout=0.001)

    async def _hang(*args: Any, **kwargs: Any) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200)  # pragma: no cover

    with patch.object(client, "_fetch_inner", side_effect=_hang):
        with pytest.raises(MppTimeoutError) as exc_info:
            await client.fetch("http://localhost:3001/slow")

    assert exc_info.value.url == "http://localhost:3001/slow"
    assert exc_info.value.timeout_ms == 1  # 0.001 s → 1 ms


# ─── 15. mpp_fetch creates shared client, reuses it ──────────────────────────


@pytest.mark.asyncio
async def test_mpp_fetch_reuses_shared_client() -> None:
    """mpp_fetch: shared client is created once and reused on 2nd call."""
    mpp_fetch_reset()  # ensure clean slate

    create_calls: list[int] = []
    original_create = create_test_client

    async def counting_create(config=None):  # noqa: ANN001
        create_calls.append(1)
        return make_client()

    with respx.mock:
        respx.get("http://localhost:3001/a").mock(
            return_value=httpx.Response(200, json={})
        )
        respx.get("http://localhost:3001/b").mock(
            return_value=httpx.Response(200, json={})
        )

        with patch("mpp_test_sdk._client.create_test_client", counting_create):
            # reset the module-level shared client
            _client_mod._shared_client = None  # type: ignore[assignment]
            await _client_mod.mpp_fetch("http://localhost:3001/a")
            await _client_mod.mpp_fetch("http://localhost:3001/b")

    assert len(create_calls) == 1


# ─── 16. mpp_fetch_reset clears shared client ────────────────────────────────


@pytest.mark.asyncio
async def test_mpp_fetch_reset_clears_shared_client() -> None:
    """mpp_fetch_reset sets shared client to None."""
    mpp_fetch_reset()
    _client_mod._shared_client = make_client()  # type: ignore[assignment]
    assert _client_mod._shared_client is not None

    mpp_fetch_reset()
    assert _client_mod._shared_client is None


# ─── Bonus: events emitted during 402 flow ───────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_402_emits_payment_retry_success_events() -> None:
    """402 flow emits payment, retry, and success step types."""
    steps: list[str] = []
    send_sol_mock = AsyncMock(return_value=FAKE_SIG)
    client = make_client(on_step=lambda s: steps.append(s.type))

    with respx.mock, patch("mpp_test_sdk._client._send_sol", send_sol_mock):
        route = respx.get("http://localhost:3001/api/paid")
        route.side_effect = [
            httpx.Response(
                402,
                headers={"payment-request": PAYMENT_REQUEST_HEADER},
                json={},
            ),
            httpx.Response(200, json={}),
        ]
        await client.fetch("http://localhost:3001/api/paid")

    assert "request" in steps
    assert "payment" in steps
    assert "retry" in steps
    assert "success" in steps


@pytest.mark.asyncio
async def test_fetch_402_payment_event_data() -> None:
    """payment event data carries correct amount and recipient."""
    events: list[PaymentStep] = []
    send_sol_mock = AsyncMock(return_value=FAKE_SIG)
    client = make_client(on_step=lambda s: events.append(s))

    with respx.mock, patch("mpp_test_sdk._client._send_sol", send_sol_mock):
        route = respx.get("http://localhost:3001/api/paid")
        route.side_effect = [
            httpx.Response(
                402,
                headers={"payment-request": PAYMENT_REQUEST_HEADER},
                json={},
            ),
            httpx.Response(200, json={}),
        ]
        await client.fetch("http://localhost:3001/api/paid")

    pay_ev = next(
        (e for e in events if e.type == "payment" and "amount" in e.data), None
    )
    assert pay_ev is not None
    assert pay_ev.data["amount"] == 0.001
    assert pay_ev.data["recipient"] == FAKE_ADDRESS
