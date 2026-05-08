"""
Tests for mpp_test_sdk._server — MppServer, create_test_server, _verify_payment.

All Solana RPC calls are intercepted.
Flask and FastAPI adapters are exercised with lightweight in-process test clients.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mpp_test_sdk import TestServerConfig, create_test_server
from mpp_test_sdk._server import MppServer, _verify_payment

from .conftest import (
    CLIENT_ADDRESS,
    FAKE_SIG,
    SERVER_ADDRESS,
    build_fake_tx,
)

# ─── Constants ────────────────────────────────────────────────────────────────

RECEIPT_EXACT = (
    f'solana; signature="{FAKE_SIG}"; network="devnet"; amount="0.001"'
)
RECEIPT_HIGH = (
    f'solana; signature="{FAKE_SIG}"; network="devnet"; amount="0.005"'
)
REQUIRED_SOL = 0.001
RPC_URL = "https://api.devnet.solana.com"


def make_server(recipient: str = SERVER_ADDRESS) -> MppServer:
    """Return an MppServer with a predictable recipient and a fake RPC URL."""
    return MppServer(
        recipient_address=recipient,
        network="devnet",
        rpc_url=RPC_URL,
    )


# ─── 1. No Payment-Receipt → _verify_payment returns (False, …) ──────────────


@pytest.mark.asyncio
async def test_no_receipt_returns_false_missing_signature() -> None:
    """Empty receipt → (False, 'missing signature') without any RPC call."""
    ok, msg = await _verify_payment(RPC_URL, "", SERVER_ADDRESS, REQUIRED_SOL)
    assert not ok
    assert "signature" in msg.lower()


# ─── 2. 402 body has correct JSON structure ───────────────────────────────────


def test_payment_required_body_structure() -> None:
    """_payment_required_body returns the correct nested dict."""
    server = make_server()
    body = server._payment_required_body("0.001")
    assert body["error"] == "Payment Required"
    assert body["payment"]["amount"] == "0.001"
    assert body["payment"]["currency"] == "SOL"
    assert body["payment"]["recipient"] == SERVER_ADDRESS
    assert body["payment"]["network"] == "devnet"


# ─── 3. Missing signature → (False, msg) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_missing_signature() -> None:
    """Receipt without signature field → (False, message mentioning 'signature')."""
    receipt = 'solana; network="devnet"; amount="0.001"'
    ok, msg = await _verify_payment(RPC_URL, receipt, SERVER_ADDRESS, REQUIRED_SOL)
    assert not ok
    assert "signature" in msg.lower()


# ─── 4. Claimed amount < required → (False, msg) ─────────────────────────────


@pytest.mark.asyncio
async def test_verify_insufficient_claimed_amount(mocker: Any) -> None:
    """Claimed 0.0001 SOL but 0.001 required → Insufficient."""
    # We don't even need RPC for this — amount check happens first
    receipt = f'solana; signature="{FAKE_SIG}"; network="devnet"; amount="0.0001"'
    ok, msg = await _verify_payment(RPC_URL, receipt, SERVER_ADDRESS, 0.001)
    assert not ok
    assert "insufficient" in msg.lower() or "0.001" in msg


# ─── 5. Transaction not found → (False, msg) ─────────────────────────────────


@pytest.mark.asyncio
async def test_verify_tx_not_found(mocker: Any) -> None:
    """getTransaction returning None → (False, 'not found')."""
    mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        new=AsyncMock(return_value=None),
    )
    ok, msg = await _verify_payment(RPC_URL, RECEIPT_EXACT, SERVER_ADDRESS, REQUIRED_SOL)
    assert not ok
    assert "not found" in msg.lower()


# ─── 6. Transaction has error → (False, msg) ─────────────────────────────────


@pytest.mark.asyncio
async def test_verify_tx_has_error(mocker: Any) -> None:
    """Transaction with meta.err → (False, 'failed on chain')."""
    mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        new=AsyncMock(return_value=build_fake_tx(SERVER_ADDRESS, failed=True)),
    )
    ok, msg = await _verify_payment(RPC_URL, RECEIPT_EXACT, SERVER_ADDRESS, REQUIRED_SOL)
    assert not ok
    assert "failed on chain" in msg.lower()


# ─── 7. Recipient not in transaction → (False, msg) ──────────────────────────


@pytest.mark.asyncio
async def test_verify_recipient_not_in_tx(mocker: Any) -> None:
    """Transaction paying a different address → (False, 'not found in transaction')."""
    other = "SomeOtherRecipient11111111111111111111111"
    mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        new=AsyncMock(return_value=build_fake_tx(other)),
    )
    ok, msg = await _verify_payment(RPC_URL, RECEIPT_EXACT, SERVER_ADDRESS, REQUIRED_SOL)
    assert not ok
    assert "not found in transaction" in msg.lower()


# ─── 8. Received < required → (False, msg) ───────────────────────────────────


@pytest.mark.asyncio
async def test_verify_received_too_small(mocker: Any) -> None:
    """Only 100 lamports received but 1_000_000 required → too small."""
    mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        new=AsyncMock(
            return_value=build_fake_tx(SERVER_ADDRESS, received_lamports=100)
        ),
    )
    ok, msg = await _verify_payment(RPC_URL, RECEIPT_EXACT, SERVER_ADDRESS, REQUIRED_SOL)
    assert not ok
    assert "too small" in msg.lower()


# ─── 9. Valid payment → (True, "") ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_exact_payment_ok(mocker: Any) -> None:
    """Exactly required amount → (True, '')."""
    mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        new=AsyncMock(
            return_value=build_fake_tx(SERVER_ADDRESS, received_lamports=1_000_000)
        ),
    )
    ok, msg = await _verify_payment(RPC_URL, RECEIPT_EXACT, SERVER_ADDRESS, REQUIRED_SOL)
    assert ok
    assert msg == ""


@pytest.mark.asyncio
async def test_verify_overpayment_ok(mocker: Any) -> None:
    """More than required → (True, '')."""
    mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        new=AsyncMock(
            return_value=build_fake_tx(SERVER_ADDRESS, received_lamports=5_000_000)
        ),
    )
    ok, msg = await _verify_payment(RPC_URL, RECEIPT_HIGH, SERVER_ADDRESS, REQUIRED_SOL)
    assert ok
    assert msg == ""


# ─── 10. Flask adapter returns 402 with correct headers ───────────────────────


def test_flask_charge_no_receipt_returns_402() -> None:
    """Flask decorator: no Payment-Receipt → 402 with correct Payment-Request header."""
    flask = pytest.importorskip("flask")

    app = flask.Flask(__name__)
    server = make_server()

    @app.get("/api/data")
    @server.flask_charge("0.001")
    def data() -> flask.Response:
        return flask.jsonify({"data": "premium"})

    with app.test_client() as client:
        res = client.get("/api/data")

    assert res.status_code == 402
    assert "Payment-Request" in res.headers
    header = res.headers["Payment-Request"]
    assert "solana" in header
    assert 'amount="0.001"' in header
    assert f'recipient="{SERVER_ADDRESS}"' in header
    assert 'network="devnet"' in header

    body = res.get_json()
    assert body["error"] == "Payment Required"
    assert body["payment"]["currency"] == "SOL"
    assert body["payment"]["amount"] == "0.001"


# ─── 11. FastAPI adapter raises HTTPException 402 with headers ────────────────


@pytest.mark.asyncio
async def test_fastapi_charge_no_receipt_raises_402() -> None:
    """FastAPI dependency: no Payment-Receipt → 402 with Payment-Request header."""
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")

    from fastapi import Depends, FastAPI  # noqa: PLC0415
    from starlette.testclient import TestClient  # noqa: PLC0415

    app = FastAPI()
    server = make_server()

    @app.get("/api/data")
    async def data(dep: None = Depends(server.charge("0.001"))) -> dict:
        return {"data": "premium"}

    with TestClient(app, raise_server_exceptions=False) as tc:
        res = tc.get("/api/data")

    assert res.status_code == 402
    # Header is set on the HTTPException by our dependency
    assert "payment-request" in res.headers
    header = res.headers["payment-request"]
    assert "solana" in header
    assert 'amount="0.001"' in header


@pytest.mark.asyncio
async def test_fastapi_charge_valid_payment_200(mocker: Any) -> None:
    """FastAPI dependency: valid receipt → 200."""
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")

    from fastapi import Depends, FastAPI  # noqa: PLC0415
    from starlette.testclient import TestClient  # noqa: PLC0415

    mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        new=AsyncMock(
            return_value=build_fake_tx(SERVER_ADDRESS, received_lamports=1_000_000)
        ),
    )

    app = FastAPI()
    server = make_server()

    @app.get("/api/data")
    async def data(dep: None = Depends(server.charge("0.001"))) -> dict:
        return {"data": "premium"}

    with TestClient(app) as tc:
        res = tc.get("/api/data", headers={"payment-receipt": RECEIPT_EXACT})

    assert res.status_code == 200
    assert res.json()["data"] == "premium"


# ─── 12. Custom recipient_address works ───────────────────────────────────────


def test_custom_recipient_address_in_header() -> None:
    """MppServer respects a custom recipient_address in the Payment-Request header."""
    custom = "CustomRecip111111111111111111111111111111"
    server = MppServer(recipient_address=custom, network="devnet", rpc_url=RPC_URL)
    assert server.recipient_address == custom
    header = server._payment_request_header("0.001")
    assert f'recipient="{custom}"' in header


def test_create_test_server_custom_recipient(mocker: Any) -> None:
    """create_test_server propagates recipient_address override correctly."""
    custom = "CustomRecip111111111111111111111111111111"

    class FakeKP:
        def __new__(cls) -> "FakeKP":  # type: ignore[misc]
            return object.__new__(cls)

        def pubkey(self):  # noqa: ANN201
            class _P:
                def __str__(self) -> str:
                    return SERVER_ADDRESS
            return _P()

    mocker.patch("solders.keypair.Keypair", FakeKP)
    server = create_test_server(TestServerConfig(recipient_address=custom))
    assert server.recipient_address == custom
