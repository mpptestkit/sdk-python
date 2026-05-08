"""
Shared pytest fixtures for the MPP Python SDK test suite.

All network calls (httpx + Solana JSON-RPC) are mocked — no real network
traffic is issued during the test run.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

# ─── Constants ────────────────────────────────────────────────────────────────

FAKE_ADDRESS = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
SERVER_ADDRESS = "ServerPubkey1111111111111111111111111111"
CLIENT_ADDRESS = "ClientPubkey11111111111111111111111111111"
FAKE_SIG = "tx_sig_abc123456789"
FAKE_BLOCKHASH = "AbCdEfGh1234567890abcdefgh"
FAKE_AIRDROP_SIG = "airdrop_sig_123"


# ─── Fake solders objects ─────────────────────────────────────────────────────


class FakePubkey:
    """Minimal stand-in for ``solders.pubkey.Pubkey``."""

    def __init__(self, key: str = FAKE_ADDRESS) -> None:
        self._key = key

    def __str__(self) -> str:
        return self._key

    def __repr__(self) -> str:  # pragma: no cover
        return f"FakePubkey({self._key!r})"


class FakeKeypair:
    """Minimal stand-in for ``solders.keypair.Keypair``."""

    def __init__(self, address: str = FAKE_ADDRESS) -> None:
        self._address = address

    def pubkey(self) -> FakePubkey:
        return FakePubkey(self._address)

    @classmethod
    def from_bytes(cls, data: bytes) -> "FakeKeypair":  # noqa: ARG003
        return cls(FAKE_ADDRESS)

    # Make __new__() callable like Keypair() for auto-gen scenario
    def __new__(cls, address: str = FAKE_ADDRESS) -> "FakeKeypair":
        obj = super().__new__(cls)
        return obj


class FakeServerKeypair:
    """FakeKeypair variant for server tests."""

    def __init__(self) -> None:
        self._address = SERVER_ADDRESS

    def pubkey(self) -> FakePubkey:
        return FakePubkey(SERVER_ADDRESS)

    @classmethod
    def from_bytes(cls, data: bytes) -> "FakeServerKeypair":  # noqa: ARG003
        return cls()


# ─── RPC mock fixture ─────────────────────────────────────────────────────────


@pytest.fixture()
def mock_rpc(mocker: Any) -> Any:
    """
    Patch ``mpp_test_sdk._client._rpc_call`` with a sensible default.

    Default behaviour:

    - ``requestAirdrop``      → ``FAKE_AIRDROP_SIG``
    - ``getSignatureStatuses`` → confirmed status
    - ``getLatestBlockhash``   → ``FAKE_BLOCKHASH``
    - ``sendTransaction``      → ``FAKE_SIG``
    """
    async def _default_rpc(rpc_url: str, method: str, params: list) -> Any:  # noqa: ARG001
        match method:
            case "requestAirdrop":
                return FAKE_AIRDROP_SIG
            case "getSignatureStatuses":
                return {
                    "value": [{"confirmationStatus": "confirmed", "err": None}]
                }
            case "getLatestBlockhash":
                return {"value": {"blockhash": FAKE_BLOCKHASH}}
            case "sendTransaction":
                return FAKE_SIG
            case _:
                return None

    return mocker.patch(
        "mpp_test_sdk._client._rpc_call",
        side_effect=_default_rpc,
    )


@pytest.fixture()
def mock_server_rpc(mocker: Any) -> Any:
    """
    Patch ``mpp_test_sdk._server._rpc_call`` (used by ``_verify_payment``).

    Default: returns a valid transaction for *SERVER_ADDRESS* with 2_000_000
    lamports received — sufficient for a ``"0.001"`` SOL check.
    """
    async def _default_rpc(rpc_url: str, method: str, params: list) -> Any:  # noqa: ARG001
        if method == "getTransaction":
            return build_fake_tx(SERVER_ADDRESS, received_lamports=2_000_000)
        return None

    return mocker.patch(
        "mpp_test_sdk._server._rpc_call",
        side_effect=_default_rpc,
    )


# ─── Transaction builder helper ──────────────────────────────────────────────


def build_fake_tx(
    recipient: str,
    received_lamports: int = 2_000_000,
    failed: bool = False,
    sender: str = CLIENT_ADDRESS,
) -> dict[str, Any]:
    """
    Build a fake Solana ``getTransaction`` JSON response.

    Parameters
    ----------
    recipient:
        Base58 address of the payment recipient.
    received_lamports:
        Lamports credited to the recipient (post − pre).
    failed:
        If ``True``, sets ``meta.err`` to a non-null value.
    sender:
        Base58 address of the sender (index 0 in accountKeys).
    """
    return {
        "meta": {
            "err": {"InstructionError": [0, "InsufficientFunds"]} if failed else None,
            "preBalances": [1_000_000_000, 0],
            "postBalances": [997_000_000, received_lamports],
        },
        "transaction": {
            "message": {
                "accountKeys": [sender, recipient],
            },
        },
    }
