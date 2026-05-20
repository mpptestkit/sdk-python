"""
Solana JSON-RPC client and shared protocol helpers.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

import httpx

LAMPORTS_PER_SOL: int = 1_000_000_000

SolanaNetwork = Literal["devnet", "testnet", "mainnet"]

NETWORK_RPC: dict[str, str] = {
    "devnet": "https://api.devnet.solana.com",
    "testnet": "https://api.testnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
}


def parse_header_params(header: str) -> dict[str, str]:
    """
    Parse a structured HTTP header value of the form::

        solana; key1="val1"; key2="val2"

    Returns lowercase key → unquoted value pairs (the scheme token is skipped).
    """
    params: dict[str, str] = {}
    semi = header.find(";")
    if semi < 0:
        return params
    header = header[semi + 1 :]
    while header:
        semi = header.find(";")
        if semi >= 0:
            part, header = header[:semi], header[semi + 1 :]
        else:
            part, header = header, ""
        part = part.strip()
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        key = key.strip().lower()
        val = val.strip().strip('"')
        if key:
            params[key] = val
    return params


class RpcClient:
    """Reusable async JSON-RPC client for a single Solana endpoint."""

    def __init__(self, endpoint: str, *, timeout: float = 30.0) -> None:
        self._endpoint = endpoint
        self._http = httpx.AsyncClient(timeout=timeout)

    async def call(self, method: str, params: list[Any]) -> Any:
        """Execute a JSON-RPC call and return the ``result`` field."""
        response = await self._http.post(
            self._endpoint,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"]["message"])
        return data["result"]

    async def get_latest_blockhash(self) -> str:
        result = await self.call("getLatestBlockhash", [{"commitment": "confirmed"}])
        return result["value"]["blockhash"]

    async def wait_for_signature(
        self,
        signature: str,
        *,
        deadline: float = 60.0,
    ) -> None:
        """Poll until *signature* is confirmed or *deadline* seconds elapse."""
        end = time.monotonic() + deadline
        poll = 0.4
        max_poll = 2.0

        while time.monotonic() < end:
            statuses = await self.call(
                "getSignatureStatuses",
                [[signature], {"searchTransactionHistory": True}],
            )
            entries = statuses.get("value") or []
            if entries and entries[0]:
                status = entries[0]
                if status.get("err"):
                    raise RuntimeError(f"Transaction {signature} failed on chain")
                conf = status.get("confirmationStatus")
                if conf in ("confirmed", "finalized"):
                    return

            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll, remaining))
            poll = min(poll * 1.5, max_poll)

        raise RuntimeError(f"Timed out waiting for signature {signature}")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> RpcClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


# Per-endpoint client cache used by the legacy _rpc_call helper (and tests).
_client_cache: dict[str, RpcClient] = {}


async def _rpc_call(rpc_url: str, method: str, params: list[Any]) -> Any:
    """Legacy JSON-RPC helper; prefer :class:`RpcClient` on hot paths."""
    client = _client_cache.setdefault(rpc_url, RpcClient(rpc_url))
    return await client.call(method, params)
