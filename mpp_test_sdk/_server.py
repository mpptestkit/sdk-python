"""
MPP server middleware -HTTP 402 gating with on-chain Solana verification.

Supports both **Flask** and **FastAPI** / Starlette.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Callable

from ._rpc import (
    LAMPORTS_PER_SOL,
    NETWORK_RPC,
    RpcClient,
    SolanaNetwork,
    parse_header_params,
)

try:
    from starlette.requests import Request as _StarletteRequest
except ImportError:  # pragma: no cover
    _StarletteRequest = Any  # type: ignore[assignment,misc]


@dataclass
class TestServerConfig:
    """
    Configuration for :func:`create_test_server`.

    Attributes
    ----------
    network:
        Solana network.  ``"devnet"`` (default), ``"testnet"``, or
        ``"mainnet"``.
    secret_key:
        64-byte server keypair secret.  Auto-generated if omitted.
    recipient_address:
        Override the recipient Solana address (base58).  Defaults to the
        server keypair's public key.
    rpc_url:
        Override the Solana JSON-RPC endpoint.  Takes precedence over
        *network*.
    """

    network: SolanaNetwork = "devnet"
    secret_key: bytes | None = None
    recipient_address: str | None = None
    rpc_url: str | None = None


async def _verify_payment(
    rpc: RpcClient,
    receipt_header: str,
    recipient_address: str,
    required_sol: float,
) -> tuple[bool, str]:
    """
    Verify an on-chain SOL payment described by *receipt_header*.

    Returns ``(success, error_message)``.  On success *error_message* is empty.
    """
    params = parse_header_params(receipt_header)
    sig = params.get("signature")
    if not sig:
        return False, "Payment-Receipt missing signature"

    try:
        claimed = float(params.get("amount", "0"))
    except (ValueError, TypeError):
        claimed = 0.0

    if claimed < required_sol:
        return (
            False,
            f"Insufficient payment: claimed {params.get('amount', '0')} SOL, "
            f"required {required_sol} SOL",
        )

    tx = await rpc.call(
        "getTransaction",
        [
            sig,
            {
                "encoding": "json",
                "commitment": "confirmed",
                "maxSupportedTransactionVersion": 0,
            },
        ],
    )

    if tx is None:
        return False, "Transaction not found on chain"

    if tx["meta"]["err"]:
        return False, "Transaction failed on chain"

    keys: list[str] = tx["transaction"]["message"]["accountKeys"]
    pre: list[int] = tx["meta"]["preBalances"]
    post: list[int] = tx["meta"]["postBalances"]

    try:
        idx = keys.index(recipient_address)
    except ValueError:
        return False, f"Recipient {recipient_address[:8]}... not found in transaction"

    received = (post[idx] - pre[idx]) / LAMPORTS_PER_SOL
    if received < required_sol:
        return (
            False,
            f"Payment too small: received {received} SOL, required {required_sol} SOL",
        )

    return True, ""


def _make_fastapi_dependency(
    *,
    rpc: RpcClient,
    recipient: str,
    network: str,
    required_sol: float,
    amount: str,
    payment_request_hdr: str,
) -> Callable:
    """Build a FastAPI dependency with a concrete ``Request`` annotation."""
    try:
        from starlette.exceptions import HTTPException  # noqa: PLC0415
    except ImportError:
        from fastapi.exceptions import HTTPException  # noqa: PLC0415  # type: ignore[no-redef]

    async def dependency(request: _StarletteRequest) -> None:  # type: ignore[valid-type]
        receipt_header: str = request.headers.get("payment-receipt", "")

        if not receipt_header:
            raise HTTPException(
                status_code=402,
                detail=_payment_required_detail(amount, recipient, network),
                headers={"Payment-Request": payment_request_hdr},
            )

        try:
            ok, err_msg = await _verify_payment(
                rpc, receipt_header, recipient, required_sol
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=403,
                detail=f"Payment verification failed: {exc}",
            ) from exc

        if not ok:
            raise HTTPException(status_code=403, detail=err_msg)

    return dependency


class MppServer:
    """
    MPP server helper that issues 402 challenges and verifies on-chain payments.

    Provides framework-specific decorators/dependencies for Flask and FastAPI.
    """

    def __init__(
        self,
        recipient_address: str,
        network: SolanaNetwork,
        rpc: RpcClient,
    ) -> None:
        self.recipient_address: str = recipient_address
        self.network: SolanaNetwork = network
        self._rpc = rpc

    def _payment_request_header(self, amount: str) -> str:
        return (
            f'solana; amount="{amount}"; '
            f'recipient="{self.recipient_address}"; '
            f'network="{self.network}"'
        )

    def _payment_required_body(self, amount: str) -> dict[str, Any]:
        return {
            "error": "Payment Required",
            "payment": {
                "amount": amount,
                "currency": "SOL",
                "recipient": self.recipient_address,
                "network": self.network,
            },
        }

    def flask_charge(self, amount: str) -> Callable:
        """
        Flask decorator that enforces SOL payment before the route handler runs.
        """
        required_sol = float(amount)

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                from flask import jsonify, request  # noqa: PLC0415

                receipt_header = request.headers.get("payment-receipt", "")

                if not receipt_header:
                    response = jsonify(self._payment_required_body(amount))
                    response.status_code = 402
                    response.headers["Payment-Request"] = self._payment_request_header(
                        amount
                    )
                    return response

                try:
                    ok, err_msg = asyncio.run(
                        _verify_payment(
                            self._rpc,
                            receipt_header,
                            self.recipient_address,
                            required_sol,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    response = jsonify({"error": f"Payment verification failed: {exc}"})
                    response.status_code = 403
                    return response

                if not ok:
                    response = jsonify({"error": err_msg})
                    response.status_code = 403
                    return response

                if asyncio.iscoroutinefunction(fn):
                    return asyncio.run(fn(*args, **kwargs))
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    def fastapi_charge(self, amount: str) -> Callable:
        """FastAPI dependency that enforces SOL payment."""
        required_sol = float(amount)
        payment_request_hdr = self._payment_request_header(amount)

        return _make_fastapi_dependency(
            rpc=self._rpc,
            recipient=self.recipient_address,
            network=self.network,
            required_sol=required_sol,
            amount=amount,
            payment_request_hdr=payment_request_hdr,
        )

    def charge(self, amount: str) -> Callable:
        """Alias for :meth:`fastapi_charge`."""
        return self.fastapi_charge(amount)


def _payment_required_detail(
    amount: str, recipient: str, network: str
) -> dict[str, Any]:
    return {
        "error": "Payment Required",
        "payment": {
            "amount": amount,
            "currency": "SOL",
            "recipient": recipient,
            "network": network,
        },
    }


def create_test_server(config: TestServerConfig | None = None) -> MppServer:
    """Create a Solana MPP-enabled server helper."""
    from solders.keypair import Keypair  # noqa: PLC0415

    cfg = config or TestServerConfig()
    network: SolanaNetwork = cfg.network
    rpc_url: str = cfg.rpc_url or NETWORK_RPC[network]

    if cfg.secret_key is not None:
        server_keypair = Keypair.from_bytes(cfg.secret_key)
    else:
        server_keypair = Keypair()

    recipient_address: str = cfg.recipient_address or str(server_keypair.pubkey())

    return MppServer(
        recipient_address=recipient_address,
        network=network,
        rpc=RpcClient(rpc_url),
    )
