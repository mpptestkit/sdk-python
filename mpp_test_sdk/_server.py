"""
MPP server middleware — HTTP 402 gating with on-chain Solana verification.

Supports both **Flask** and **FastAPI** / Starlette.

Usage (Flask)::

    from flask import Flask
    from mpp_test_sdk import create_test_server

    app = Flask(__name__)
    mpp = create_test_server()

    @app.get("/api/data")
    @mpp.flask_charge("0.001")
    def data():
        return {"data": "premium content"}

Usage (FastAPI)::

    from fastapi import FastAPI, Depends
    from mpp_test_sdk import create_test_server

    app = FastAPI()
    mpp = create_test_server()

    @app.get("/api/data")
    async def data(dep=Depends(mpp.charge("0.001"))):
        return {"data": "premium content"}
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from ._client import LAMPORTS_PER_SOL, NETWORK_RPC, SolanaNetwork, _parse_header_params, _rpc_call

# Attempt a top-level import of starlette.requests.Request so that FastAPI's
# dependency-injection engine can resolve the annotation without string-eval.
try:
    from starlette.requests import Request as _StarletteRequest  # noqa: TCH002
    _HAS_STARLETTE = True
except ImportError:  # pragma: no cover
    _StarletteRequest = Any  # type: ignore[assignment,misc]
    _HAS_STARLETTE = False

# ─── Config ───────────────────────────────────────────────────────────────────


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


# ─── Payment verification ─────────────────────────────────────────────────────


async def _verify_payment(
    rpc_url: str,
    receipt_header: str,
    recipient_address: str,
    required_sol: float,
) -> tuple[bool, str]:
    """
    Verify an on-chain SOL payment described by *receipt_header*.

    Returns a ``(success, error_message)`` tuple.  On success *error_message*
    is an empty string.
    """
    params = _parse_header_params(receipt_header)
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

    tx = await _rpc_call(
        rpc_url,
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


# ─── FastAPI dependency factory ───────────────────────────────────────────────


def _make_fastapi_dependency(
    *,
    rpc_url: str,
    recipient: str,
    network: str,
    required_sol: float,
    amount: str,
    payment_request_hdr: str,
) -> Callable:
    """
    Build a FastAPI-compatible async dependency function with a concrete
    ``starlette.requests.Request`` annotation so the DI engine injects it.

    This is a module-level factory (not a method) so the annotation refers to
    the module-level ``_StarletteRequest`` name which is always resolvable.
    """
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
                rpc_url, receipt_header, recipient, required_sol
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=403,
                detail=f"Payment verification failed: {exc}",
            ) from exc

        if not ok:
            raise HTTPException(status_code=403, detail=err_msg)

    return dependency


# ─── MppServer ────────────────────────────────────────────────────────────────


class MppServer:
    """
    MPP server helper that issues 402 challenges and verifies on-chain payments.

    Provides framework-specific decorators/dependencies for Flask and FastAPI.

    Do not instantiate directly; use :func:`create_test_server` instead.

    Attributes
    ----------
    recipient_address:
        Base58 Solana address where payments are sent.
    network:
        Solana network this server is configured for.
    """

    def __init__(
        self,
        recipient_address: str,
        network: SolanaNetwork,
        rpc_url: str,
    ) -> None:
        self.recipient_address: str = recipient_address
        self.network: SolanaNetwork = network
        self._rpc_url: str = rpc_url

    # ─── Internal ─────────────────────────────────────────────────────────────

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

    # ─── Flask ────────────────────────────────────────────────────────────────

    def flask_charge(self, amount: str) -> Callable:
        """
        Flask decorator that enforces SOL payment before the route handler runs.

        - **No receipt** → 402 with ``Payment-Request`` header.
        - **Valid receipt + on-chain confirmation** → handler is called normally.
        - **Invalid or insufficient payment** → 403.

        Parameters
        ----------
        amount:
            Required SOL amount as a string, e.g. ``"0.001"``.

        Examples
        --------
        ::

            @app.get("/api/data")
            @mpp.flask_charge("0.001")
            def data():
                return jsonify({"data": "premium"})
        """
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                from flask import jsonify, request  # noqa: PLC0415

                receipt_header = request.headers.get("payment-receipt", "")

                if not receipt_header:
                    response = jsonify(self._payment_required_body(amount))
                    response.status_code = 402
                    response.headers["Payment-Request"] = self._payment_request_header(amount)
                    return response

                # Verify on-chain
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        ok, err_msg = loop.run_until_complete(
                            _verify_payment(
                                self._rpc_url,
                                receipt_header,
                                self.recipient_address,
                                float(amount),
                            )
                        )
                    finally:
                        loop.close()
                except Exception as exc:  # noqa: BLE001
                    response = jsonify({"error": f"Payment verification failed: {exc}"})
                    response.status_code = 403
                    return response

                if not ok:
                    response = jsonify({"error": err_msg})
                    response.status_code = 403
                    return response

                # Payment verified — call the actual route handler
                if asyncio.iscoroutinefunction(fn):
                    loop2 = asyncio.new_event_loop()
                    try:
                        return loop2.run_until_complete(fn(*args, **kwargs))
                    finally:
                        loop2.close()
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    # ─── FastAPI ──────────────────────────────────────────────────────────────

    def fastapi_charge(self, amount: str) -> Callable:
        """
        FastAPI dependency that enforces SOL payment.

        Use with ``Depends``::

            @app.get("/api/data")
            async def data(dep=Depends(mpp.fastapi_charge("0.001"))):
                return {"data": "premium"}

        - **No receipt** → raises ``HTTPException(402)`` with ``Payment-Request``
          header.
        - **Valid receipt + on-chain confirmation** → dependency resolves.
        - **Invalid or insufficient payment** → raises ``HTTPException(403)``.

        Parameters
        ----------
        amount:
            Required SOL amount as a string, e.g. ``"0.001"``.
        """
        rpc_url = self._rpc_url
        recipient = self.recipient_address
        network = self.network
        required_sol = float(amount)
        payment_request_hdr = self._payment_request_header(amount)

        # Build the dependency with a concrete starlette Request annotation so
        # FastAPI's dependency-injection engine can recognise it.  We create the
        # function inside a helper that captures the correct annotation at call
        # time rather than relying on a forward-reference string.
        dep = _make_fastapi_dependency(
            rpc_url=rpc_url,
            recipient=recipient,
            network=network,
            required_sol=required_sol,
            amount=amount,
            payment_request_hdr=payment_request_hdr,
        )
        return dep

    def charge(self, amount: str) -> Callable:
        """
        Alias for :meth:`fastapi_charge` — the most common use case.

        For Flask routes use :meth:`flask_charge` instead.

        Parameters
        ----------
        amount:
            Required SOL amount as a string, e.g. ``"0.001"``.
        """
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


# ─── create_test_server ───────────────────────────────────────────────────────


def create_test_server(config: TestServerConfig | None = None) -> MppServer:
    """
    Create a Solana MPP-enabled server helper.

    Auto-generates a server wallet if no *secret_key* is provided.

    Parameters
    ----------
    config:
        Optional :class:`TestServerConfig`.  Defaults to devnet with an
        auto-generated keypair.

    Returns
    -------
    MppServer

    Examples
    --------
    Flask::

        from flask import Flask
        from mpp_test_sdk import create_test_server

        app = Flask(__name__)
        mpp = create_test_server()

        @app.get("/api/data")
        @mpp.flask_charge("0.001")
        def data():
            return {"data": "premium"}

    FastAPI::

        from fastapi import FastAPI, Depends
        from mpp_test_sdk import create_test_server

        app = FastAPI()
        mpp = create_test_server()

        @app.get("/api/data")
        async def data(dep=Depends(mpp.charge("0.001"))):
            return {"data": "premium"}
    """
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
        rpc_url=rpc_url,
    )
