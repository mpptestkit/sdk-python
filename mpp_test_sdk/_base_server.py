"""
Base (Ethereum L2) on-chain payment verification helpers.
"""

from __future__ import annotations

import secrets
from typing import Any

from ._rpc import parse_header_params


def _require_web3() -> Any:
    try:
        from web3 import Web3  # noqa: PLC0415
        from web3.middleware import ExtraDataToPOAMiddleware  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "web3 is required for Base chain support. Install it: pip install web3"
        ) from exc
    return Web3, ExtraDataToPOAMiddleware


def random_evm_address() -> str:
    return "0x" + secrets.token_hex(20)


def make_base_web3(rpc_url: str) -> Any:
    Web3, ExtraDataToPOAMiddleware = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


async def verify_base_payment(
    w3: Any,
    receipt_header: str,
    recipient_address: str,
    required_eth: float,
    amount_str: str,
) -> tuple[bool, str]:
    params = parse_header_params(receipt_header)
    tx_hash = params.get("txhash") or params.get("signature")
    if not tx_hash:
        return False, "Payment-Receipt missing txHash field"

    try:
        claimed = float(params.get("amount", "0"))
    except (ValueError, TypeError):
        claimed = 0.0

    if claimed < required_eth:
        return (
            False,
            f"Insufficient payment: claimed {params.get('amount', '0')} ETH, "
            f"required {required_eth} ETH",
        )

    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
    except Exception:
        receipt = None

    if not receipt or receipt.get("status") != 1:
        return False, "Transaction not confirmed on Base"

    try:
        tx = w3.eth.get_transaction(tx_hash)
    except Exception:
        tx = None

    if not tx:
        return False, "Transaction not found on Base"

    to_addr = tx.get("to")
    if not to_addr or to_addr.lower() != recipient_address.lower():
        return False, f"Transaction recipient mismatch: expected {recipient_address}"

    required_wei = w3.to_wei(amount_str, "ether")
    if tx["value"] < required_wei:
        received = w3.from_wei(tx["value"], "ether")
        return (
            False,
            f"Payment too small: received {received} ETH, required {amount_str} ETH",
        )

    return True, ""
