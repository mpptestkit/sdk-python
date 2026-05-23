"""
Shared chain types and RPC endpoints for Solana and Base.
"""

from __future__ import annotations

from typing import Literal

ChainType = Literal["solana", "base"]
BaseNetwork = Literal["sepolia", "mainnet"]

BASE_NETWORK_RPC: dict[str, str] = {
    "sepolia": "https://sepolia.base.org",
    "mainnet": "https://mainnet.base.org",
}
