"""
mpp-test-sdk -Test pay-per-request APIs on Solana (devnet, testnet, or mainnet).

Quick start::

    from mpp_test_sdk import create_test_client, mpp_fetch, mpp_fetch_reset
    from mpp_test_sdk import create_test_server, MppServer

    # Client
    client = await create_test_client()
    res = await client.fetch("http://localhost:3001/api/data")

    # Shared lazy client
    res = await mpp_fetch("http://localhost:3001/api/data")
    mpp_fetch_reset()

    # Server (Flask)
    mpp = create_test_server()

    @app.get("/api/data")
    @mpp.flask_charge("0.001")
    def data():
        return {"ok": True}

    # Server (FastAPI)
    @app.get("/api/data")
    async def data(dep=Depends(mpp.charge("0.001"))):
        return {"ok": True}
"""

from ._client import (
    PaymentStep,
    TestClient,
    TestClientConfig,
    create_test_client,
    mpp_fetch,
    mpp_fetch_reset,
)
from ._rpc import LAMPORTS_PER_SOL, NETWORK_RPC, SolanaNetwork
from ._server import MppServer, TestServerConfig, create_test_server
from .errors import (
    MppError,
    MppFaucetError,
    MppNetworkError,
    MppPaymentError,
    MppTimeoutError,
)

__version__ = "1.1.0"

__all__ = [
    # Client
    "create_test_client",
    "mpp_fetch",
    "mpp_fetch_reset",
    "TestClient",
    "TestClientConfig",
    "PaymentStep",
    "SolanaNetwork",
    # Server
    "create_test_server",
    "MppServer",
    "TestServerConfig",
    # Errors
    "MppError",
    "MppFaucetError",
    "MppPaymentError",
    "MppTimeoutError",
    "MppNetworkError",
    # Metadata
    "__version__",
]
