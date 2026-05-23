"""
MPP test client for Base (Ethereum L2) payments.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx

from ._chain import BASE_NETWORK_RPC, BaseNetwork
from ._rpc import parse_header_params
from .errors import MppPaymentError, MppTimeoutError
from ._client import PaymentStep, TestClientConfig


def _require_web3() -> Any:
    try:
        from web3 import Web3  # noqa: PLC0415
        from web3.middleware import ExtraDataToPOAMiddleware  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "web3 is required for Base chain support. Install it: pip install web3"
        ) from exc
    return Web3, ExtraDataToPOAMiddleware


async def create_base_test_client(config: TestClientConfig) -> "BaseTestClient":
    Web3, ExtraDataToPOAMiddleware = _require_web3()
    from eth_account import Account  # noqa: PLC0415

    emit: Callable[[PaymentStep], None] = config.on_step or (lambda _: None)
    base_network: BaseNetwork = config.base_network or "sepolia"
    rpc_url = config.rpc_url or BASE_NETWORK_RPC[base_network]

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    if config.private_key:
        account = Account.from_key(config.private_key)
    else:
        account = Account.create()
    address = account.address

    emit(
        PaymentStep(
            type="wallet-created",
            message=f"Wallet {address}",
            data={"address": address, "network": base_network, "chain": "base"},
        )
    )

    if not config.private_key:
        emit(
            PaymentStep(
                type="funded",
                message="Fund your wallet at https://coinbase.com/faucets/base-ethereum-sepolia-faucet",
                data={
                    "network": base_network,
                    "faucet": "https://coinbase.com/faucets/base-ethereum-sepolia-faucet",
                },
            )
        )

    http = httpx.AsyncClient(timeout=config.timeout)
    return BaseTestClient(
        address=address,
        network=base_network,
        account=account,
        w3=w3,
        http=http,
        on_step=emit,
        timeout=config.timeout,
    )


class BaseTestClient:
    """MPP async HTTP client backed by a Base (EVM) wallet."""

    def __init__(
        self,
        address: str,
        network: BaseNetwork,
        *,
        account: Any,
        w3: Any,
        http: httpx.AsyncClient,
        on_step: Callable[[PaymentStep], None],
        timeout: float,
    ) -> None:
        self.address = address
        self.network: BaseNetwork = network
        self.chain = "base"
        self.method = "base"
        self._account = account
        self._w3 = w3
        self._http = http
        self._on_step = on_step
        self._timeout = timeout

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch(self, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return await asyncio.wait_for(
                self._fetch_inner(url, **kwargs),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise MppTimeoutError(url, int(self._timeout * 1000)) from exc

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
                url, 402, ValueError("Payment-Request missing recipient")
            )
        if not params.get("amount"):
            raise MppPaymentError(url, 402, ValueError("Payment-Request missing amount"))

        try:
            amount_eth = float(params["amount"])
            if amount_eth <= 0:
                raise ValueError(f"Invalid payment amount: {params['amount']}")
        except (ValueError, TypeError) as exc:
            raise MppPaymentError(
                url,
                402,
                ValueError(f"Invalid payment amount: {params.get('amount')}"),
            ) from exc

        recipient = params["recipient"]
        emit(
            PaymentStep(
                type="payment",
                message=f"Paying {amount_eth} ETH → {recipient[:10]}...",
                data={"amount": amount_eth, "recipient": recipient, "chain": "base"},
            )
        )

        tx_hash = await self._send_eth(recipient, params["amount"])

        emit(
            PaymentStep(
                type="payment",
                message=f"Confirmed: {tx_hash[:18]}...",
                data={"txHash": tx_hash, "amount": amount_eth, "chain": "base"},
            )
        )

        emit(
            PaymentStep(
                type="retry",
                message="↑ Retrying with payment proof",
                data={"txHash": tx_hash},
            )
        )

        receipt_header = (
            f'base; txHash="{tx_hash}"; '
            f'network="{self.network}"; '
            f'amount="{amount_eth}"'
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
                data={"status": retry_res.status_code, "txHash": tx_hash, "chain": "base"},
            )
        )

        return retry_res

    async def _send_eth(self, recipient: str, amount_eth: str) -> str:
        w3 = self._w3
        account = self._account
        value = w3.to_wei(amount_eth, "ether")
        nonce = w3.eth.get_transaction_count(account.address)
        gas_price = w3.eth.gas_price
        tx = {
            "nonce": nonce,
            "to": w3.to_checksum_address(recipient),
            "value": value,
            "gas": 21000,
            "gasPrice": gas_price,
            "chainId": w3.eth.chain_id,
        }
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.get("status") != 1:
            raise RuntimeError("Base payment transaction reverted")
        return receipt["transactionHash"].hex()
