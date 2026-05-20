# mpp-test-sdk (Python)

Test pay-per-request APIs on Solana -devnet, testnet, or mainnet.

## Installation

```bash
pip install mpp-test-sdk
```

## Quick start

```python
from mpp_test_sdk import create_test_client

client = await create_test_client()
res = await client.fetch("http://localhost:3001/api/data")
```
