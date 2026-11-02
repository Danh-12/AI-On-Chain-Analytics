"""
Sui Blockchain Data Collector
Fetches on-chain data from Sui network via JSON-RPC API.
"""
import json
import time
import requests
import logging
from typing import Dict, List, Optional, Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    SUI_RPC_URL, SUI_RPC_HEADERS, COLLECTION_BATCH_SIZE,
    MAX_TRANSACTIONS, REQUEST_TIMEOUT, REQUEST_DELAY, MAX_RETRIES,
    DATA_RAW_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SuiCollector:
    """Collects transaction and object data from Sui blockchain via JSON-RPC."""

    def __init__(self, rpc_url: str = None):
        self.rpc_url = rpc_url or SUI_RPC_URL
        self.session = requests.Session()
        self.session.headers.update(SUI_RPC_HEADERS)
        self._request_id = 0

    # ── Low-level RPC ────────────────────────────────────────────────

    def _rpc_call(self, method: str, params: list = None) -> Any:
        """Make a JSON-RPC call with retries."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or [],
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.post(
                    self.rpc_url,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    logger.warning(f"RPC error: {data['error']}")
                    return None
                return data.get("result")
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * (attempt + 1))
        return None

    # ── Checkpoint Methods ───────────────────────────────────────────

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint sequence number."""
        result = self._rpc_call("sui_getLatestCheckpointSequenceNumber")
        if result is not None:
            logger.info(f"Latest checkpoint: {result}")
        return result

    def get_checkpoint(self, sequence_number: str) -> Optional[Dict]:
        """Get checkpoint data including transaction digests."""
        return self._rpc_call("sui_getCheckpoint", [str(sequence_number)])

    # ── Transaction Methods ──────────────────────────────────────────

    def get_transaction(self, digest: str) -> Optional[Dict]:
        """Get full transaction block details with balance changes and effects."""
        options = {
            "showInput": True,
            "showEffects": True,
            "showEvents": True,
            "showBalanceChanges": True,
            "showObjectChanges": True,
        }
        return self._rpc_call("sui_getTransactionBlock", [digest, options])

    def query_transactions(
        self,
        cursor: str = None,
        limit: int = None,
        descending: bool = True,
    ) -> Optional[Dict]:
        """Query transaction blocks with pagination."""
        limit = limit or COLLECTION_BATCH_SIZE
        query = {"filter": None, "options": {
            "showInput": True,
            "showEffects": True,
            "showEvents": True,
            "showBalanceChanges": True,
            "showObjectChanges": True,
        }}
        params = [
            query.get("filter"),
            cursor,
            limit,
            descending,
        ]

        result = self._rpc_call("suix_queryTransactionBlocks", [
            {"filter": None, "options": query["options"]},
            cursor,
            limit,
            descending,
        ])
        return result

    # ── Object Methods ───────────────────────────────────────────────

    def get_object(self, object_id: str) -> Optional[Dict]:
        """Get object details."""
        options = {
            "showType": True,
            "showOwner": True,
            "showContent": True,
        }
        return self._rpc_call("sui_getObject", [object_id, options])

    def get_owned_objects(self, address: str, cursor: str = None, limit: int = 50) -> Optional[Dict]:
        """Get objects owned by an address."""
        return self._rpc_call("suix_getOwnedObjects", [
            address,
            {"filter": None, "options": {"showType": True, "showContent": True}},
            cursor,
            limit,
        ])

    # ── Balance Methods ──────────────────────────────────────────────

    def get_all_balances(self, address: str) -> Optional[List]:
        """Get all coin balances for an address."""
        return self._rpc_call("suix_getAllBalances", [address])

    # ── Event Methods ────────────────────────────────────────────────

    def query_events(self, event_filter: Dict, cursor: str = None, limit: int = 50) -> Optional[Dict]:
        """Query events with a filter."""
        return self._rpc_call("suix_queryEvents", [event_filter, cursor, limit, True])

    # ── High-level Collection ────────────────────────────────────────

    def collect_transactions(self, max_count: int = None) -> List[Dict]:
        """
        Collect recent transactions from Sui network.
        Returns a flat list of transaction dicts.
        """
        max_count = max_count or MAX_TRANSACTIONS
        all_txs = []
        cursor = None
        batch_num = 0

        logger.info(f"Starting collection of up to {max_count} transactions...")

        while len(all_txs) < max_count:
            remaining = max_count - len(all_txs)
            batch_size = min(COLLECTION_BATCH_SIZE, remaining)
            batch_num += 1

            result = self.query_transactions(cursor=cursor, limit=batch_size)

            if not result or "data" not in result:
                logger.warning(f"No data returned in batch {batch_num}")
                break

            txs = result["data"]
            all_txs.extend(txs)
            logger.info(f"Batch {batch_num}: collected {len(txs)} txs (total: {len(all_txs)})")

            if not result.get("hasNextPage", False):
                logger.info("No more pages available.")
                break

            cursor = result.get("nextCursor")
            time.sleep(REQUEST_DELAY)

        logger.info(f"Collection complete: {len(all_txs)} transactions")
        return all_txs

    def collect_and_save(self, max_count: int = None) -> List[Dict]:
        """Collect transactions and save raw data to disk."""
        txs = self.collect_transactions(max_count)
        if txs:
            filepath = os.path.join(DATA_RAW_DIR, "raw_transactions.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(txs, f, indent=2, default=str)
            logger.info(f"Saved {len(txs)} transactions to {filepath}")
        return txs


if __name__ == "__main__":
    collector = SuiCollector()
    cp = collector.get_latest_checkpoint()
    print(f"Latest checkpoint: {cp}")
    txs = collector.collect_transactions(max_count=10)
    print(f"Collected {len(txs)} transactions")
    if txs:
        print(json.dumps(txs[0], indent=2, default=str)[:2000])
