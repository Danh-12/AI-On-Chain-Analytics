"""
Data Preprocessor & Feature Engineer
Transforms raw Sui transaction data into ML-ready features.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Processes raw Sui transactions into structured DataFrames and feature sets."""

    # ── Transaction Flattening ───────────────────────────────────────

    def process_transactions(self, raw_txs: List[Dict]) -> pd.DataFrame:
        """
        Flatten raw transaction blocks into a clean DataFrame.
        Extracts: digest, sender, recipients, gas, amounts, timestamp, type, status.
        """
        records = []
        for tx in raw_txs:
            try:
                record = self._extract_tx_record(tx)
                if record:
                    records.append(record)
            except Exception as e:
                logger.debug(f"Skipping malformed tx: {e}")
                continue

        if not records:
            logger.warning("No valid transactions extracted")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Type conversions
        if "timestamp_ms" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"].astype(float), unit="ms", errors="coerce")
        if "gas_used" in df.columns:
            df["gas_used"] = pd.to_numeric(df["gas_used"], errors="coerce").fillna(0)
        if "total_amount" in df.columns:
            df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)

        logger.info(f"Processed {len(df)} transactions into DataFrame")
        return df

    def _extract_tx_record(self, tx: Dict) -> Optional[Dict]:
        """Extract a flat record from a single transaction block."""
        digest = tx.get("digest", "")

        # ── Sender
        tx_input = tx.get("transaction", {})
        sender = tx_input.get("data", {}).get("sender", "unknown")

        # ── Effects
        effects = tx.get("effects", {})
        status = effects.get("status", {}).get("status", "unknown")
        gas_summary = effects.get("gasUsed", {})
        gas_used = (
            int(gas_summary.get("computationCost", 0)) +
            int(gas_summary.get("storageCost", 0)) -
            int(gas_summary.get("storageRebate", 0))
        )

        # ── Timestamp
        timestamp_ms = tx.get("timestampMs", 0)

        # ── Balance Changes → recipients and amounts
        balance_changes = tx.get("balanceChanges", []) or []
        recipients = set()
        total_amount = 0
        sui_amount = 0
        token_types = set()

        for bc in balance_changes:
            owner = bc.get("owner", {})
            if isinstance(owner, dict):
                addr = owner.get("AddressOwner", "")
            else:
                addr = str(owner)

            if addr and addr != sender:
                recipients.add(addr)

            amount = abs(int(bc.get("amount", 0)))
            total_amount += amount
            coin_type = bc.get("coinType", "")
            token_types.add(coin_type)
            if "sui" in coin_type.lower():
                sui_amount += amount

        # ── Object Changes
        object_changes = tx.get("objectChanges", []) or []
        created_objects = sum(1 for o in object_changes if o.get("type") == "created")
        mutated_objects = sum(1 for o in object_changes if o.get("type") == "mutated")
        deleted_objects = sum(1 for o in object_changes if o.get("type") == "deleted")

        # ── Transaction Type (from programmable tx kind)
        tx_kind = tx_input.get("data", {}).get("transaction", {}).get("kind", "unknown")

        # ── Events count
        events = tx.get("events", []) or []

        return {
            "digest": digest,
            "sender": sender,
            "recipients": list(recipients),
            "num_recipients": len(recipients),
            "status": status,
            "gas_used": gas_used,
            "timestamp_ms": timestamp_ms,
            "total_amount": total_amount,
            "sui_amount": sui_amount,
            "num_token_types": len(token_types),
            "token_types": list(token_types),
            "created_objects": created_objects,
            "mutated_objects": mutated_objects,
            "deleted_objects": deleted_objects,
            "tx_kind": tx_kind,
            "num_events": len(events),
            "is_self_transfer": sender in recipients,
        }

    # ── Wallet Profiling ─────────────────────────────────────────────

    def build_wallet_profiles(self, tx_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-wallet statistics for classification.
        Returns a DataFrame indexed by wallet address with features.
        """
        if tx_df.empty:
            return pd.DataFrame()

        profiles = defaultdict(lambda: {
            "tx_count": 0,
            "sent_count": 0,
            "received_count": 0,
            "total_sent_amount": 0,
            "total_received_amount": 0,
            "total_gas_spent": 0,
            "unique_counterparties": set(),
            "self_transfers": 0,
            "timestamps": [],
            "gas_values": [],
            "amounts": [],
            "object_creates": 0,
            "event_counts": [],
            "token_types_used": set(),
        })

        for _, row in tx_df.iterrows():
            sender = row["sender"]
            p = profiles[sender]
            p["tx_count"] += 1
            p["sent_count"] += 1
            p["total_sent_amount"] += row.get("total_amount", 0)
            p["total_gas_spent"] += row.get("gas_used", 0)
            p["self_transfers"] += int(row.get("is_self_transfer", False))
            p["object_creates"] += row.get("created_objects", 0)
            p["event_counts"].append(row.get("num_events", 0))
            p["gas_values"].append(row.get("gas_used", 0))
            p["amounts"].append(row.get("total_amount", 0))

            if "timestamp" in row and pd.notna(row["timestamp"]):
                p["timestamps"].append(row["timestamp"])

            for r in row.get("recipients", []):
                p["unique_counterparties"].add(r)

            for t in row.get("token_types", []):
                p["token_types_used"].add(t)

            # Also record the recipient side
            for r in row.get("recipients", []):
                rp = profiles[r]
                rp["tx_count"] += 1
                rp["received_count"] += 1
                rp["total_received_amount"] += row.get("total_amount", 0)
                rp["unique_counterparties"].add(sender)

        # Convert to feature vectors
        records = []
        for addr, p in profiles.items():
            timestamps = sorted(p["timestamps"])
            time_diffs = []
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    time_diffs.append(diff)

            gas_arr = np.array(p["gas_values"]) if p["gas_values"] else np.array([0])
            amt_arr = np.array(p["amounts"]) if p["amounts"] else np.array([0])

            records.append({
                "address": addr,
                "tx_count": p["tx_count"],
                "sent_count": p["sent_count"],
                "received_count": p["received_count"],
                "total_sent_amount": p["total_sent_amount"],
                "total_received_amount": p["total_received_amount"],
                "total_gas_spent": p["total_gas_spent"],
                "unique_counterparties": len(p["unique_counterparties"]),
                "self_transfer_ratio": p["self_transfers"] / max(p["sent_count"], 1),
                "avg_amount": float(amt_arr.mean()),
                "std_amount": float(amt_arr.std()),
                "max_amount": float(amt_arr.max()),
                "avg_gas": float(gas_arr.mean()),
                "std_gas": float(gas_arr.std()),
                "avg_time_between_txs": float(np.mean(time_diffs)) if time_diffs else 0,
                "std_time_between_txs": float(np.std(time_diffs)) if time_diffs else 0,
                "min_time_between_txs": float(np.min(time_diffs)) if time_diffs else 0,
                "object_creates": p["object_creates"],
                "avg_events": float(np.mean(p["event_counts"])) if p["event_counts"] else 0,
                "num_token_types": len(p["token_types_used"]),
                "sent_received_ratio": p["sent_count"] / max(p["received_count"], 1),
            })

        df = pd.DataFrame(records)
        logger.info(f"Built profiles for {len(df)} wallets")
        return df

    # ── Token Metrics ────────────────────────────────────────────────

    def build_token_metrics(self, tx_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-token metrics for anomaly detection.
        """
        if tx_df.empty:
            return pd.DataFrame()

        token_data = defaultdict(lambda: {
            "transfer_count": 0,
            "total_volume": 0,
            "unique_senders": set(),
            "unique_receivers": set(),
            "amounts": [],
            "timestamps": [],
        })

        for _, row in tx_df.iterrows():
            for token in row.get("token_types", []):
                t = token_data[token]
                t["transfer_count"] += 1
                t["total_volume"] += row.get("total_amount", 0)
                t["unique_senders"].add(row["sender"])
                t["amounts"].append(row.get("total_amount", 0))

                if "timestamp" in row and pd.notna(row["timestamp"]):
                    t["timestamps"].append(row["timestamp"])

                for r in row.get("recipients", []):
                    t["unique_receivers"].add(r)

        records = []
        for token, t in token_data.items():
            amt_arr = np.array(t["amounts"]) if t["amounts"] else np.array([0])
            n_senders = len(t["unique_senders"])
            n_receivers = len(t["unique_receivers"])

            # Gini coefficient for amount concentration
            gini = self._gini_coefficient(amt_arr)

            records.append({
                "token_type": token,
                "transfer_count": t["transfer_count"],
                "total_volume": t["total_volume"],
                "unique_senders": n_senders,
                "unique_receivers": n_receivers,
                "avg_transfer_amount": float(amt_arr.mean()),
                "std_transfer_amount": float(amt_arr.std()),
                "max_transfer_amount": float(amt_arr.max()),
                "sender_receiver_ratio": n_senders / max(n_receivers, 1),
                "volume_per_sender": t["total_volume"] / max(n_senders, 1),
                "gini_coefficient": gini,
            })

        df = pd.DataFrame(records)
        logger.info(f"Built metrics for {len(df)} token types")
        return df

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """Calculate Gini coefficient for concentration measurement."""
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals)))

    @staticmethod
    def normalize_features(df: pd.DataFrame, columns: List[str] = None) -> Tuple[pd.DataFrame, object]:
        """Normalize numerical features using StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[columns] = scaler.fit_transform(df[columns].fillna(0))
        return df_scaled, scaler
