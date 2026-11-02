"""
Wallet Behavior Classifier
Categorizes Sui wallets into behavioral types: Bot, Whale, Retail, DEX Router, Suspicious.
Uses KMeans clustering for unsupervised labeling + Random Forest for classification.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import KMEANS_N_CLUSTERS, RANDOM_FOREST_N_ESTIMATORS

logger = logging.getLogger(__name__)


# Wallet categories
WALLET_TYPES = {
    0: "Retail",
    1: "Bot",
    2: "Whale",
    3: "DEX/Contract",
    4: "Suspicious",
}


class WalletClassifier:
    """
    Classifies Sui wallets based on behavioral patterns.
    1. KMeans clustering to discover natural groups
    2. Heuristic label assignment based on cluster centroids
    3. Random Forest for final classification with feature importances
    """

    CLASSIFICATION_FEATURES = [
        "tx_count", "sent_count", "received_count",
        "total_sent_amount", "total_received_amount",
        "total_gas_spent", "unique_counterparties",
        "self_transfer_ratio", "avg_amount", "std_amount",
        "max_amount", "avg_gas", "std_gas",
        "avg_time_between_txs", "std_time_between_txs",
        "min_time_between_txs", "object_creates",
        "avg_events", "num_token_types", "sent_received_ratio",
    ]

    def __init__(self):
        self.kmeans = KMeans(
            n_clusters=KMEANS_N_CLUSTERS,
            random_state=42,
            n_init=10,
        )
        self.rf_classifier = RandomForestClassifier(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    # ── Main Analysis ────────────────────────────────────────────────

    def analyze(self, wallet_df: pd.DataFrame) -> Dict:
        """
        Classify all wallets and return results.
        """
        if wallet_df.empty or len(wallet_df) < KMEANS_N_CLUSTERS:
            return self._empty_result(wallet_df)

        logger.info(f"Classifying {len(wallet_df)} wallets...")

        # 1. Prepare features
        features = [f for f in self.CLASSIFICATION_FEATURES if f in wallet_df.columns]
        X = wallet_df[features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        # 2. KMeans clustering
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # 3. Assign wallet types based on cluster characteristics
        wallet_types = self._assign_types_from_clusters(wallet_df, features, cluster_labels)

        # 4. Train Random Forest on the assigned labels for feature importance
        self.rf_classifier.fit(X_scaled, wallet_types)
        self._is_fitted = True
        feature_importances = dict(zip(features, self.rf_classifier.feature_importances_))

        # 5. Build result
        wallet_df = wallet_df.copy()
        wallet_df["cluster"] = cluster_labels
        wallet_df["wallet_type"] = wallet_types
        wallet_df["wallet_type_name"] = wallet_df["wallet_type"].map(WALLET_TYPES)

        # Distribution
        type_counts = wallet_df["wallet_type_name"].value_counts().to_dict()

        result = {
            "total_wallets": len(wallet_df),
            "type_distribution": type_counts,
            "feature_importances": {k: round(v, 4) for k, v in
                                     sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)},
            "wallet_profiles": self._profiles_to_records(wallet_df),
            "cluster_centers": self._cluster_centers_to_records(features),
        }

        logger.info(f"Classification complete: {type_counts}")
        return result

    # ── Cluster → Type Assignment ────────────────────────────────────

    def _assign_types_from_clusters(
        self,
        wallet_df: pd.DataFrame,
        features: List[str],
        cluster_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Assign wallet types based on cluster characteristics.
        Uses heuristic rules on cluster centroids.
        """
        wallet_types = np.zeros(len(wallet_df), dtype=int)  # default: Retail

        for cluster_id in range(KMEANS_N_CLUSTERS):
            mask = cluster_labels == cluster_id
            cluster_wallets = wallet_df[mask]

            if cluster_wallets.empty:
                continue

            # Compute cluster characteristics
            avg_tx = cluster_wallets["tx_count"].mean()
            avg_amount = cluster_wallets.get("avg_amount", pd.Series(0)).mean()
            avg_time = cluster_wallets.get("avg_time_between_txs", pd.Series(0)).mean()
            avg_self_ratio = cluster_wallets.get("self_transfer_ratio", pd.Series(0)).mean()
            avg_counterparties = cluster_wallets.get("unique_counterparties", pd.Series(0)).mean()
            avg_events = cluster_wallets.get("avg_events", pd.Series(0)).mean()

            # Classification heuristics
            total_avg_tx = wallet_df["tx_count"].mean()
            total_avg_amount = wallet_df.get("avg_amount", pd.Series(0)).mean()

            if avg_self_ratio > 0.4 or (avg_tx > total_avg_tx * 3 and avg_counterparties < 3):
                # Suspicious: high self-transfer or high frequency with few counterparties
                wallet_types[mask] = 4
            elif avg_tx > total_avg_tx * 2 and avg_time < 60 and avg_time > 0:
                # Bot: high frequency, very regular timing (< 60s between txs)
                wallet_types[mask] = 1
            elif avg_amount > total_avg_amount * 5:
                # Whale: very high value transactions
                wallet_types[mask] = 2
            elif avg_events > 3 or avg_counterparties > total_avg_tx * 0.5:
                # DEX/Contract: many events or interacts with many addresses
                wallet_types[mask] = 3
            else:
                # Retail: normal patterns
                wallet_types[mask] = 0

        return wallet_types

    # ── Helpers ───────────────────────────────────────────────────────

    def _profiles_to_records(self, wallet_df: pd.DataFrame) -> List[Dict]:
        """Convert wallet profiles to serializable records."""
        records = []
        for _, row in wallet_df.iterrows():
            records.append({
                "address": row.get("address", ""),
                "wallet_type": row.get("wallet_type_name", "Unknown"),
                "cluster": int(row.get("cluster", 0)),
                "tx_count": int(row.get("tx_count", 0)),
                "total_sent": float(row.get("total_sent_amount", 0)),
                "total_received": float(row.get("total_received_amount", 0)),
                "unique_counterparties": int(row.get("unique_counterparties", 0)),
                "self_transfer_ratio": round(float(row.get("self_transfer_ratio", 0)), 4),
                "avg_amount": round(float(row.get("avg_amount", 0)), 2),
                "avg_gas": round(float(row.get("avg_gas", 0)), 2),
            })
        return sorted(records, key=lambda x: x["tx_count"], reverse=True)

    def _cluster_centers_to_records(self, features: List[str]) -> List[Dict]:
        """Get cluster centers as records."""
        centers = []
        for i, center in enumerate(self.kmeans.cluster_centers_):
            centers.append({
                "cluster": i,
                "type": WALLET_TYPES.get(i, "Unknown"),
                "features": {f: round(float(v), 4) for f, v in zip(features, center)}
            })
        return centers

    def _empty_result(self, wallet_df: pd.DataFrame) -> Dict:
        n = len(wallet_df) if wallet_df is not None else 0
        profiles = []
        if wallet_df is not None and not wallet_df.empty:
            wallet_df = wallet_df.copy()
            wallet_df["wallet_type_name"] = "Retail"
            wallet_df["cluster"] = 0
            profiles = self._profiles_to_records(wallet_df)
        return {
            "total_wallets": n,
            "type_distribution": {"Retail": n},
            "feature_importances": {},
            "wallet_profiles": profiles,
            "cluster_centers": [],
        }
