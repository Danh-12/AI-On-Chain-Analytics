"""
Fraud & Scam Detection Module
Detects malicious transactions, rug pulls, and wash trading on Sui.
Uses Isolation Forest for anomaly scoring and Random Forest for classification.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_N_ESTIMATORS,
)

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Detects fraudulent patterns in Sui blockchain transactions.
    - Isolation Forest for unsupervised anomaly detection
    - Heuristic rules for rug pull and wash trading indicators
    - Combined risk scoring
    """

    FRAUD_FEATURES = [
        "gas_used", "total_amount", "num_recipients",
        "created_objects", "mutated_objects", "deleted_objects",
        "num_events", "num_token_types",
    ]

    def __init__(self):
        self.iso_forest = IsolationForest(
            contamination=ISOLATION_FOREST_CONTAMINATION,
            n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    # ── Main Analysis ────────────────────────────────────────────────

    def analyze(self, tx_df: pd.DataFrame, wallet_df: pd.DataFrame = None) -> Dict:
        """
        Run full fraud detection pipeline.
        Returns dict with flagged transactions, risk scores, and summary.
        """
        if tx_df.empty:
            return self._empty_result()

        logger.info(f"Running fraud detection on {len(tx_df)} transactions...")

        # 1. Anomaly detection with Isolation Forest
        anomaly_scores = self._isolation_forest_scores(tx_df)

        # 2. Heuristic flags
        heuristic_flags = self._apply_heuristics(tx_df)

        # 3. Wash trading detection
        wash_trading_flags = self._detect_wash_trading(tx_df)

        # 4. Rug pull indicators (if wallet data available)
        rug_pull_flags = self._detect_rug_pull_indicators(tx_df, wallet_df)

        # 5. Combined risk score
        risk_scores = self._compute_risk_scores(
            anomaly_scores, heuristic_flags, wash_trading_flags, rug_pull_flags
        )

        # Build result
        tx_df = tx_df.copy()
        tx_df["anomaly_score"] = anomaly_scores
        tx_df["risk_score"] = risk_scores
        tx_df["is_suspicious"] = risk_scores > 0.6
        tx_df["wash_trading_flag"] = wash_trading_flags
        tx_df["rug_pull_flag"] = rug_pull_flags

        flagged = tx_df[tx_df["is_suspicious"]].copy()

        result = {
            "total_analyzed": len(tx_df),
            "total_flagged": int(flagged["is_suspicious"].sum()),
            "flagged_pct": round(float(flagged["is_suspicious"].sum() / len(tx_df) * 100), 2),
            "wash_trading_detected": int(wash_trading_flags.sum()),
            "rug_pull_alerts": int(rug_pull_flags.sum()),
            "avg_risk_score": round(float(risk_scores.mean()), 4),
            "risk_distribution": {
                "low": int((risk_scores <= 0.3).sum()),
                "medium": int(((risk_scores > 0.3) & (risk_scores <= 0.6)).sum()),
                "high": int(((risk_scores > 0.6) & (risk_scores <= 0.8)).sum()),
                "critical": int((risk_scores > 0.8).sum()),
            },
            "flagged_transactions": self._flagged_to_records(flagged),
            "all_risk_scores": risk_scores.tolist(),
        }

        logger.info(f"Fraud detection complete: {result['total_flagged']} flagged ({result['flagged_pct']}%)")
        return result

    # ── Isolation Forest ─────────────────────────────────────────────

    def _isolation_forest_scores(self, tx_df: pd.DataFrame) -> np.ndarray:
        """Run Isolation Forest and return anomaly scores (0=normal, 1=anomalous)."""
        features = [f for f in self.FRAUD_FEATURES if f in tx_df.columns]
        if not features:
            return np.zeros(len(tx_df))

        X = tx_df[features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        self.iso_forest.fit(X_scaled)
        self._is_fitted = True

        # score_samples returns negative scores; more negative = more anomalous
        raw_scores = self.iso_forest.score_samples(X_scaled)
        # Normalize to 0-1 where 1 is most anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            normalized = np.zeros(len(raw_scores))

        return normalized

    # ── Heuristic Rules ──────────────────────────────────────────────

    def _apply_heuristics(self, tx_df: pd.DataFrame) -> pd.Series:
        """Apply rule-based heuristics for suspicious patterns."""
        flags = pd.Series(0.0, index=tx_df.index)

        # High gas with low value (possible contract manipulation)
        if "gas_used" in tx_df.columns and "total_amount" in tx_df.columns:
            gas_mean = tx_df["gas_used"].mean()
            gas_std = tx_df["gas_used"].std() or 1
            high_gas = (tx_df["gas_used"] - gas_mean) / gas_std > 2
            low_value = tx_df["total_amount"] == 0
            flags += (high_gas & low_value).astype(float) * 0.3

        # Self-transfers
        if "is_self_transfer" in tx_df.columns:
            flags += tx_df["is_self_transfer"].astype(float) * 0.2

        # Excessive object deletion (possible asset draining)
        if "deleted_objects" in tx_df.columns:
            flags += (tx_df["deleted_objects"] > 2).astype(float) * 0.3

        # Many events (complex operations, possible exploit)
        if "num_events" in tx_df.columns:
            event_threshold = tx_df["num_events"].quantile(0.95)
            flags += (tx_df["num_events"] > event_threshold).astype(float) * 0.2

        return flags.clip(0, 1)

    # ── Wash Trading Detection ───────────────────────────────────────

    def _detect_wash_trading(self, tx_df: pd.DataFrame) -> pd.Series:
        """Detect circular/wash trading patterns (A→B→A)."""
        flags = pd.Series(False, index=tx_df.index)

        if "sender" not in tx_df.columns or "recipients" not in tx_df.columns:
            return flags

        # Build transfer graph: track A→B pairs
        transfer_pairs = {}
        for idx, row in tx_df.iterrows():
            sender = row["sender"]
            for recipient in row.get("recipients", []):
                pair = (sender, recipient)
                reverse = (recipient, sender)
                if reverse in transfer_pairs:
                    # Circular transfer detected
                    flags.at[idx] = True
                    flags.at[transfer_pairs[reverse]] = True
                transfer_pairs[pair] = idx

        logger.info(f"Wash trading: {flags.sum()} circular transfers detected")
        return flags

    # ── Rug Pull Indicators ──────────────────────────────────────────

    def _detect_rug_pull_indicators(self, tx_df: pd.DataFrame, wallet_df: pd.DataFrame = None) -> pd.Series:
        """Score transactions for rug pull potential."""
        flags = pd.Series(False, index=tx_df.index)

        if wallet_df is None or wallet_df.empty:
            return flags

        # Flag wallets with very high sent-to-received ratio (draining)
        drainers = set()
        if "sent_received_ratio" in wallet_df.columns:
            high_ratio = wallet_df[wallet_df["sent_received_ratio"] > 5]
            drainers = set(high_ratio["address"].values)

        # Flag wallets with high self-transfer ratio
        self_transferers = set()
        if "self_transfer_ratio" in wallet_df.columns:
            high_self = wallet_df[wallet_df["self_transfer_ratio"] > 0.5]
            self_transferers = set(high_self["address"].values)

        suspicious_addrs = drainers | self_transferers

        for idx, row in tx_df.iterrows():
            if row["sender"] in suspicious_addrs:
                flags.at[idx] = True

        logger.info(f"Rug pull indicators: {flags.sum()} flagged transactions")
        return flags

    # ── Risk Score Aggregation ───────────────────────────────────────

    def _compute_risk_scores(
        self,
        anomaly_scores: np.ndarray,
        heuristic_flags: pd.Series,
        wash_flags: pd.Series,
        rug_flags: pd.Series,
    ) -> np.ndarray:
        """Combine all signals into a single risk score [0, 1]."""
        combined = (
            0.35 * anomaly_scores +
            0.25 * heuristic_flags.values +
            0.20 * wash_flags.astype(float).values +
            0.20 * rug_flags.astype(float).values
        )
        return np.clip(combined, 0, 1)

    # ── Helpers ───────────────────────────────────────────────────────

    def _flagged_to_records(self, flagged_df: pd.DataFrame) -> List[Dict]:
        """Convert flagged transactions to serializable records."""
        records = []
        for _, row in flagged_df.iterrows():
            records.append({
                "digest": row.get("digest", ""),
                "sender": row.get("sender", ""),
                "risk_score": round(float(row.get("risk_score", 0)), 4),
                "anomaly_score": round(float(row.get("anomaly_score", 0)), 4),
                "total_amount": float(row.get("total_amount", 0)),
                "gas_used": float(row.get("gas_used", 0)),
                "wash_trading": bool(row.get("wash_trading_flag", False)),
                "rug_pull": bool(row.get("rug_pull_flag", False)),
                "is_self_transfer": bool(row.get("is_self_transfer", False)),
            })
        return sorted(records, key=lambda x: x["risk_score"], reverse=True)

    def _empty_result(self) -> Dict:
        return {
            "total_analyzed": 0, "total_flagged": 0, "flagged_pct": 0,
            "wash_trading_detected": 0, "rug_pull_alerts": 0, "avg_risk_score": 0,
            "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "flagged_transactions": [], "all_risk_scores": [],
        }
