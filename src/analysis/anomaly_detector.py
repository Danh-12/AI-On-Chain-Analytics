"""
DeFi Protocol Anomaly Detector
Detects unusual patterns: volume spikes, liquidity changes, token concentration anomalies.
Uses statistical methods (Z-scores, rolling stats) + Isolation Forest.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    ANOMALY_ZSCORE_THRESHOLD,
    ROLLING_WINDOW,
)

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in Sui DeFi protocol activity.
    - Time-series volume analysis with rolling Z-scores
    - Token concentration anomalies (Gini coefficient spikes)
    - Flash activity patterns (burst detection)
    - Isolation Forest on aggregated features
    """

    def __init__(self):
        self.iso_forest = IsolationForest(
            contamination=ISOLATION_FOREST_CONTAMINATION,
            n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    # ── Main Analysis ────────────────────────────────────────────────

    def analyze(self, tx_df: pd.DataFrame, token_df: pd.DataFrame = None) -> Dict:
        """
        Run full anomaly detection pipeline.
        Returns dict with anomalies, time-series data, and summary stats.
        """
        if tx_df.empty:
            return self._empty_result()

        logger.info(f"Running anomaly detection on {len(tx_df)} transactions...")

        # 1. Time-series volume analysis
        volume_analysis = self._analyze_volume_timeseries(tx_df)

        # 2. Token concentration anomalies
        token_anomalies = self._analyze_token_concentration(token_df)

        # 3. Flash activity detection
        flash_patterns = self._detect_flash_activity(tx_df)

        # 4. Isolation Forest on tx-level features
        tx_anomaly_scores = self._isolation_forest_anomalies(tx_df)

        # Build result
        total_volume_anomalies = len(volume_analysis.get("anomaly_points", []))
        total_token_anomalies = len(token_anomalies.get("anomalous_tokens", []))
        total_flash = len(flash_patterns.get("flash_windows", []))

        result = {
            "total_analyzed": len(tx_df),
            "volume_anomalies": total_volume_anomalies,
            "token_anomalies": total_token_anomalies,
            "flash_patterns_detected": total_flash,
            "isolation_forest_outliers": int((tx_anomaly_scores > 0.7).sum()),
            "volume_timeseries": volume_analysis,
            "token_analysis": token_anomalies,
            "flash_activity": flash_patterns,
            "anomaly_scores": tx_anomaly_scores.tolist(),
            "summary": {
                "total_anomalies": total_volume_anomalies + total_token_anomalies + total_flash,
                "severity": self._compute_severity(
                    total_volume_anomalies, total_token_anomalies, total_flash, len(tx_df)
                ),
            },
        }

        logger.info(f"Anomaly detection complete: {result['summary']['total_anomalies']} anomalies found")
        return result

    # ── Time-Series Volume Analysis ──────────────────────────────────

    def _analyze_volume_timeseries(self, tx_df: pd.DataFrame) -> Dict:
        """Analyze transaction volume over time with rolling Z-scores."""
        if "timestamp" not in tx_df.columns or tx_df["timestamp"].isna().all():
            return {"timeseries": [], "anomaly_points": [], "stats": {}}

        df = tx_df.copy()
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        # Resample into time buckets
        df = df.set_index("timestamp")
        freq = self._determine_frequency(df)

        volume_series = df["total_amount"].resample(freq).sum().fillna(0)
        count_series = df["total_amount"].resample(freq).count().fillna(0)
        gas_series = df["gas_used"].resample(freq).sum().fillna(0) if "gas_used" in df.columns else pd.Series()

        # Rolling statistics
        window = min(ROLLING_WINDOW, max(3, len(volume_series) // 3))
        rolling_mean = volume_series.rolling(window=window, min_periods=1).mean()
        rolling_std = volume_series.rolling(window=window, min_periods=1).std().fillna(1)

        # Z-scores
        z_scores = ((volume_series - rolling_mean) / rolling_std.replace(0, 1)).fillna(0)

        # Detect anomalies
        anomaly_mask = z_scores.abs() > ANOMALY_ZSCORE_THRESHOLD
        anomaly_points = []
        for ts, z in z_scores[anomaly_mask].items():
            anomaly_points.append({
                "timestamp": ts.isoformat(),
                "z_score": round(float(z), 4),
                "volume": float(volume_series.loc[ts]),
                "expected_volume": round(float(rolling_mean.loc[ts]), 2),
                "type": "spike" if z > 0 else "drop",
            })

        # Build timeseries data for charting
        timeseries = []
        for ts in volume_series.index:
            timeseries.append({
                "timestamp": ts.isoformat(),
                "volume": float(volume_series.loc[ts]),
                "tx_count": int(count_series.loc[ts]) if ts in count_series.index else 0,
                "rolling_mean": round(float(rolling_mean.loc[ts]), 2),
                "z_score": round(float(z_scores.loc[ts]), 4),
                "is_anomaly": bool(anomaly_mask.loc[ts]),
            })

        return {
            "timeseries": timeseries,
            "anomaly_points": anomaly_points,
            "stats": {
                "mean_volume": round(float(volume_series.mean()), 2),
                "std_volume": round(float(volume_series.std()), 2),
                "max_volume": round(float(volume_series.max()), 2),
                "total_periods": len(volume_series),
                "anomalous_periods": len(anomaly_points),
                "frequency": freq,
            },
        }

    def _determine_frequency(self, df: pd.DataFrame) -> str:
        """Determine the best time bucket frequency based on data span."""
        if df.index.empty:
            return "1h"
        span = (df.index.max() - df.index.min()).total_seconds()
        if span < 3600:       return "1min"
        elif span < 86400:    return "5min"
        elif span < 604800:   return "1h"
        else:                 return "1D"

    # ── Token Concentration Analysis ─────────────────────────────────

    def _analyze_token_concentration(self, token_df: pd.DataFrame) -> Dict:
        """Detect tokens with unusual concentration or activity patterns."""
        if token_df is None or token_df.empty:
            return {"anomalous_tokens": [], "all_tokens": []}

        anomalous = []
        all_tokens = []

        for _, row in token_df.iterrows():
            token_info = {
                "token_type": row.get("token_type", ""),
                "transfer_count": int(row.get("transfer_count", 0)),
                "total_volume": float(row.get("total_volume", 0)),
                "unique_senders": int(row.get("unique_senders", 0)),
                "unique_receivers": int(row.get("unique_receivers", 0)),
                "gini_coefficient": round(float(row.get("gini_coefficient", 0)), 4),
                "sender_receiver_ratio": round(float(row.get("sender_receiver_ratio", 0)), 4),
            }
            all_tokens.append(token_info)

            # Flag anomalous tokens
            is_anomalous = False
            reasons = []

            # High concentration (few senders, many transfers)
            if row.get("gini_coefficient", 0) > 0.7:
                is_anomalous = True
                reasons.append("high_concentration")

            # Sender/receiver imbalance
            if row.get("sender_receiver_ratio", 1) > 3:
                is_anomalous = True
                reasons.append("sender_receiver_imbalance")

            # Very high volume per sender
            if row.get("volume_per_sender", 0) > token_df["volume_per_sender"].quantile(0.95):
                is_anomalous = True
                reasons.append("high_volume_per_sender")

            if is_anomalous:
                token_info["reasons"] = reasons
                anomalous.append(token_info)

        return {
            "anomalous_tokens": anomalous,
            "all_tokens": all_tokens,
        }

    # ── Flash Activity Detection ─────────────────────────────────────

    def _detect_flash_activity(self, tx_df: pd.DataFrame) -> Dict:
        """Detect bursts of activity in short time windows."""
        if "timestamp" not in tx_df.columns or tx_df["timestamp"].isna().all():
            return {"flash_windows": []}

        df = tx_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(df) < 5:
            return {"flash_windows": []}

        flash_windows = []

        # Sliding window: look for windows with unusually many transactions
        window_size = pd.Timedelta(minutes=5)
        threshold = max(10, len(df) * 0.1)  # 10% of all txs in 5-min window

        for i in range(len(df)):
            window_end = df.iloc[i]["timestamp"] + window_size
            window_txs = df[(df["timestamp"] >= df.iloc[i]["timestamp"]) & (df["timestamp"] <= window_end)]

            if len(window_txs) >= threshold:
                # Check if we already captured this window
                window_start = df.iloc[i]["timestamp"]
                if flash_windows and pd.Timestamp(flash_windows[-1]["start"]) >= window_start - window_size:
                    continue

                unique_senders = window_txs["sender"].nunique()
                total_volume = window_txs["total_amount"].sum()

                flash_windows.append({
                    "start": window_start.isoformat(),
                    "end": window_end.isoformat(),
                    "tx_count": len(window_txs),
                    "unique_senders": int(unique_senders),
                    "total_volume": float(total_volume),
                    "is_single_sender": unique_senders == 1,
                })

        return {"flash_windows": flash_windows}

    # ── Isolation Forest on Transactions ─────────────────────────────

    def _isolation_forest_anomalies(self, tx_df: pd.DataFrame) -> np.ndarray:
        """Run Isolation Forest on transaction-level features."""
        features = ["gas_used", "total_amount", "num_recipients",
                     "num_events", "created_objects", "num_token_types"]
        available = [f for f in features if f in tx_df.columns]

        if not available:
            return np.zeros(len(tx_df))

        X = tx_df[available].fillna(0).values
        if len(X) < 10:
            return np.zeros(len(tx_df))

        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)

        raw_scores = self.iso_forest.score_samples(X_scaled)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            normalized = np.zeros(len(raw_scores))

        return normalized

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_severity(vol_anomalies: int, token_anomalies: int, flash: int, total: int) -> str:
        """Compute overall severity level."""
        total_anomalies = vol_anomalies + token_anomalies + flash
        ratio = total_anomalies / max(total, 1)
        if ratio > 0.2:
            return "CRITICAL"
        elif ratio > 0.1:
            return "HIGH"
        elif ratio > 0.05:
            return "MEDIUM"
        else:
            return "LOW"

    def _empty_result(self) -> Dict:
        return {
            "total_analyzed": 0, "volume_anomalies": 0,
            "token_anomalies": 0, "flash_patterns_detected": 0,
            "isolation_forest_outliers": 0,
            "volume_timeseries": {"timeseries": [], "anomaly_points": [], "stats": {}},
            "token_analysis": {"anomalous_tokens": [], "all_tokens": []},
            "flash_activity": {"flash_windows": []},
            "anomaly_scores": [],
            "summary": {"total_anomalies": 0, "severity": "LOW"},
        }
