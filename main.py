"""
Sui Blockchain Analytics — Main Pipeline
Orchestrates: Data Collection → Preprocessing → Analysis → Export
"""
import json
import os
import sys
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import DATA_RESULTS_DIR, MAX_TRANSACTIONS
from src.collector.sui_collector import SuiCollector
from src.preprocessing.preprocessor import DataPreprocessor
from src.analysis.fraud_detector import FraudDetector
from src.analysis.wallet_classifier import WalletClassifier
from src.analysis.anomaly_detector import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("SuiAnalytics")


def run_pipeline(max_transactions: int = None):
    """Run the full analytics pipeline."""
    max_transactions = max_transactions or MAX_TRANSACTIONS
    start_time = time.time()

    print("\n" + "=" * 70)
    print("  🔍  SUI BLOCKCHAIN ANALYTICS PIPELINE")
    print("=" * 70)

    # ── 1. Data Collection ───────────────────────────────────────────
    print("\n📡 [1/5] Collecting on-chain data from Sui testnet...")
    collector = SuiCollector()

    # Verify connection
    checkpoint = collector.get_latest_checkpoint()
    if checkpoint is None:
        logger.error("Failed to connect to Sui RPC. Check your connection.")
        return
    print(f"    ✓ Connected. Latest checkpoint: {checkpoint}")

    raw_txs = collector.collect_and_save(max_count=max_transactions)
    if not raw_txs:
        logger.error("No transactions collected. Exiting.")
        return
    print(f"    ✓ Collected {len(raw_txs)} transactions")

    # ── 2. Preprocessing ─────────────────────────────────────────────
    print("\n⚙️  [2/5] Preprocessing & feature engineering...")
    preprocessor = DataPreprocessor()

    tx_df = preprocessor.process_transactions(raw_txs)
    print(f"    ✓ Processed {len(tx_df)} transactions into DataFrame")

    wallet_df = preprocessor.build_wallet_profiles(tx_df)
    print(f"    ✓ Built profiles for {len(wallet_df)} wallets")

    token_df = preprocessor.build_token_metrics(tx_df)
    print(f"    ✓ Built metrics for {len(token_df)} token types")

    # ── 3. Fraud Detection ───────────────────────────────────────────
    print("\n🚨 [3/5] Running fraud detection...")
    fraud_detector = FraudDetector()
    fraud_results = fraud_detector.analyze(tx_df, wallet_df)
    print(f"    ✓ Flagged: {fraud_results['total_flagged']}/{fraud_results['total_analyzed']} transactions")
    print(f"    ✓ Wash trading: {fraud_results['wash_trading_detected']} detected")
    print(f"    ✓ Rug pull alerts: {fraud_results['rug_pull_alerts']}")
    print(f"    ✓ Risk distribution: {fraud_results['risk_distribution']}")

    # ── 4. Wallet Classification ─────────────────────────────────────
    print("\n👛 [4/5] Classifying wallets...")
    wallet_classifier = WalletClassifier()
    wallet_results = wallet_classifier.analyze(wallet_df)
    print(f"    ✓ Classified {wallet_results['total_wallets']} wallets")
    print(f"    ✓ Distribution: {wallet_results['type_distribution']}")

    # ── 5. Anomaly Detection ─────────────────────────────────────────
    print("\n📊 [5/5] Detecting anomalies...")
    anomaly_detector = AnomalyDetector()
    anomaly_results = anomaly_detector.analyze(tx_df, token_df)
    print(f"    ✓ Volume anomalies: {anomaly_results['volume_anomalies']}")
    print(f"    ✓ Token anomalies: {anomaly_results['token_anomalies']}")
    print(f"    ✓ Flash patterns: {anomaly_results['flash_patterns_detected']}")
    print(f"    ✓ Severity: {anomaly_results['summary']['severity']}")

    # ── Export Results ────────────────────────────────────────────────
    print("\n💾 Exporting results...")
    _export_results(fraud_results, wallet_results, anomaly_results, tx_df, wallet_df)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  ✅  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  📁  Results saved to: {DATA_RESULTS_DIR}")
    print(f"  🌐  Open dashboard/index.html to view results")
    print(f"{'=' * 70}\n")


def _sanitize(obj):
    """Recursively replace NaN/Infinity with JSON-safe values."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _export_results(fraud, wallets, anomalies, tx_df, wallet_df):
    """Save all results as JSON for the dashboard."""

    # Combined dashboard data
    dashboard_data = _sanitize({
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_transactions": len(tx_df),
            "total_wallets": len(wallet_df),
        },
        "fraud_detection": fraud,
        "wallet_classification": wallets,
        "anomaly_detection": anomalies,
    })

    filepath = os.path.join(DATA_RESULTS_DIR, "dashboard_data.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    print(f"    ✓ Dashboard data → {filepath}")

    # Individual result files
    for name, data in [("fraud_results.json", fraud),
                        ("wallet_results.json", wallets),
                        ("anomaly_results.json", anomalies)]:
        path = os.path.join(DATA_RESULTS_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(data), f, indent=2, default=str)
        print(f"    ✓ {name} → {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sui Blockchain Analytics Pipeline")
    parser.add_argument("-n", "--num-transactions", type=int, default=None,
                        help=f"Number of transactions to collect (default: {MAX_TRANSACTIONS})")
    args = parser.parse_args()
    run_pipeline(max_transactions=args.num_transactions)
