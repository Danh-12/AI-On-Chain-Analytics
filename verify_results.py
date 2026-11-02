import json

d = json.load(open("data/results/dashboard_data.json", encoding="utf-8"))
m = d["metadata"]
f = d["fraud_detection"]
w = d["wallet_classification"]
a = d["anomaly_detection"]

print(f"=== RESULTS SUMMARY ===")
print(f"Transactions: {m['total_transactions']}")
print(f"Wallets: {m['total_wallets']}")
print(f"Generated: {m['generated_at']}")
print()
print(f"--- Fraud Detection ---")
print(f"Flagged: {f['total_flagged']} / {f['total_analyzed']} ({f['flagged_pct']}%)")
print(f"Wash trading: {f['wash_trading_detected']}")
print(f"Rug pull alerts: {f['rug_pull_alerts']}")
print(f"Risk dist: {f['risk_distribution']}")
print(f"Top flagged: {len(f['flagged_transactions'])} transactions")
print()
print(f"--- Wallet Classification ---")
print(f"Total: {w['total_wallets']}")
print(f"Types: {w['type_distribution']}")
print(f"Top features: {dict(list(w['feature_importances'].items())[:5])}")
print()
print(f"--- Anomaly Detection ---")
print(f"Volume anomalies: {a['volume_anomalies']}")
print(f"Token anomalies: {a['token_anomalies']}")
print(f"Flash patterns: {a['flash_patterns_detected']}")
print(f"Severity: {a['summary']['severity']}")
