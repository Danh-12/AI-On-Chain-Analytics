# 🔍 Sui Blockchain Analytics Platform

**Hackathon 2026 — On-chain Intelligence for Sui**

A Python-based analytics platform that collects on-chain data from the Sui blockchain, engineers features, and applies machine learning to detect fraud, classify wallets, and identify anomalies.

## 🏗️ Architecture

```
sui_block_chain/
├── config/settings.py          # Configuration + environment variables
├── src/
│   ├── collector/sui_collector.py    # Sui JSON-RPC data collection
│   ├── preprocessing/preprocessor.py # Feature engineering
│   └── analysis/
│       ├── fraud_detector.py         # Fraud & scam detection
│       ├── wallet_classifier.py      # Wallet behavior classification
│       └── anomaly_detector.py       # DeFi anomaly detection
├── dashboard/index.html        # Interactive web dashboard + AI chatbot
├── main.py                     # Pipeline orchestrator
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the analytics pipeline
python main.py

# 3. Open the dashboard
# Open dashboard/index.html in your browser
```

## 🧠 Analysis Modules

### 🚨 Fraud Detection
- **Isolation Forest** for unsupervised anomaly scoring
- **Wash trading detection** via circular transfer graph analysis (A→B→A)
- **Rug pull indicators**: liquidity draining, self-transfer ratios, creator concentration
- **Heuristic rules**: high gas / low value, excessive object deletion, event spikes

### 👛 Wallet Classification
- **KMeans clustering** to discover natural wallet groups
- **Heuristic label assignment**: Bot, Whale, Retail, DEX/Contract, Suspicious
- **Random Forest** for feature importance ranking
- Features: tx frequency, timing patterns, value distributions, counterparty analysis

### 📊 Anomaly Detection
- **Rolling Z-score analysis** for volume time-series
- **Gini coefficient** for token concentration measurement
- **Flash activity detection**: burst of transactions in short windows
- **Isolation Forest** on transaction-level features

## 🤖 AI Chatbot
The dashboard includes a Gemini-powered AI assistant that can answer questions about the analytics results:
- *"Which wallets are flagged as suspicious?"*
- *"How many wash trading patterns were detected?"*
- *"Explain the anomalies found"*

## 🛠️ Technologies
- **Python**: Pandas, NumPy, Scikit-learn
- **Sui SDK**: JSON-RPC API (`https://fullnode.testnet.sui.io`)
- **ML**: Isolation Forest, Random Forest, KMeans, StandardScaler
- **Frontend**: HTML5, CSS3, Chart.js, Gemini AI API
- **Design**: Dark theme, glassmorphism, responsive layout

## 📈 Configuration

Edit `.env` for API keys and `config/settings.py` for parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_TRANSACTIONS` | 500 | Transactions to collect |
| `ISOLATION_FOREST_CONTAMINATION` | 0.1 | Anomaly proportion |
| `KMEANS_N_CLUSTERS` | 5 | Wallet classification groups |
| `ANOMALY_ZSCORE_THRESHOLD` | 2.5 | Volume spike sensitivity |
