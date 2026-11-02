"""
Sui Blockchain Analytics — Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Sui RPC ──────────────────────────────────────────────────────────
SUI_RPC_URL = os.getenv("SUI_RPC_URL", "https://fullnode.testnet.sui.io")
SUI_RPC_HEADERS = {"Content-Type": "application/json"}

# ── Gemini AI ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# ── Data Collection ──────────────────────────────────────────────────
COLLECTION_BATCH_SIZE = 50          # transactions per RPC call
MAX_TRANSACTIONS = 500              # total transactions to collect
REQUEST_TIMEOUT = 30                # seconds
REQUEST_DELAY = 0.2                 # delay between RPC calls (rate limiting)
MAX_RETRIES = 3                     # retries for failed requests

# ── Feature Engineering ──────────────────────────────────────────────
MIN_TX_COUNT = 2                    # minimum txs for wallet profiling
ROLLING_WINDOW = 20                 # rolling window for time-series features

# ── Model Hyperparameters ────────────────────────────────────────────
ISOLATION_FOREST_CONTAMINATION = 0.1
ISOLATION_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_N_ESTIMATORS = 100
KMEANS_N_CLUSTERS = 5
ANOMALY_ZSCORE_THRESHOLD = 2.5

# ── Paths ────────────────────────────────────────────────────────────
DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DATA_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")

# Ensure directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
