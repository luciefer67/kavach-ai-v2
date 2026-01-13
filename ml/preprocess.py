import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# PATH SETUP (SAFE)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_CSV = os.path.join(
    BASE_DIR,
    "ml",
    "datasets",
    "raw",
    "Friday-WorkingHours-Afternoon-PortScan.csv"
)

print("🔍 Looking for CSV at:", RAW_CSV)

if not os.path.exists(RAW_CSV):
    raise FileNotFoundError(f"❌ CSV file not found at: {RAW_CSV}")

# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------
df = pd.read_csv(RAW_CSV)
df = df.fillna(0)

print("📌 CSV Columns detected:")
print(df.columns.tolist())

# --------------------------------------------------
# COLUMN AUTO-MAPPING (NO MORE KEYERROR)
# --------------------------------------------------
def pick_column(possible_names):
    for col in possible_names:
        if col in df.columns:
            return df[col]
    return pd.Series([0] * len(df))

FWD_PKTS = pick_column([
    "Total Fwd Packets", "Tot Fwd Pkts", "Fwd Packets", "Tot Fwd Pkts"
])

BWD_PKTS = pick_column([
    "Total Backward Packets", "Tot Bwd Pkts", "Bwd Packets", "Tot Bwd Pkts"
])

FLOW_PKTS_RATE = pick_column([
    "Flow Packets/s", "Flow Pkts/s"
])

FLOW_BYTES_RATE = pick_column([
    "Flow Bytes/s", "Flow Byts/s"
])

LABEL_COL = pick_column([
    "Label", "Attack", "Class"
])

# --------------------------------------------------
# FEATURE ENGINEERING (SAFE)
# --------------------------------------------------
df["packet_count"] = FWD_PKTS + BWD_PKTS
df["packet_rate"] = FLOW_PKTS_RATE
df["byte_rate"] = FLOW_BYTES_RATE

def label_to_int(label):
    label = str(label).lower()
    if label == "benign":
        return 0
    if "portscan" in label:
        return 1
    return 2  # other attacks

df["y"] = LABEL_COL.apply(label_to_int)

FEATURE_COLS = ["packet_count", "packet_rate", "byte_rate"]

# --------------------------------------------------
# SEQUENCE BUILDING (3D DATA)
# --------------------------------------------------
WINDOW_SIZE = 10
X_seq, y_seq = [], []

for i in range(len(df) - WINDOW_SIZE):
    window = df.iloc[i:i + WINDOW_SIZE]
    X_seq.append(window[FEATURE_COLS].values)
    y_seq.append(window["y"].iloc[-1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
OUT_DIR = os.path.join(BASE_DIR, "ml", "datasets", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

np.save(os.path.join(OUT_DIR, "X_seq.npy"), X_seq)
np.save(os.path.join(OUT_DIR, "y_seq.npy"), y_seq)

print("✅ Sequence dataset ready")
print("X shape:", X_seq.shape)
print("y shape:", y_seq.shape)
