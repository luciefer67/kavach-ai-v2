import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load("ml/datasets/processed/X_seq.npy")
y = np.load("ml/datasets/processed/y_seq.npy")

print("📦 Data Loaded")
print("X:", X.shape)
print("y:", y.shape)

# -----------------------------
# 🔧 FIX LABELS (VERY IMPORTANT)
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("✅ Labels normalized")
print("Classes:", label_encoder.classes_)
print("Unique y:", np.unique(y))

NUM_CLASSES = len(np.unique(y))

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64
)

# -----------------------------
# LSTM MODEL
# -----------------------------
class KavachLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last time-step
        return self.fc(out)

model = KavachLSTM(
    input_size=X.shape[2],
    hidden_size=64,
    num_classes=NUM_CLASSES
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("🧠 Model initialized on:", device)

# -----------------------------
# TRAINING SETUP
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# TRAIN LOOP
# -----------------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"🔥 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# -----------------------------
# SAVE MODEL + LABEL MAP
# -----------------------------
torch.save({
    "model_state": model.state_dict(),
    "label_encoder": label_encoder
}, "ml/kavach_lstm_v2.pth")

print("✅ Kavach AI v2 LSTM model saved successfully")
