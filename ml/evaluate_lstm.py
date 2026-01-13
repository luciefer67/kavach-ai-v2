import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
X = np.load("ml/datasets/processed/X_seq.npy")
y = np.load("ml/datasets/processed/y_seq.npy")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Load model
checkpoint = torch.load("ml/kavach_lstm_v2.pth", map_location="cpu")
model_state = checkpoint["model_state"]
label_encoder = checkpoint["label_encoder"]

import torch.nn as nn

class KavachLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = KavachLSTM(X.shape[2], 64, len(np.unique(y)))
model.load_state_dict(model_state)
model.eval()

# Predict
with torch.no_grad():
    preds = model(torch.tensor(X, dtype=torch.float32))
    y_pred = torch.argmax(preds, dim=1).numpy()

print("✅ Accuracy:", accuracy_score(y, y_pred))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\n📄 Classification Report:")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))
