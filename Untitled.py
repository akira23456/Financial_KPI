# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# --- Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from financial_KPI_cleaner import clean_financial_data

# --- Step 1: Clean raw data ---
input_csv = "combined_financial_statements.csv"
output_csv = "cleaned_financial_data.csv"
clean_financial_data(input_csv, output_csv)

# --- Step 2: Load cleaned data ---
df = pd.read_csv(output_csv)

# Features (X) and target (y)
X = df.drop(columns=["netIncome"])
X = X.select_dtypes(include=[np.number])
y = df["netIncome"]

# --- Step 2a: Scale inputs ---
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# --- Step 2b: Scale target ---
scale_factor = 1_000_000
y_scaled = y / scale_factor

# --- Step 3: PyTorch Dataset ---
class CSVDataset(Dataset):
    def __init__(self, X, y):
        # X: NumPy array (n_samples, n_features)
        # y: Pandas Series or NumPy array (n_samples,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32).squeeze()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CSVDataset(X_scaled, y_scaled)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Step 4: Define a simple NN ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNN(input_size=X_scaled.shape[1], hidden_size=64, output_size=1)

# --- Step 5: Train ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# --- Step 6: Generate predictions ---
with torch.no_grad():
    preds_scaled = model(torch.tensor(X_scaled, dtype=torch.float32)).squeeze()

# --- Step 6a: Rescale predictions ---
preds = preds_scaled * scale_factor

# --- Step 7: Save for Power BI ---
preds_df = pd.DataFrame({
    "Actual": y.values,
    "Predicted": preds.numpy()
})
preds_df.to_csv("predictions_for_powerbi.csv", index=False)

print("✅ Predictions saved to predictions_for_powerbi.csv")


# %%

# %%
