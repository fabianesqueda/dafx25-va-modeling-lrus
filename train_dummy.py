import torch
import torch.nn as nn
import torch.optim as optim

from model import Model

# Device selection (MPS -> CUDA -> CPU)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Dummy Dataset Parameters
sample_rate   = 96000
num_sequences = 1000
seq_length    = 9600 # 100ms

# Dummy Training Parameters
batch_size = 64
epochs     = 10
learning_rate = 1e-3

# Dummy dataset: white noise -> tanh
X = torch.randn(num_sequences, seq_length, 1)
Y = torch.tanh(X)

dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
)

# Model
N = 16 # State size
H = 8  # Hidden size
D = 3  # Depth size

model = Model(  input_channels=1
              , output_channels=1
              , N=N
              , H=H
              , D=D).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    
    total_loss = 0.0
    for xb, yb in loader:

        # Move batch to the selected device
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        yb = yb.to(device, non_blocking=(device.type == "cuda"))

        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
