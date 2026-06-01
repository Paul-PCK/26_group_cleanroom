# This code contains the creation of the architecture, the training and testing of the non-human based neural network.

import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# Configuration
machines_dir = os.path.join(os.path.dirname(__file__), 'hotspot_machines')

# Load machine-only data
machine_inputs = []
machine_targets = []

for filename in os.listdir(machines_dir):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(machines_dir, filename)
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ast.literal_eval(line)
                if (isinstance(entry, list) and len(entry) == 3 and
                        isinstance(entry[1], list) and len(entry[1]) == 4 and
                        isinstance(entry[2], list) and len(entry[2]) == 2 and
                        entry[0].lower() != 'person'):
                    machine_inputs.append(entry[1])
                    machine_targets.append(entry[2])
            except Exception as e:
                print(f"Skipping malformed line in {filename}: {line}\n{e}")

# Convert to tensor
inputs_tensor = torch.tensor(machine_inputs, dtype=torch.float32)
targets_tensor = torch.tensor(machine_targets, dtype=torch.float32)

normalized_factor = 640.0 #i.e. the max pixel value in each normalized image
inputs_tensor = inputs_tensor / normalized_factor

# Dataset and DataLoaders
class MachineDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = MachineDataset(inputs_tensor, targets_tensor)

N = len(dataset)

# Split into train+val and test
trainval_idx, test_idx = train_test_split(range(N), test_size=0.05, random_state=99)
# Then split trainval into train and val
train_idx, val_idx = train_test_split(trainval_idx, test_size=0.20, random_state=2)  # 0.20 of 95% ≈ 19%

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define NN
class MachineNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

model = MachineNN()

# Training
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 80

train_losses = []
val_losses = []

# Loop for training
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    total_train_loss /= len(train_loader)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            total_val_loss += loss_fn(val_outputs, val_targets).item()
    total_val_loss /= len(val_loader)

    train_losses.append(total_train_loss)
    val_losses.append(total_val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")

# Evaluation
model.eval()
total_test_loss = 0.0
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        total_test_loss += loss_fn(test_outputs, test_targets).item()
total_test_loss /= len(test_loader)

print(f"\nFinal Test Loss: {total_test_loss:.4f}")

# Example
example_input = inputs_tensor[0].unsqueeze(0)
predicted_layout = model(example_input)

print("\nExample Machine Input BBox (normalized):", example_input)
print("Predicted Layout:", predicted_layout.detach().numpy())
print("Actual Layout:", machine_targets[0])

# Save
model_save_path = os.path.join(os.path.dirname(__file__), "hotspot_machine_nn_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Machine model saved to: {model_save_path}")

# Plot train/val losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MachineNN Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
