# This code contains the creation of the architecture, the training and testing of the human-based neural network.

import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Folder location
humans_dir = os.path.join(os.path.dirname(__file__), 'hotspot_people_loc')

# Load human-only data
human_inputs = []
human_targets = []

for filename in os.listdir(humans_dir):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(humans_dir, filename)
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
                        entry[0].lower() == 'person'):
                    human_inputs.append(entry[1])
                    human_targets.append(entry[2])
            except Exception as e:
                print(f"Skipping malformed line in {filename}: {line}\n{e}")

# Convert and normalize
inputs_tensor = torch.tensor(human_inputs, dtype=torch.float32)
targets_tensor = torch.tensor(human_targets, dtype=torch.float32)

normalized_factor = 640.0
inputs_tensor = inputs_tensor / normalized_factor

# Train/val/test split
class HumanDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = HumanDataset(inputs_tensor, targets_tensor)
full_indices = list(range(len(dataset)))

train_val_indices, test_indices = train_test_split(full_indices, test_size=0.05, random_state=99)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2105, random_state=2)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# NN
class HumanNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.sigmoid(self.output_layer(x))  # constrain to [0, 1]
        x_scaled = torch.stack([x[:, 0] * 14, x[:, 1] * 8], dim=1)  # scale to [0, 14] and [0, 8]
        return x_scaled

model = HumanNN()

# Training
loss_fn = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 60

train_losses = []
val_losses = []

# Loop for training
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    total_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss += loss_fn(val_outputs, val_targets).item()
    val_loss /= len(val_loader)

    train_losses.append(total_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

# Evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        test_loss += loss_fn(test_outputs, test_targets).item()
test_loss /= len(test_loader)
print(f"\nFinal Test Loss: {test_loss:.4f}")

# Example
example_input = (inputs_tensor[0].unsqueeze(0))  # shape [1, 4]
predicted_layout = model(example_input)

print("\nExample Human Input BBox (normalized):", inputs_tensor[0].numpy())
print("Predicted Layout:", predicted_layout.detach().numpy())
print("Actual Layout:", human_targets[0])

# Save
model_save_path = os.path.join(os.path.dirname(__file__), "human_nn_model.pth")
torch.save({'model_state_dict': model.state_dict()}, model_save_path)
print(f"Model saved to: {model_save_path}")

# trainlLoss/val plots
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# testrun
nn_input = torch.tensor([[394.4772, 231.198, 563.3987, 481.9715]], dtype=torch.float32)
nn_input = nn_input / normalized_factor
prediction = model(nn_input)
print("\nManual input prediction (layout):", prediction)
