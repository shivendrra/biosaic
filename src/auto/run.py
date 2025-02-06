import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from .model import DNA_VQVAE, ModelConfig
from .dataset import DNADataset, pad_collate_fn, dna_to_onehot
import matplotlib.pyplot as plt

# setup
device = "cuda" if torch.cuda.is_available else "cpu"
model = DNA_VQVAE(ModelConfig).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
dataset = DNADataset("file.txt")

# train-test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
epochs = 2000
loss_history = []

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate_fn)

for epoch in range(epochs):
  total_loss = 0
  for dna_seq in train_loader:
    optimizer.zero_grad()
    dna_seq = dna_seq.to(device)
    x_recon, vq_loss, _ = model(dna_seq)
    
    recon_loss = F.cross_entropy(x_recon.view(-1, 4), dna_seq.view(-1, 4))
    loss = recon_loss + vq_loss
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    if (epoch + 1) % 100 == 0:
      avg_loss = total_loss / len(train_loader)
      loss_history.append((epoch + 1, avg_loss))
      print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

epochs_logged, losses = zip(*loss_history)
plt.figure(figsize=(8, 5))
plt.plot(epochs_logged, losses, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()