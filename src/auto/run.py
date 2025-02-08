import torch
from torch.nn import functional as F
from .model import DNA_VQVAE, ModelConfig
from .dataset import dna_to_onehot, fix_dna_lines
import matplotlib.pyplot as plt

# setup
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
_model = DNA_VQVAE(ModelConfig).to("cpu")
n_param = sum(p.numel() for p in _model.parameters())/1e6
print(f"{n_param:.2f} million")
optimizer = torch.optim.Adam(_model.parameters(), lr=1e-5)

# train-test split
file_path = "/content/drive/MyDrive/dna_data.txt"
dataset = fix_dna_lines(file_path)
train_size = int(0.8 * len(dataset))
train_data, val_data = dataset[:train_size], dataset[train_size:]

epochs = 2000
loss_history = []
batch_size = 16
block_size = 128
eval_interval = 10
eval_iters = 5
learning_rate = 1e-5

## batch training
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([dna_to_onehot(data[i:i+block_size]) for i in ix])  # Convert to one-hot
  y = torch.stack([dna_to_onehot(data[i+1:i+block_size+1]) for i in ix])  # Shifted target
  return x.to("cpu"), y.to("cpu")

@torch.no_grad()
def estimate_loss():
  out = {}
  _model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      x_recon, vq_loss, _ = _model(X)
      recon_loss = F.cross_entropy(x_recon.view(-1, 4), Y.view(-1, 4))
      losses[k] = (recon_loss + vq_loss).item()
    out[split] = losses.mean()
  _model.train()
  return out

for epoch in range(epochs):
  xb, yb = get_batch('train')

  optimizer.zero_grad()
  x_recon, vq_loss, _ = _model(xb)
  recon_loss = F.cross_entropy(x_recon.view(-1, 4), yb.view(-1, 4))
  loss = recon_loss + vq_loss
  loss.backward()
  optimizer.step()

  if (epoch + 1) % eval_interval == 0:
    losses = estimate_loss()
    print(f"Epoch {epoch+1}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f}")
    loss_history.append((epoch + 1, losses['train'], losses['val']))

torch.save(_model.state_dict(), f'biosaic_{n_param:.0f}m.pth')

import matplotlib.pyplot as plt

epochs_logged, train_losses, val_losses = zip(*loss_history)
plt.figure(figsize=(8, 5))
plt.plot(epochs_logged, train_losses, label="Train Loss", marker='o', linestyle='-')
plt.plot(epochs_logged, val_losses, label="Val Loss", marker='o', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss Over Time")
plt.grid(True)
plt.show()