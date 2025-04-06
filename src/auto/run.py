## training loop for training tokenizer

import torch
from torch.nn import functional as F
from .model import DNA_VQVAE, ModelConfig
from .dataset import Dataset
import matplotlib.pyplot as plt

class TrainConfig:
  device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  learning_rate = 1e-4         # bumped from 1e-5
  weight_decay  = 1e-4
  amsgrad       = True
  warmup_epochs = 50           # linear warm‑up
  epochs        = 2000
  eval_interval = 100
  eval_iters    = 30
  batch_size    = 6
  block_size    = 256
loss_history  = []

# setup
_model = DNA_VQVAE(ModelConfig).to("cpu")
n_param = sum(p.numel() for p in _model.parameters())/1e6
print(f"{n_param:.2f} million")
optimizer = torch.optim.Adam(_model.parameters(), lr=TrainConfig.learning_rate, amsgrad=True, weight_decay=1e-5, betas=(0.9, 0.95))

# ------ Learning‑rate Schedulers ------
# 1) Warm‑up: linearly ramp LR from 0 → lr over warmup_epochs
warmup_scheduler = LambdaLR(
  optimizer,
  lr_lambda=lambda epoch: min((epoch+1)/TrainConfig.warmup_epochs, 1.0)
)
# 2) After warm‑up, cosine decay from lr → 0 over remaining epochs
cosine_scheduler = CosineAnnealingLR(
  optimizer,
  T_max=TrainConfig.epochs - TrainConfig.warmup_epochs,
  eta_min=1e-6
)

# train-test split
file_path = "/content/drive/MyDrive/dna_data.txt"
data = Dataset(file_path, ratio=0.2)
train_data, val_data = data.train_test_split()

torch.manual_seed(400)
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - TrainConfig.block_size, (TrainConfig.batch_size,))
  x = torch.stack([data[i:i+TrainConfig.block_size] for i in ix]).float()  # Convert to float
  y = torch.stack([data[i+1:i+TrainConfig.block_size+1] for i in ix]).float()  # Convert to float
  return x.to("cpu"), y.to("cpu")

@torch.no_grad()
def estimate_loss():
  out = {}
  _model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(TrainConfig.eval_iters)
    for k in range(TrainConfig.eval_iters):
      X, Y = get_batch(split)
      x_recon, vq_loss, _ = _model(X)
      recon_loss = F.cross_entropy(x_recon.view(-1, 4), Y.view(-1, 4))
      losses[k] = (recon_loss + vq_loss).item()
    out[split] = losses.mean()
  _model.train()
  return out

import timeit

start_time = timeit.default_timer()
for epoch in range(TrainConfig.epochs):
  xb, yb = get_batch('train')

  x_recon, vq_loss, _ = _model(xb)
  recon_ce  = F.cross_entropy(x_recon.view(-1,4), yb.view(-1,4))
  recon_mse = F.mse_loss(torch.softmax(x_recon, dim=-1), yb)
  recon_loss = recon_ce + 0.5*recon_mse

  optimizer.zero_grad()
  recon_loss.backward()
  # -- Gradient clipping --
  torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)

  optimizer.step()

  # -- Scheduler step --
  if epoch < TrainConfig.warmup_epochs:
    warmup_scheduler.step()
  else:
    cosine_scheduler.step()

  # -- Logging & eval --
  if (epoch+1) % TrainConfig.eval_interval == 0:
    losses = estimate_loss()
    print(f"Epoch {epoch+1:4d} | train {losses['train']:.4f}  val {losses['val']:.4f}")
    loss_history.append((epoch+1, losses['train'], losses['val']))

end_time = timeit.default_timer()
print(f"Total time taken: {(end_time - start_time) / 3600} hrs")

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

## training loop for embedder

import numpy as np
from .embedder import Evoformer, ModelConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

class TrainConfig:
  DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  LR           = 1e-4
  WD           = 1e-4
  AMS          = True
  WARMUP       = 50
  EPOCHS       = 500
  BATCH        = 8
  MSA_SEQ      = 32       # number of sequences in each MSA
  L_SEQ        = 256      # length of each sequence
  EVAL_ITERS   = 5
  EVAL_INTV    = 50

msa_data  = np.load("msa.npy")   # shape (D, N, L, A)
pair_data = np.load("pair.npy")  # shape (D, L, L, C)
assert msa_data.ndim==4 and pair_data.ndim==4

# ------ 3. Train/Val Split ------
D = msa_data.shape[0]
split = int(D * 0.85)
msa_train, msa_val   = msa_data[:split], msa_data[split:]
pair_train, pair_val = pair_data[:split], pair_data[split:]

# ------ 4. Model, Optimizer, Scheduler ------
model = Evoformer(ModelConfig).to(ModelConfig.DEVICE)
opt   = AdamW(model.parameters(), lr=TrainConfig.LR, weight_decay=TrainConfig.WD, amsgrad=TrainConfig.AMS)
warm  = LambdaLR(opt, lambda e: min((e+1)/TrainConfig.WARMUP, 1.0))
cos   = CosineAnnealingLR(opt, T_max=TrainConfig.EPOCHS-TrainConfig.WARMUP, eta_min=1e-6)

# ------ 5. Batch Sampler ------
def get_batch(split):
  if split=="train":
    msa, pair = msa_train, pair_train
  else:
    msa, pair = msa_val,   pair_val
  idx = np.random.randint(0, msa.shape[0], size=TrainConfig.BATCH)
  # each batch: (B, N, L, A) and (B, L, L, C)
  return (
    torch.tensor(msa[idx],  dtype=torch.float32, device=TrainConfig.DEVICE),
    torch.tensor(pair[idx], dtype=torch.float32, device=TrainConfig.DEVICE)
  )

# ------ 6. Eval Loss (masked‑token CE) ------
@torch.no_grad()
def estimate_loss():
  model.eval()
  out = {}
  for split in ("train","val"):
    losses = []
    for _ in range(TrainConfig.EVAL_ITERS):
      M, P = get_batch(split)
      logits, _ = model(M, P)
      # masked‑token: randomly mask 15% of msa positions
      mask = (torch.rand_like(logits[...,0]) < 0.15)
      target = M.argmax(-1)  # (B,N,L)
      logits = logits[mask]
      target = target[mask]
      losses.append(F.cross_entropy(logits, target).item())
    out[split] = sum(losses)/len(losses)
  model.train()
  return out

# ------ 7. Training Loop ------
history = []
for epoch in range(TrainConfig.EPOCHS):
  M, P = get_batch("train")
  opt.zero_grad()
  logits, _ = model(M, P)
  mask   = (torch.rand_like(logits[...,0]) < 0.15)
  target = M.argmax(-1)
  loss   = F.cross_entropy(logits[mask], target[mask])
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  opt.step()
  if epoch < TrainConfig.WARMUP: warm.step()
  else:            cos.step()

  if (epoch+1)%TrainConfig.EVAL_INTV==0:
    losses = estimate_loss()
    print(f"Epoch {epoch+1:4d} | train {losses['train']:.4f}  val {losses['val']:.4f}")
    history.append((epoch+1, losses['train'], losses['val']))

# ------ 8. Save & Plot ------
torch.save(model.state_dict(), "af_tokenizer.pt")
try:
  import matplotlib.pyplot as plt
  e,t,v = zip(*history)
  plt.plot(e,t,label="train"); plt.plot(e,v,label="val")
  plt.legend(); plt.show()
except:
  pass