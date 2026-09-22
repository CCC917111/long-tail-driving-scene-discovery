#!/usr/bin/env python3

# --- Role in this repository -----------------------------------------------
# This is the FINAL method reported in the project report: AbsTopK sparsity
# (Eq. 6-8) + the tail-guided training objective with a tail-activation reward
# term (Eq. 11-12). Evaluated on Cosmos-Reason1-7B layer-27/28 last-token
# features, this configuration produced the paper's headline result:
#   long-tail Precision=0.9023, Recall=0.7984, F1=0.8472 (layer 28, best),
#   vs. Precision=0.1429/Recall=0.0130/F1=0.0238 for the raw Cosmos-Reason1-7B
#   baseline on the same evaluation set. See ../README.md#results.
# The layer path below (layer27_...) reflects one specific extraction run;
# point FEATURES_NPY / META_JSON / JSON_PARTS at your own extracted-feature
# directory and change the layer index to sweep layers.
# ------------------------------------------------------------------------------

"""
LongTail‑Guided SAE with AbsTop‑K sparsity and Tail Activation Reward.
- Top-K selection based on absolute values (abstopk).
- Tail reward (encourages z_t activation for long-tail samples, with a maximum cap).
- Loss breakdown printed every 20 epochs.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ==================== Configuration ====================
FEATURES_NPY = "/123090047/cosmos_extract_output/layer27_mlp_output_last_token.npy"
META_JSON = "/123090047/cosmos_extract_output/meta.json"
JSON_PARTS = [
    "/123090047/nuscene_part04_qwen397b.json",
    "/123090047/nuscene_part05_qwen397b.json",
    "/123090047/nuscene_part06_qwen397b.json"
]
OUTPUT_DIR = Path("/123090047/cosmos_sae_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

INPUT_DIM = 3584
HIDDEN_DIM = 7168          # you can adjust this if you want
TAIL_RATIO = 0.5
DROPOUT = 0.2
SPARSITY_K = 512
ALPHA = 1.0                # tail reconstruction weight
BETA = 0.1                 # penaltiy for normal z_t activation

# NEW: tail reward parameters
TAIL_REWARD_COEFF = 0.5    # how strongly to encourage tail z_t activation
TAIL_REWARD_MAX = 2.0      # maximum reward (to avoid unbounded norm growth)

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ==================== Load data ====================
print(f"Loading features from {FEATURES_NPY}")
features = np.load(FEATURES_NPY)

print(f"Loading metadata from {META_JSON}")
with open(META_JSON, 'r', encoding='utf-8') as f:
    meta = json.load(f)

print("Loading original json labels...")
sample_to_label = {}
for path in JSON_PARTS:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for s in data['samples']:
            label_dict = s.get('labels', {})
            raw_label = label_dict.get('label', 'unknown')
            mapped_label = "白天" if raw_label == "normal" else raw_label
            sample_to_label[s['sample_token']] = mapped_label

scene_types = []
for m in meta:
    tok = m.get('sample_token')
    scene_types.append(sample_to_label.get(tok, 'unknown'))

scene_types = np.array(scene_types)
NORMAL_CLASS = "白天"

N = len(features)
print(f"Total samples: {N}, feature dimension: {INPUT_DIM}")
print(f"Categories: {np.unique(scene_types)}")

# Standardization
mean = features.mean(axis=0, keepdims=True)
std = features.std(axis=0, keepdims=True)
features = (features - mean) / (std + 1e-8)
print(f"After normalization: mean={features.mean():.4f}, std={features.std():.4f}")

is_tail = (scene_types != NORMAL_CLASS).astype(np.float32)
print(f"Normal samples: {(is_tail == 0).sum()}, Tail samples: {(is_tail == 1).sum()}")

# Train/val split
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
indices = np.random.permutation(N)
split_idx = int(TRAIN_RATIO * N)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

train_feat = torch.FloatTensor(features[train_idx])
val_feat = torch.FloatTensor(features[val_idx])
train_tail = torch.FloatTensor(is_tail[train_idx])
val_tail = torch.FloatTensor(is_tail[val_idx])

train_loader = DataLoader(TensorDataset(train_feat, train_tail), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_feat, val_tail), batch_size=BATCH_SIZE)

# ==================== Model definition (modified) ====================
class LongTailGuidedSAE_AbsTopK(nn.Module):
    def __init__(self, input_dim, hidden_dim, tail_ratio=0.5,
                 alpha=2.0, beta=0.01, dropout=0.2, sparsity_k=512,
                 tail_reward_coeff=0.5, tail_reward_max=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sparsity_k = sparsity_k
        self.tail_reward_coeff = tail_reward_coeff
        self.tail_reward_max = tail_reward_max

        self.tail_dim = int(hidden_dim * tail_ratio)
        self.normal_dim = hidden_dim - self.tail_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        z = self.encoder(x)
        z = self.dropout(z)

        # ----- AbsTop‑K selection -----
        abs_z = z.abs()
        _, topk_idx = torch.topk(abs_z, k=self.sparsity_k, dim=1)
        mask = torch.zeros_like(z)
        mask.scatter_(1, topk_idx, 1.0)
        z = z * mask                        # keep original values

        z_n = z[:, :self.normal_dim]
        z_t = z[:, self.normal_dim:]
        return z, z_n, z_t

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, is_tail):
        z, z_n, z_t = self.encode(x)
        x_hat = self.decode(z)

        # Weighted reconstruction loss
        weight = 1 + self.alpha * is_tail.unsqueeze(1)
        mse = (x_hat - x) ** 2
        L_recon = (weight * mse).mean()

        # Penalty for normal samples (z_t should be small)
        normal_mask = (is_tail == 0)
        if normal_mask.sum() > 0:
            z_t_norm_sq = torch.norm(z_t[normal_mask], p=2, dim=1) ** 2
            L_normal = self.beta * z_t_norm_sq.mean()
        else:
            L_normal = torch.tensor(0.0, device=x.device)

        # ----- Tail reward: encourage z_t activation for tail samples -----
        tail_mask = (is_tail == 1)
        if tail_mask.sum() > 0:
            z_t_norm = torch.norm(z_t[tail_mask], p=2, dim=1)   # [n_tail]
            # Reward = - coeff * min(norm, max)   (negative loss)
            L_tail = -self.tail_reward_coeff * torch.clamp(z_t_norm, max=self.tail_reward_max).mean()
        else:
            L_tail = torch.tensor(0.0, device=x.device)

        total_loss = L_recon + L_normal + L_tail
        loss_dict = {
            "total": total_loss,
            "recon": L_recon,
            "normal": L_normal,
            "tail_reward": L_tail,          # will be negative
        }
        return total_loss, loss_dict, z_n, z_t, x_hat

# ==================== Initialize ====================
model = LongTailGuidedSAE_AbsTopK(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    tail_ratio=TAIL_RATIO,
    alpha=ALPHA,
    beta=BETA,
    dropout=DROPOUT,
    sparsity_k=SPARSITY_K,
    tail_reward_coeff=TAIL_REWARD_COEFF,
    tail_reward_max=TAIL_REWARD_MAX
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# ==================== Training (with loss breakdown) ====================
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_total = 0.0
    epoch_recon = 0.0
    epoch_normal = 0.0
    epoch_tail = 0.0         # tail_reward accumulator
    num_batches = 0

    for x, t in train_loader:
        x, t = x.to(DEVICE), t.to(DEVICE)
        loss, loss_dict, _, _, _ = model(x, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_total += loss_dict['total'].item()
        epoch_recon += loss_dict['recon'].item()
        epoch_normal += loss_dict['normal'].item()
        epoch_tail += loss_dict['tail_reward'].item()
        num_batches += 1

    avg_train_total = epoch_total / num_batches
    avg_train_recon = epoch_recon / num_batches
    avg_train_normal = epoch_normal / num_batches
    avg_train_tail = epoch_tail / num_batches

    # Validation (no tail reward accumulation needed, but we still compute loss)
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for x, t in val_loader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            loss, loss_dict, _, _, _ = model(x, t)
            val_total += loss_dict['total'].item()
    val_total /= len(val_loader)

    train_loss_history.append(avg_train_total)
    val_loss_history.append(val_total)

    scheduler.step(val_total)

    if val_total < best_val_loss:
        best_val_loss = val_total
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    # Print every 20 epochs with detailed loss components
    if (epoch+1) % 20 == 0 or epoch == 0:
        print(f"[Epoch {epoch+1:3d}/{EPOCHS}] "
              f"Train total={avg_train_total:.4f} recon={avg_train_recon:.4f} "
              f"normal={avg_train_normal:.4f} tail_reward={avg_train_tail:.4f} | "
              f"Val total={val_total:.4f} (best: {best_val_loss:.4f})")

# ==================== Training curve ====================
plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.legend()
plt.savefig(OUTPUT_DIR / "training_loss_curve.png")
plt.close()

# ==================== Extract z_n, z_t ====================
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))
model.eval()
all_z_n, all_z_t = [], []
with torch.no_grad():
    for i in range(0, len(features), BATCH_SIZE):
        x = torch.FloatTensor(features[i:i+BATCH_SIZE]).to(DEVICE)
        _, z_n, z_t = model.encode(x)
        all_z_n.append(z_n.cpu().numpy())
        all_z_t.append(z_t.cpu().numpy())

all_z_n = np.concatenate(all_z_n, axis=0)
all_z_t = np.concatenate(all_z_t, axis=0)

# ==================== Activation statistics ====================
def compute_activation_stats(z_n, z_t, is_tail_labels, threshold=0.01):
    z_n_active = (np.abs(z_n) > threshold).any(axis=1)
    z_t_active = (np.abs(z_t) > threshold).any(axis=1)
    
    normal_mask = (is_tail_labels == 0)
    tail_mask = (is_tail_labels == 1)
    
    stats = {
        'normal_zn_active': z_n_active[normal_mask].sum(),
        'normal_zn_total': normal_mask.sum(),
        'normal_zt_active': z_t_active[normal_mask].sum(),
        'normal_zt_total': normal_mask.sum(),
        'tail_zn_active': z_n_active[tail_mask].sum(),
        'tail_zn_total': tail_mask.sum(),
        'tail_zt_active': z_t_active[tail_mask].sum(),
        'tail_zt_total': tail_mask.sum(),
    }
    return stats

print("\n" + "="*70)
print("全样本激活统计 (All Samples)")
print("="*70)
all_stats = compute_activation_stats(all_z_n, all_z_t, is_tail)
print(f"普通样本 (Normal): {all_stats['normal_zn_total']}")
print(f"  - z_n激活: {all_stats['normal_zn_active']}/{all_stats['normal_zn_total']} ({all_stats['normal_zn_active']/max(1,all_stats['normal_zn_total'])*100:.1f}%)")
print(f"  - z_t激活: {all_stats['normal_zt_active']}/{all_stats['normal_zt_total']} ({all_stats['normal_zt_active']/max(1,all_stats['normal_zt_total'])*100:.1f}%)")
print(f"长尾样本 (LongTail): {all_stats['tail_zn_total']}")
print(f"  - z_n激活: {all_stats['tail_zn_active']}/{all_stats['tail_zn_total']} ({all_stats['tail_zn_active']/max(1,all_stats['tail_zn_total'])*100:.1f}%)")
print(f"  - z_t激活: {all_stats['tail_zt_active']}/{all_stats['tail_zt_total']} ({all_stats['tail_zt_active']/max(1,all_stats['tail_zt_total'])*100:.1f}%)")

# Split stats for train/val
train_z_n = all_z_n[train_idx]
train_z_t = all_z_t[train_idx]
train_is_tail = is_tail[train_idx]

val_z_n = all_z_n[val_idx]
val_z_t = all_z_t[val_idx]
val_is_tail = is_tail[val_idx]

print("\n" + "="*70)
print("训练集激活统计 (Training Set)")
print("="*70)
train_stats = compute_activation_stats(train_z_n, train_z_t, train_is_tail)
print(f"普通样本 (Normal): {train_stats['normal_zn_total']}")
print(f"  - z_n激活: {train_stats['normal_zn_active']}/{train_stats['normal_zn_total']} ({train_stats['normal_zn_active']/max(1,train_stats['normal_zn_total'])*100:.1f}%)")
print(f"  - z_t激活: {train_stats['normal_zt_active']}/{train_stats['normal_zt_total']} ({train_stats['normal_zt_active']/max(1,train_stats['normal_zt_total'])*100:.1f}%)")
print(f"长尾样本 (LongTail): {train_stats['tail_zn_total']}")
print(f"  - z_n激活: {train_stats['tail_zn_active']}/{train_stats['tail_zn_total']} ({train_stats['tail_zn_active']/max(1,train_stats['tail_zn_total'])*100:.1f}%)")
print(f"  - z_t激活: {train_stats['tail_zt_active']}/{train_stats['tail_zt_total']} ({train_stats['tail_zt_active']/max(1,train_stats['tail_zt_total'])*100:.1f}%)")

print("\n" + "="*70)
print("验证集激活统计 (Validation Set)")
print("="*70)
val_stats = compute_activation_stats(val_z_n, val_z_t, val_is_tail)
print(f"普通样本 (Normal): {val_stats['normal_zn_total']}")
print(f"  - z_n激活: {val_stats['normal_zn_active']}/{val_stats['normal_zn_total']} ({val_stats['normal_zn_active']/max(1,val_stats['normal_zn_total'])*100:.1f}%)")
print(f"  - z_t激活: {val_stats['normal_zt_active']}/{val_stats['normal_zt_total']} ({val_stats['normal_zt_active']/max(1,val_stats['normal_zt_total'])*100:.1f}%)")
print(f"长尾样本 (LongTail): {val_stats['tail_zn_total']}")
print(f"  - z_n激活: {val_stats['tail_zn_active']}/{val_stats['tail_zn_total']} ({val_stats['tail_zn_active']/max(1,val_stats['tail_zn_total'])*100:.1f}%)")
print(f"  - z_t激活: {val_stats['tail_zt_active']}/{val_stats['tail_zt_total']} ({val_stats['tail_zt_active']/max(1,val_stats['tail_zt_total'])*100:.1f}%)")

# Overall AUC
l2_scores = np.linalg.norm(all_z_t, axis=1)
auc_total = roc_auc_score(is_tail, l2_scores)
print(f"\n{'='*70}")
print(f"Overall Tail detection AUC (using ||z_t||₂): {auc_total:.4f}")
print(f"{'='*70}")

# Save extracted features
np.save(OUTPUT_DIR / "z_n_general.npy", all_z_n)
np.save(OUTPUT_DIR / "z_t_longtail.npy", all_z_t)
np.save(OUTPUT_DIR / "is_tail.npy", is_tail)
np.save(OUTPUT_DIR / "scene_types.npy", scene_types)

print(f"\n✅ 所有输出已保存到: {OUTPUT_DIR}")
