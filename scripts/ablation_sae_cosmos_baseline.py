
#!/usr/bin/env python3

# --- Role in this repository -----------------------------------------------
# This is the second ABLATION variant referenced in the project report as
# "Train SAE Cosmos": AbsTopK sparsity with a training-time-only reward that
# is added to tail-subspace scores *before* Top-K selection (rather than
# rewarding the resulting activation norm directly, as in
# scripts/sae_abstopk_tail_reward.py). It is the weakest of the three SAE
# variants compared in the report, e.g. best F1=0.8071 at layer 28 vs. 0.8472
# for the final method. Kept here for completeness / reproducibility of the
# ablation study. See ../README.md#results.
# ------------------------------------------------------------------------------

"""
LongTail‑Guided SAE with AbsTop‑K sparsity + Tail Reward for Cosmos Extracted Features
- Data: layer27_mlp_output_last_token.npy (3584-dim, last token)
- Metadata: meta.json (to map to JSON labels)
- Labels sourced from 3 nuscene json files (labels -> label: "normal" vs "long_tail")
- Model: separate z_n (general) / z_t (tail), AbsTop‑K sparsity
- Tail reward encourages z_t activation for long‑tail samples, capped by max threshold
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
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

INPUT_DIM = 3584   # Cosmo 7B hidden dimension
HIDDEN_DIM = 3584
TAIL_RATIO = 0.5
DROPOUT = 0.4
SPARSITY_K = 512         # Top‑K sparsity (now based on absolute values)
ALPHA = 1.0                # Tail reconstruction weight multiplier
BETA = 0.01               # Penalty for normal samples z_t

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# NEW: Tail reward parameters
TAIL_REWARD = 1.0          # base reward added to absolute scores of z_t for tail samples
TAIL_REWARD_MAX = 2.0      # maximum reward (clamp upper bound)
TAIL_REWARD_TRAINING_ONLY = True   # only apply reward during training

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

# ==================== Model definition (Modified) ====================
class LongTailGuidedSAE_AbsTopK(nn.Module):
    def __init__(self, input_dim, hidden_dim, tail_ratio=0.5,
                 alpha=2.0, beta=0.01, dropout=0.2, sparsity_k=512,
                 tail_reward=1.0, tail_reward_max=2.0):   # NEW
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sparsity_k = sparsity_k

        # NEW: tail reward parameters
        self.tail_reward = tail_reward
        self.tail_reward_max = tail_reward_max

        self.tail_dim = int(hidden_dim * tail_ratio)
        self.normal_dim = hidden_dim - self.tail_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, is_tail=None, apply_reward=False):
        """
        Args:
            x: input tensor [B, D]
            is_tail: [B] 0/1 tensor indicating tail samples
            apply_reward: whether to add tail reward (only during training)
        Returns:
            z_masked, z_n, z_t
        """
        z = self.encoder(x)
        z = self.dropout(z)                # [B, hidden_dim]

        # ----- compute selection scores (abs + optional reward) -----
        scores = z.abs()                    # [B, hidden_dim]

        if apply_reward and is_tail is not None:
            # reward only on the z_t part for tail samples
            z_t_scores = scores[:, self.normal_dim:]     # [B, tail_dim]
            # reward factor: per sample, same for all tail dimensions
            reward = torch.where(
                is_tail.unsqueeze(1).bool(),
                torch.full_like(z_t_scores, self.tail_reward),
                torch.zeros_like(z_t_scores)
            )   # [B, tail_dim]
            # clamp reward max
            reward = torch.clamp(reward, max=self.tail_reward_max)
            scores[:, self.normal_dim:] = z_t_scores + reward

        # ----- abstopk: choose k largest absolute scores -----
        _, topk_idx = torch.topk(scores, k=self.sparsity_k, dim=1)
        mask = torch.zeros_like(z)
        mask.scatter_(1, topk_idx, 1.0)
        z = z * mask                     # keep original values, not the modified scores

        z_n = z[:, :self.normal_dim]
        z_t = z[:, self.normal_dim:]
        return z, z_n, z_t

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, is_tail, apply_reward=False):   # NEW: apply_reward flag
        z, z_n, z_t = self.encode(x, is_tail, apply_reward)
        x_hat = self.decode(z)

        weight = 1 + self.alpha * is_tail.unsqueeze(1)
        mse = (x_hat - x) ** 2
        L_recon = (weight * mse).mean()

        normal_mask = (is_tail == 0)
        if normal_mask.sum() > 0:
            z_t_norm_sq = torch.norm(z_t[normal_mask], p=2, dim=1) ** 2
            L_normal = self.beta * z_t_norm_sq.mean()
        else:
            L_normal = torch.tensor(0.0, device=x.device)

        total_loss = L_recon + L_normal
        loss_dict = {
            "total": total_loss,
            "recon": L_recon,
            "normal": L_normal,
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
    tail_reward=TAIL_REWARD,           # NEW
    tail_reward_max=TAIL_REWARD_MAX    # NEW
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# ==================== Training (modified to print loss parts every 20 epochs) ====================
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_total = 0.0
    epoch_recon = 0.0          # NEW: accumulate loss parts
    epoch_normal = 0.0
    num_batches = 0

    for x, t in train_loader:
        x, t = x.to(DEVICE), t.to(DEVICE)
        # Training with tail reward applied
        loss, loss_dict, _, _, _ = model(x, t, apply_reward=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_total += loss_dict['total'].item()
        epoch_recon += loss_dict['recon'].item()
        epoch_normal += loss_dict['normal'].item()
        num_batches += 1

    avg_train_total = epoch_total / num_batches
    avg_train_recon = epoch_recon / num_batches
    avg_train_normal = epoch_normal / num_batches

    # Validation (no reward)
    model.eval()
    val_total = 0.0
    val_recon = 0.0
    val_normal = 0.0
    with torch.no_grad():
        for x, t in val_loader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            loss, loss_dict, _, _, _ = model(x, t, apply_reward=False)
            val_total += loss_dict['total'].item()
            val_recon += loss_dict['recon'].item()
            val_normal += loss_dict['normal'].item()
    val_total /= len(val_loader)
    val_recon /= len(val_loader)
    val_normal /= len(val_loader)

    train_loss_history.append(avg_train_total)
    val_loss_history.append(val_total)

    scheduler.step(val_total)

    if val_total < best_val_loss:
        best_val_loss = val_total
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    # NEW: print loss components every 20 epochs
    if (epoch+1) % 20 == 0 or epoch == 0:
        print(f"[Epoch {epoch+1:3d}/{EPOCHS}] "
              f"Train: total={avg_train_total:.4f} recon={avg_train_recon:.4f} normal={avg_train_normal:.4f} | "
              f"Val: total={val_total:.4f} recon={val_recon:.4f} normal={val_normal:.4f} "
              f"(best val total: {best_val_loss:.4f})")

# ==================== Training curve ====================
plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.legend()
plt.savefig(OUTPUT_DIR / "training_loss_curve.png")
plt.close()

# ==================== Extract z_n, z_t (inference without reward) ====================
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))
model.eval()
all_z_n, all_z_t = [], []
with torch.no_grad():
    for i in range(0, len(features), BATCH_SIZE):
        x = torch.FloatTensor(features[i:i+BATCH_SIZE]).to(DEVICE)
        # Inference without tail reward
        tail_labels = torch.FloatTensor(is_tail[i:i+BATCH_SIZE]).to(DEVICE) if i < len(is_tail) else None
        _, z_n, z_t = model.encode(x, is_tail=tail_labels, apply_reward=False)
        all_z_n.append(z_n.cpu().numpy())
        all_z_t.append(z_t.cpu().numpy())

all_z_n = np.concatenate(all_z_n, axis=0)
all_z_t = np.concatenate(all_z_t, axis=0)

# ==================== Activation statistics (keep the better one) ====================
def compute_activation_stats(z_n, z_t, is_tail_labels, threshold=0.01):
    """Count how many samples activate (have any feature > threshold) in z_n and z_t"""
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
