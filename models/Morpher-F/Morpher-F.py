# ============================================================
# Morpher: Autoregressive binary mask sequence forecasting
# - Dataset: YOLO-seg polygon TXT -> rasterized binary masks
# - Model: SpatialEncoder -> temporal backbone (GRU/LSTM/RNN/Transformer) -> DecoderWithMultiSkip
# - Inference: strict autoregressive (use predicted mask as next input)
# - Loss: Focal + SoftIoU + optional boundary loss (torch or CPU)
# - Metrics: mIoU, mAP@[.50:.95], HD, HD95, ASSD (plus optional phys_stats in test)
# NOTE:
# - The original code logic is kept unchanged; ONLY comments are added.
# - Some non-research utility parts are intentionally left minimally commented as requested
#   (e.g., remap_group_to_morphon, certain CLI plumbing).
# ============================================================

import os
import sys
import csv
import re
import random
import math
import functools
import argparse
import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

try:
    from timm.layers import DropPath as _TimmDropPath

    # Safe wrapper: if timm is available, reuse its DropPath implementation directly.
    class DropPathSafe(_TimmDropPath):
        pass
except Exception:
    # Fallback DropPath implementation (stochastic depth) if timm is unavailable.
    class DropPathSafe(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            # In eval mode or drop_prob=0, return identity.
            if (not self.training) or self.drop_prob == 0.0:
                return x
            keep_prob = 1.0 - self.drop_prob
            # Broadcast mask across non-batch dimensions.
            shape = (x.shape[0],) + (1,) * (x.dim() - 1)
            mask = x.new_empty(shape).bernoulli_(keep_prob)
            # Rescale to keep expected value unchanged.
            return x / keep_prob * mask

import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2
from torch.cuda.amp import autocast, GradScaler

# ============================================================
# Reproducibility / backend settings
# ============================================================
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        # Prefer higher precision matmul when supported (PyTorch 2.x).
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ============================================================
# Collate function for variable-length sequences
# - Input: list of (seq[T,H,W], length)
# - Output:
#   - padded batch: [B, T_max, H, W]
#   - frame_mask:  [B, T_max] bool, True for valid frames
# ============================================================
def pad_collate(batch):
    seqs, lens = zip(*batch)
    T_max = max(s.shape[0] for s in seqs)
    padded, masks = [], []
    for s, L in zip(seqs, lens):
        T, H, W = s.shape
        if T < T_max:
            pad = torch.zeros((T_max - T, H, W), dtype=s.dtype)
            s = torch.cat([s, pad], dim=0)
        padded.append(s)
        m = torch.zeros(T_max, dtype=torch.bool)
        m[:L] = True
        masks.append(m)
    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

# ============================================================
# Global config (acts like a mutable singleton)
# ============================================================
class Config:
    results_dir = "results"
    img_size = 640
    batch_size = 2
    step = 25
    obs_ratio = 0.8
    pool_size = img_size // 32
    nheads = 8
    dim_feedforward = 512
    compress_dim = 256
    num_layers = 3
    lr = 5e-5
    epochs_per_fold = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    augment_times = 1
    warmup_ratio = 0.1
    use_morphon = True
    train_path = r"dataset\train"
    val_path = r"dataset\test"
    num_workers_train = min(8, (os.cpu_count() or 8))
    num_workers_val = max(1, min(4, (os.cpu_count() or 4)))
    prefetch_factor_train = 4
    prefetch_factor_val = 2
    use_ckpt = False
    ckpt_segments = 3
    use_cpu_boundary_loss = False
    use_torch_boundary_loss = True
    loss_downscale = 0.25
    train_metric_every = 0
    eval_compute_hd_assd = True
    metrics_on_gpu = True
    use_torch_compile = True
    torch_compile_mode = "max-autotune"

# ============================================================
# Loss components
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal loss for binary segmentation (logits + binary target).
    Helps handle class imbalance by down-weighting easy examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce)
        return self.alpha * (1 - pt) ** self.gamma * bce

# ============================================================
# Positional encoding (cached sinusoidal encoding)
# - Cached by (T, d_model, device_str)
# ============================================================
@functools.lru_cache(maxsize=None)
def sinusoidal_encoding_cached(T, d_model, device_str):
    device = torch.device(device_str)
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(d_model // 2, dtype=torch.float32, device=device)
    rates = 1 / (10000 ** (2 * i / d_model))
    angs = pos * rates
    enc = torch.zeros((T, d_model), device=device)
    enc[:, 0::2] = torch.sin(angs)
    enc[:, 1::2] = torch.cos(angs)
    return enc

def sinusoidal_encoding(T, d_model, device):
    return sinusoidal_encoding_cached(T, d_model, str(device))

# ============================================================
# Soft IoU loss (differentiable IoU proxy)
# ============================================================
def soft_iou_loss(logits, mask, smooth=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * mask).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) - inter
    union = torch.where(union == 0, torch.ones_like(union), union)
    return 1 - ((inter + smooth) / (union + smooth))

# ============================================================
# CPU boundary IoU helper (uses scipy binary erosion)
# ============================================================
@functools.lru_cache(maxsize=None)
def _cached_struct(distance):
    return np.ones((2 * distance + 1, 2 * distance + 1), dtype=bool)

def boundary_iou_per_frame(pred, tgt, distance=None, smooth=1e-6):
    """
    Boundary IoU-style loss computed on CPU:
    - Convert pred->binary boundary map and tgt->boundary map
    - Compute IoU on boundaries
    """
    N, _, H, W = tgt.shape
    if distance is None:
        distance = int(round(0.02 * math.hypot(H, W)))

    prob = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()
    tgt_np = tgt.cpu().numpy()
    struct = _cached_struct(distance)

    def get_boundary(x_float01: np.ndarray) -> np.ndarray:
        xb = x_float01.astype(np.bool_)
        er = ndi.binary_erosion(xb, structure=struct, border_value=0)
        b = np.logical_and(xb, np.logical_not(er))
        return b.astype(np.float32)

    pb = np.stack([get_boundary(p[0]) for p in prob])
    tb = np.stack([get_boundary(t[0]) for t in tgt_np])

    pb = torch.from_numpy(pb).float().to(pred.device).unsqueeze(1)
    tb = torch.from_numpy(tb).float().to(pred.device).unsqueeze(1)

    inter = (pb * tb).sum(dim=(1, 2, 3))
    union = pb.sum(dim=(1, 2, 3)) + tb.sum(dim=(1, 2, 3)) - inter
    union = torch.where(union == 0, torch.ones_like(union), union)
    return 1 - ((inter + smooth) / (union + smooth))

# ============================================================
# Torch boundary loss (fast GPU-friendly edge approximation)
# - Uses maxpool to derive a boundary-like map
# ============================================================
def boundary_loss_torch(prob_or_logits, target_bin, k: int = 3, smooth: float = 1e-6):
    if prob_or_logits.dtype.is_floating_point:
        prob = torch.sigmoid(prob_or_logits)
    else:
        prob = prob_or_logits

    pad = (k - 1) // 2
    pooled_p = F.max_pool2d(prob, kernel_size=k, stride=1, padding=pad)
    edge_p = torch.clamp(pooled_p - prob, 0, 1)

    pooled_t = F.max_pool2d(target_bin, kernel_size=k, stride=1, padding=pad)
    edge_t = torch.clamp(pooled_t - target_bin, 0, 1)

    inter = (edge_p * edge_t).sum(dim=(1, 2, 3))
    union = edge_p.sum(dim=(1, 2, 3)) + edge_t.sum(dim=(1, 2, 3)) - inter
    return 1.0 - (inter + smooth) / (union + smooth)

# ============================================================
# Total loss (frame-wise) for predicted future frames
# - preds: [B, P, H, W] logits
# - tgts : [B, P, H, W] binary
# - pred_mask: [B, P] bool, valid predicted frames
# ============================================================
def total_loss_fn(preds, tgts, pred_mask):
    B, P, H, W = preds.shape
    p = preds.reshape(B * P, 1, H, W)
    t = tgts.reshape(B * P, 1, H, W)
    m = pred_mask.reshape(B * P)

    # If nothing valid, return zeros (avoid NaNs).
    if m.sum() == 0:
        z = torch.tensor(0., device=preds.device, dtype=torch.float32)
        return z, z, z, z

    p = p[m]
    t = t[m]

    fl = FocalLoss()(p, t).mean()
    si = soft_iou_loss(p, t).mean()

    bi = torch.tensor(0.0, device=p.device)
    if Config.use_torch_boundary_loss or Config.use_cpu_boundary_loss:
        # Optional downscale to reduce boundary-loss cost.
        if Config.loss_downscale and Config.loss_downscale < 1.0:
            scale = Config.loss_downscale
            p_ds = F.interpolate(p, scale_factor=scale, mode='bilinear', align_corners=False)
            t_ds = F.interpolate(t, scale_factor=scale, mode='nearest')
        else:
            p_ds, t_ds = p, t

        if Config.use_torch_boundary_loss:
            bi = boundary_loss_torch(p_ds, t_ds).mean()
        elif Config.use_cpu_boundary_loss:
            bi = boundary_iou_per_frame(p_ds, t_ds).mean()

    # Boundary term weighted by 0.5 in the total objective.
    total = fl + si + 0.5 * bi
    return total, fl, si, bi

# ============================================================
# YOLO-seg TXT polygon -> raster mask (cached)
# - Each line: class_id x1 y1 x2 y2 ... (normalized [0,1])
# ============================================================
@functools.lru_cache(maxsize=None)
def _txt2mask_cached(path, img_size):
    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    with open(path, 'r') as f:
        for ln in f:
            pts = list(map(float, ln.split()))
            if len(pts) < 7 or (len(pts) - 1) % 2:
                continue
            poly = [(x * img_size, y * img_size) for x, y in zip(pts[1::2], pts[2::2])]
            draw.polygon(poly, fill=1)
    return np.array(img, dtype=np.float32)

# ============================================================
# Dataset: folder-per-sequence, TXT-per-frame
# - root/
#    seq_001/
#       0.txt, 1.txt, ...
#    seq_002/
#       ...
# - step: temporal downsampling (take every `step`-th frame)
# - augment: random affine + flips (applied to masks)
# ============================================================
class MultiSequenceDataset(Dataset):
    def __init__(self, root, step=None, augment=None):
        self.root = root
        self.step = int(step) if step is not None else int(Config.step)
        self.augment = augment if augment is not None else 'train' in os.path.basename(root).lower()
        self._num_re = re.compile(r'(\d+)(?=\D*$)')

        folders = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
            key=lambda x: int(self._num(x)) if self._num(x) is not None else x
        )
        self.samples = []
        for fd in folders:
            sub = os.path.join(root, fd)
            txts = sorted(
                [os.path.join(sub, f) for f in os.listdir(sub) if f.endswith('.txt')],
                key=lambda x: int(self._num(os.path.basename(x)))
            )
            if len(txts) >= 2:
                self.samples.append(txts)

    def _num(self, s):
        m = self._num_re.search(s)
        return m.group(1) if m else None

    def __len__(self):
        # Repeat each sequence `augment_times` during training augmentation.
        return len(self.samples) * (Config.augment_times if self.augment else 1)

    def __getitem__(self, idx):
        # Map repeated indices back to the original sequence index.
        si = idx // Config.augment_times if self.augment else idx
        files = self.samples[si][::self.step]
        base_masks = [_txt2mask_cached(f, Config.img_size) for f in files]
        masks = base_masks

        # Simple geometric augmentation on binary masks (affine + flips).
        if self.augment:
            angle = random.uniform(-180, 180)
            tr = (random.uniform(-30, 30), random.uniform(-30, 30))
            hf, vf = random.random() < 0.5, random.random() < 0.5
            aug = []
            for m in masks:
                im = Image.fromarray((m * 255).astype(np.uint8))
                im = TF.affine(im, angle, tr, scale=1.0, shear=0, fill=0)
                if hf:
                    im = TF.hflip(im)
                if vf:
                    im = TF.vflip(im)
                aug.append(np.array(im, dtype=np.float32) / 255.)
            masks = aug

        seq = torch.from_numpy(np.stack(masks)).float()
        length = seq.shape[0]
        return seq, length

# ============================================================
# Spatial encoder: per-frame mask -> latent vector z_t and multi-scale feature maps
# - Input : [B, 1, H, W]
# - Output:
#   - z: [B, compress_dim]
#   - fmaps: skip features at multiple resolutions (for the decoder)
# ============================================================
class SpatialEncoder(nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(2)

        # Force a fixed spatial size for the latent projection.
        self.adapt_pool = nn.AdaptiveAvgPool2d((pool, pool))

        # Flatten -> compress_dim latent vector.
        self.spatial_pool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pool * pool, Config.compress_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        x4 = self.conv4(p3)
        p4 = self.pool4(x4)

        p5 = self.adapt_pool(p4)

        z = self.spatial_pool(p5)

        # Multi-scale feature maps for decoder skip connections.
        f1 = p1
        f2 = p2
        f3 = p3
        f4 = p4
        f5 = p5
        return z, [f1, f2, f3, f4, f5]

# ============================================================
# Decoder with multi-scale skip fusion
# - Input: latent vector z (for a target future frame) + last observed/predicted fmaps
# - Output: predicted mask logits [B, 1, H, W]
# ============================================================
class DecoderWithMultiSkip(nn.Module):
    def __init__(self, pool, c_latent=Config.compress_dim):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(c_latent, c_latent), nn.ReLU(), nn.Dropout(0.2))
        self.fc2 = nn.Sequential(nn.Linear(c_latent, 64 * pool * pool), nn.ReLU(), nn.Dropout(0.2))

        # Fuse latent feature with deepest skip f5, then upsample progressively.
        self.fuse5 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.ReLU(),
        )

        self.fuse4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.ReLU(),
        )

        self.fuse3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
        )

        self.fuse2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
        )

        # Final upsampling to original resolution (x2).
        self.final_up = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z, fmaps):
        f1, f2, f3, f4, f5 = fmaps
        x = self.fc1(z)
        x = self.fc2(x)
        ph, pw = f5.size(2), f5.size(3)
        x = x.view(x.size(0), 64, ph, pw)

        x = torch.cat([x, f5], dim=1)
        x = self.fuse5(x)
        x = self.up5(x)

        x = torch.cat([x, f4], dim=1)
        x = self.fuse4(x)
        x = self.up4(x)

        x = torch.cat([x, f3], dim=1)
        x = self.fuse3(x)
        x = self.up3(x)

        x = torch.cat([x, f2], dim=1)
        x = self.fuse2(x)
        x = self.up2(x)

        x = torch.cat([x, f1], dim=1)
        x = self.fuse1(x)

        x = self.final_up(x)
        return x

# ============================================================
# Morpher (GRU backbone)
# - Core idea:
#   1) Encode observed masks -> latent sequence {z_1..z_mid}
#   2) Morphon pooling/attention to inject global sequence context into tokens
#   3) Autoregressive loop:
#       - temporal backbone predicts next latent z_{t+1}
#       - decoder produces next mask logits
#       - predicted mask -> re-encode -> append to token sequence
# ============================================================
class Morpher(nn.Module):
    def __init__(self):
        super().__init__()
        # Gating scalar for blending Morphon global token with per-time tokens.
        self.alpha = nn.Parameter(torch.tensor(0.5))
        assert Config.img_size % 32 == 0

        # Morphon: compute per-token scores -> weighted pool -> query -> attention over tokens.
        self.morphon_score_mlp = nn.Linear(Config.compress_dim, 1)
        self.morphon_query_mlp = nn.Sequential(
            nn.Linear(Config.compress_dim, Config.compress_dim),
            nn.ReLU(),
            nn.Linear(Config.compress_dim, Config.compress_dim)
        )
        self.morphon_attn = nn.MultiheadAttention(Config.compress_dim, 4, batch_first=True)

        pool = Config.pool_size
        self.encoder = SpatialEncoder(pool)

        # Normalize tokens before temporal modeling; DropPath for regularization.
        self.pre_ln = nn.LayerNorm(Config.compress_dim)
        self.dp_in = DropPathSafe(0.1)

        # Temporal backbone: GRU over latent tokens.
        self.rnn = nn.GRU(
            input_size=Config.compress_dim,
            hidden_size=Config.compress_dim,
            num_layers=Config.num_layers,
            batch_first=True,
            dropout=0.2 if Config.num_layers > 1 else 0.0
        )

        self.decoder = DecoderWithMultiSkip(pool)

    def encode_frame(self, frame):
        # Convert [B,H,W] -> [B,1,H,W] and use channels_last for CUDA speed.
        x = frame.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        if Config.use_ckpt and self.training:
            x.requires_grad_(True)
            z, fmaps = self.encoder(x)
        else:
            z, fmaps = self.encoder(x)
        return z, fmaps

    def _single_sequence_forward(self, seq_i):
        """
        Forward on a single sequence (variable length).
        seq_i: [T_i, H, W] float mask in {0,1}
        Returns:
          pred: [P_i, H, W] logits for predicted future frames
          tgt : [P_i, H, W] binary targets (ground truth future frames)
        """
        device = seq_i.device
        T_i = seq_i.shape[0]

        # Split point based on obs_ratio (clamped for safety).
        r = float(Config.obs_ratio)
        r = 0.8 if (not np.isfinite(r)) else r
        r = min(0.999, max(0.001, r))
        mid_i = max(1, min(T_i - 1, int(math.ceil(r * T_i))))
        P_i = T_i - mid_i

        obs = seq_i[:mid_i]
        tgt = seq_i[mid_i:T_i]

        # Encode observed frames -> latent token sequence.
        past_seq = []
        fmaps_last = None
        for t in range(mid_i):
            z_t, fmaps_t = self.encode_frame(obs[t].unsqueeze(0))
            past_seq.append(z_t.squeeze(0))
            fmaps_last = fmaps_t

        past_seq = torch.stack(past_seq, dim=0).unsqueeze(0)

        # Morphon token mixing (global query attends over all past tokens).
        ps = past_seq
        scores = self.morphon_score_mlp(ps)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * ps).sum(dim=1)
        query = self.morphon_query_mlp(pooled).unsqueeze(1)
        tok, _ = self.morphon_attn(query, ps, ps)
        grp = tok.squeeze(1)
        g = torch.sigmoid(self.alpha)
        ps = (1 - g) * ps + g * grp.unsqueeze(1)
        past_seq = ps

        preds = []
        cur = past_seq.clone()
        fml = fmaps_last

        # Autoregressive prediction loop over P_i future steps.
        for _ in range(P_i):
            ps = cur.float()
            te = sinusoidal_encoding(ps.size(1), Config.compress_dim, device).unsqueeze(0).float()
            x_in = self.pre_ln(ps) + te
            x_in = self.dp_in(x_in)

            out_seq = self.rnn(x_in)
            if isinstance(out_seq, tuple):
                out = out_seq[0][:, -1]
            else:
                out = out_seq[:, -1]

            out = out.to(ps.dtype)
            dec = self.decoder(out, fml)
            preds.append(dec[:, 0])

            # Feed predicted mask (sigmoid) back into encoder for next-step token.
            nxt = torch.sigmoid(dec[:, 0]).detach()
            z_next, fml = self.encode_frame(nxt)
            cur = torch.cat([cur, z_next.unsqueeze(1)], dim=1)

        pred = torch.cat(preds, dim=0) if len(preds) else seq_i.new_zeros((0, seq_i.shape[1], seq_i.shape[2]))
        return pred, tgt

    def forward(self, seq, frame_mask):
        """
        Batch forward for variable-length sequences.
        seq: [B, T, H, W]
        frame_mask: [B, T] bool
        Returns:
          pred:  [B, Pmax, H, W]
          tgt:   [B, Pmax, H, W]
          pmask: [B, Pmax] bool for valid predicted frames
        """
        B, T, H, W = seq.shape
        preds_list, tgts_list, lens = [], [], []

        for i in range(B):
            L_i = int(frame_mask[i].sum().item())
            if L_i < 2:
                empty_pred = torch.zeros((1, H, W), device=seq.device)
                empty_tgt = torch.zeros((1, H, W), device=seq.device)
                preds_list.append(empty_pred)
                tgts_list.append(empty_tgt)
                lens.append(1)
                continue

            seq_i = seq[i, :L_i]
            pred_i, tgt_i = self._single_sequence_forward(seq_i)
            preds_list.append(pred_i)
            tgts_list.append(tgt_i)
            lens.append(pred_i.shape[0])

        # Pad predicted-length dimension to the max across batch.
        Pmax = max(lens) if len(lens) else 1
        pred_batch, tgt_batch, pmask_batch = [], [], []
        for p, t, L in zip(preds_list, tgts_list, lens):
            if L < Pmax:
                padp = torch.zeros((Pmax - L, H, W), device=p.device)
                p = torch.cat([p, padp], dim=0)
                padt = torch.zeros((Pmax - L, H, W), device=t.device)
                t = torch.cat([t, padt], dim=0)
            pred_batch.append(p)
            tgt_batch.append(t)
            m = torch.zeros(Pmax, dtype=torch.bool, device=p.device)
            m[:L] = True
            pmask_batch.append(m)

        pred = torch.stack(pred_batch, dim=0)
        tgt = torch.stack(tgt_batch, dim=0)
        pmask = torch.stack(pmask_batch, dim=0)
        return pred, tgt, pmask

# ============================================================
# MorpherLSTM / MorpherRNN / MorpherTransformer:
# - Same pipeline as Morpher (GRU) but different temporal backbone.
# ============================================================
class MorpherLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        assert Config.img_size % 32 == 0

        self.morphon_score_mlp = nn.Linear(Config.compress_dim, 1)
        self.morphon_query_mlp = nn.Sequential(
            nn.Linear(Config.compress_dim, Config.compress_dim),
            nn.ReLU(),
            nn.Linear(Config.compress_dim, Config.compress_dim)
        )
        self.morphon_attn = nn.MultiheadAttention(Config.compress_dim, 4, batch_first=True)

        pool = Config.pool_size
        self.encoder = SpatialEncoder(pool)

        self.pre_ln = nn.LayerNorm(Config.compress_dim)
        self.dp_in = DropPathSafe(0.1)

        self.rnn = nn.LSTM(
            input_size=Config.compress_dim,
            hidden_size=Config.compress_dim,
            num_layers=Config.num_layers,
            batch_first=True,
            dropout=0.2 if Config.num_layers > 1 else 0.0
        )

        self.decoder = DecoderWithMultiSkip(pool)

    def encode_frame(self, frame):
        x = frame.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        if Config.use_ckpt and self.training:
            x.requires_grad_(True)
            z, fmaps = self.encoder(x)
        else:
            z, fmaps = self.encoder(x)
        return z, fmaps

    def _single_sequence_forward(self, seq_i):
        device = seq_i.device
        T_i = seq_i.shape[0]

        r = float(Config.obs_ratio)
        r = 0.8 if (not np.isfinite(r)) else r
        r = min(0.999, max(0.001, r))
        mid_i = max(1, min(T_i - 1, int(math.ceil(r * T_i))))
        P_i = T_i - mid_i

        obs = seq_i[:mid_i]
        tgt = seq_i[mid_i:T_i]

        past_seq = []
        fmaps_last = None
        for t in range(mid_i):
            z_t, fmaps_t = self.encode_frame(obs[t].unsqueeze(0))
            past_seq.append(z_t.squeeze(0))
            fmaps_last = fmaps_t
        past_seq = torch.stack(past_seq, dim=0).unsqueeze(0)

        ps = past_seq
        scores = self.morphon_score_mlp(ps)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * ps).sum(dim=1)
        query = self.morphon_query_mlp(pooled).unsqueeze(1)
        tok, _ = self.morphon_attn(query, ps, ps)
        grp = tok.squeeze(1)
        g = torch.sigmoid(self.alpha)
        ps = (1 - g) * ps + g * grp.unsqueeze(1)
        past_seq = ps

        preds = []
        cur = past_seq.clone()
        fml = fmaps_last

        for _ in range(P_i):
            ps = cur.float()
            te = sinusoidal_encoding(ps.size(1), Config.compress_dim, device).unsqueeze(0).float()
            x_in = self.pre_ln(ps) + te
            x_in = self.dp_in(x_in)

            out_seq = self.rnn(x_in)
            if isinstance(out_seq, tuple):
                out = out_seq[0][:, -1]
            else:
                out = out_seq[:, -1]

            out = out.to(ps.dtype)
            dec = self.decoder(out, fml)
            preds.append(dec[:, 0])

            nxt = torch.sigmoid(dec[:, 0]).detach()
            z_next, fml = self.encode_frame(nxt)
            cur = torch.cat([cur, z_next.unsqueeze(1)], dim=1)

        pred = torch.cat(preds, dim=0) if len(preds) else seq_i.new_zeros((0, seq_i.shape[1], seq_i.shape[2]))
        return pred, tgt

    def forward(self, seq, frame_mask):
        B, T, H, W = seq.shape
        preds_list, tgts_list, lens = [], [], []

        for i in range(B):
            L_i = int(frame_mask[i].sum().item())
            if L_i < 2:
                empty_pred = torch.zeros((1, H, W), device=seq.device)
                empty_tgt = torch.zeros((1, H, W), device=seq.device)
                preds_list.append(empty_pred)
                tgts_list.append(empty_tgt)
                lens.append(1)
                continue

            seq_i = seq[i, :L_i]
            pred_i, tgt_i = self._single_sequence_forward(seq_i)
            preds_list.append(pred_i)
            tgts_list.append(tgt_i)
            lens.append(pred_i.shape[0])

        Pmax = max(lens) if len(lens) else 1
        pred_batch, tgt_batch, pmask_batch = [], [], []
        for p, t, L in zip(preds_list, tgts_list, lens):
            if L < Pmax:
                padp = torch.zeros((Pmax - L, H, W), device=p.device)
                p = torch.cat([p, padp], dim=0)
                padt = torch.zeros((Pmax - L, H, W), device=t.device)
                t = torch.cat([t, padt], dim=0)
            pred_batch.append(p)
            tgt_batch.append(t)
            m = torch.zeros(Pmax, dtype=torch.bool, device=p.device)
            m[:L] = True
            pmask_batch.append(m)

        pred = torch.stack(pred_batch, dim=0)
        tgt = torch.stack(tgt_batch, dim=0)
        pmask = torch.stack(pmask_batch, dim=0)
        return pred, tgt, pmask

class MorpherRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        assert Config.img_size % 32 == 0

        self.morphon_score_mlp = nn.Linear(Config.compress_dim, 1)
        self.morphon_query_mlp = nn.Sequential(
            nn.Linear(Config.compress_dim, Config.compress_dim),
            nn.ReLU(),
            nn.Linear(Config.compress_dim, Config.compress_dim)
        )
        self.morphon_attn = nn.MultiheadAttention(Config.compress_dim, 4, batch_first=True)

        pool = Config.pool_size
        self.encoder = SpatialEncoder(pool)

        self.pre_ln = nn.LayerNorm(Config.compress_dim)
        self.dp_in = DropPathSafe(0.1)

        self.rnn = nn.RNN(
            input_size=Config.compress_dim,
            hidden_size=Config.compress_dim,
            num_layers=Config.num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=0.2 if Config.num_layers > 1 else 0.0
        )

        self.decoder = DecoderWithMultiSkip(pool)

    def encode_frame(self, frame):
        x = frame.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        if Config.use_ckpt and self.training:
            x.requires_grad_(True)
            z, fmaps = self.encoder(x)
        else:
            z, fmaps = self.encoder(x)
        return z, fmaps

    def _single_sequence_forward(self, seq_i):
        device = seq_i.device
        T_i = seq_i.shape[0]

        r = float(Config.obs_ratio)
        r = 0.8 if (not np.isfinite(r)) else r
        r = min(0.999, max(0.001, r))
        mid_i = max(1, min(T_i - 1, int(math.ceil(r * T_i))))
        P_i = T_i - mid_i

        obs = seq_i[:mid_i]
        tgt = seq_i[mid_i:T_i]

        past_seq = []
        fmaps_last = None
        for t in range(mid_i):
            z_t, fmaps_t = self.encode_frame(obs[t].unsqueeze(0))
            past_seq.append(z_t.squeeze(0))
            fmaps_last = fmaps_t
        past_seq = torch.stack(past_seq, dim=0).unsqueeze(0)

        ps = past_seq
        scores = self.morphon_score_mlp(ps)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * ps).sum(dim=1)
        query = self.morphon_query_mlp(pooled).unsqueeze(1)
        tok, _ = self.morphon_attn(query, ps, ps)
        grp = tok.squeeze(1)
        g = torch.sigmoid(self.alpha)
        ps = (1 - g) * ps + g * grp.unsqueeze(1)
        past_seq = ps

        preds = []
        cur = past_seq.clone()
        fml = fmaps_last

        for _ in range(P_i):
            ps = cur.float()
            te = sinusoidal_encoding(ps.size(1), Config.compress_dim, device).unsqueeze(0).float()
            x_in = self.pre_ln(ps) + te
            x_in = self.dp_in(x_in)

            out_seq = self.rnn(x_in)
            if isinstance(out_seq, tuple):
                out = out_seq[0][:, -1]
            else:
                out = out_seq[:, -1]

            out = out.to(ps.dtype)
            dec = self.decoder(out, fml)
            preds.append(dec[:, 0])

            nxt = torch.sigmoid(dec[:, 0]).detach()
            z_next, fml = self.encode_frame(nxt)
            cur = torch.cat([cur, z_next.unsqueeze(1)], dim=1)

        pred = torch.cat(preds, dim=0) if len(preds) else seq_i.new_zeros((0, seq_i.shape[1], seq_i.shape[2]))
        return pred, tgt

    def forward(self, seq, frame_mask):
        B, T, H, W = seq.shape
        preds_list, tgts_list, lens = [], [], []

        for i in range(B):
            L_i = int(frame_mask[i].sum().item())
            if L_i < 2:
                empty_pred = torch.zeros((1, H, W), device=seq.device)
                empty_tgt = torch.zeros((1, H, W), device=seq.device)
                preds_list.append(empty_pred)
                tgts_list.append(empty_tgt)
                lens.append(1)
                continue

            seq_i = seq[i, :L_i]
            pred_i, tgt_i = self._single_sequence_forward(seq_i)
            preds_list.append(pred_i)
            tgts_list.append(tgt_i)
            lens.append(pred_i.shape[0])

        Pmax = max(lens) if len(lens) else 1
        pred_batch, tgt_batch, pmask_batch = [], [], []
        for p, t, L in zip(preds_list, tgts_list, lens):
            if L < Pmax:
                padp = torch.zeros((Pmax - L, H, W), device=p.device)
                p = torch.cat([p, padp], dim=0)
                padt = torch.zeros((Pmax - L, H, W), device=t.device)
                t = torch.cat([t, padt], dim=0)
            pred_batch.append(p)
            tgt_batch.append(t)
            m = torch.zeros(Pmax, dtype=torch.bool, device=p.device)
            m[:L] = True
            pmask_batch.append(m)

        pred = torch.stack(pred_batch, dim=0)
        tgt = torch.stack(tgt_batch, dim=0)
        pmask = torch.stack(pmask_batch, dim=0)
        return pred, tgt, pmask

class MorpherTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        assert Config.img_size % 32 == 0

        self.morphon_score_mlp = nn.Linear(Config.compress_dim, 1)
        self.morphon_query_mlp = nn.Sequential(
            nn.Linear(Config.compress_dim, Config.compress_dim),
            nn.ReLU(),
            nn.Linear(Config.compress_dim, Config.compress_dim)
        )
        self.morphon_attn = nn.MultiheadAttention(Config.compress_dim, 4, batch_first=True)

        pool = Config.pool_size
        self.encoder = SpatialEncoder(pool)

        self.pre_ln = nn.LayerNorm(Config.compress_dim)
        self.dp_in = DropPathSafe(0.1)

        # TransformerEncoder as temporal backbone over latent tokens.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=Config.compress_dim,
            nhead=Config.nheads,
            dim_feedforward=Config.dim_feedforward,
            dropout=0.2,
            batch_first=True
        )
        self.encoder_temporal = nn.TransformerEncoder(enc_layer, num_layers=Config.num_layers)

        self.decoder = DecoderWithMultiSkip(pool)

    def encode_frame(self, frame):
        x = frame.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        if Config.use_ckpt and self.training:
            x.requires_grad_(True)
            z, fmaps = self.encoder(x)
        else:
            z, fmaps = self.encoder(x)
        return z, fmaps

    def _single_sequence_forward(self, seq_i):
        device = seq_i.device
        T_i = seq_i.shape[0]

        r = float(Config.obs_ratio)
        r = 0.8 if (not np.isfinite(r)) else r
        r = min(0.999, max(0.001, r))
        mid_i = max(1, min(T_i - 1, int(math.ceil(r * T_i))))
        P_i = T_i - mid_i

        obs = seq_i[:mid_i]
        tgt = seq_i[mid_i:T_i]

        past_seq = []
        fmaps_last = None
        for t in range(mid_i):
            z_t, fmaps_t = self.encode_frame(obs[t].unsqueeze(0))
            past_seq.append(z_t.squeeze(0))
            fmaps_last = fmaps_t
        past_seq = torch.stack(past_seq, dim=0).unsqueeze(0)

        ps = past_seq
        scores = self.morphon_score_mlp(ps)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * ps).sum(dim=1)
        query = self.morphon_query_mlp(pooled).unsqueeze(1)
        tok, _ = self.morphon_attn(query, ps, ps)
        grp = tok.squeeze(1)
        g = torch.sigmoid(self.alpha)
        ps = (1 - g) * ps + g * grp.unsqueeze(1)
        past_seq = ps

        preds = []
        cur = past_seq.clone()
        fml = fmaps_last

        for _ in range(P_i):
            ps = cur.float()
            te = sinusoidal_encoding(ps.size(1), Config.compress_dim, device).unsqueeze(0).float()
            x_in = self.pre_ln(ps) + te
            x_in = self.dp_in(x_in)

            out_seq = self.encoder_temporal(x_in)
            out = out_seq[:, -1]

            out = out.to(ps.dtype)
            dec = self.decoder(out, fml)
            preds.append(dec[:, 0])

            nxt = torch.sigmoid(dec[:, 0]).detach()
            z_next, fml = self.encode_frame(nxt)
            cur = torch.cat([cur, z_next.unsqueeze(1)], dim=1)

        pred = torch.cat(preds, dim=0) if len(preds) else seq_i.new_zeros((0, seq_i.shape[1], seq_i.shape[2]))
        return pred, tgt

    def forward(self, seq, frame_mask):
        B, T, H, W = seq.shape
        preds_list, tgts_list, lens = [], [], []

        for i in range(B):
            L_i = int(frame_mask[i].sum().item())
            if L_i < 2:
                empty_pred = torch.zeros((1, H, W), device=seq.device)
                empty_tgt = torch.zeros((1, H, W), device=seq.device)
                preds_list.append(empty_pred)
                tgts_list.append(empty_tgt)
                lens.append(1)
                continue

            seq_i = seq[i, :L_i]
            pred_i, tgt_i = self._single_sequence_forward(seq_i)
            preds_list.append(pred_i)
            tgts_list.append(tgt_i)
            lens.append(pred_i.shape[0])

        Pmax = max(lens) if len(lens) else 1
        pred_batch, tgt_batch, pmask_batch = [], [], []
        for p, t, L in zip(preds_list, tgts_list, lens):
            if L < Pmax:
                padp = torch.zeros((Pmax - L, H, W), device=p.device)
                p = torch.cat([p, padp], dim=0)
                padt = torch.zeros((Pmax - L, H, W), device=t.device)
                t = torch.cat([t, padt], dim=0)
            pred_batch.append(p)
            tgt_batch.append(t)
            m = torch.zeros(Pmax, dtype=torch.bool, device=p.device)
            m[:L] = True
            pmask_batch.append(m)

        pred = torch.stack(pred_batch, dim=0)
        tgt = torch.stack(tgt_batch, dim=0)
        pmask = torch.stack(pmask_batch, dim=0)
        return pred, tgt, pmask

# ============================================================
# Fast batch metrics:
# - mIoU over valid predicted frames
# - mAP@[.50:.95] computed as mean of thresholded IoU hits
# ============================================================
def compute_batch_metrics(pred, tgt, pmask, thr_list=None):
    if thr_list is None:
        thr_list = [0.50 + 0.05 * i for i in range(10)]
    with torch.no_grad():
        # GPU path: avoid CPU sync and speed up evaluation.
        if Config.metrics_on_gpu and pred.is_cuda:
            pp = torch.sigmoid(pred)
            tb = tgt
            mk = pmask
            binp = (pp > 0.5)
            binl = (tb > 0.5)
            inter = (binp & binl).sum(dim=(2, 3)).float()
            union = (binp | binl).sum(dim=(2, 3)).float()
            both_empty = (binp.sum(dim=(2, 3)) == 0) & (binl.sum(dim=(2, 3)) == 0)
            iou = torch.where(both_empty, torch.ones_like(union), inter / torch.clamp(union, min=1))

            iou_list = []
            ap_hits = [0.0 for _ in range(len(thr_list))]
            ap_cnt = 0
            B, P = pred.shape[:2]
            for b in range(B):
                valid = mk[b]
                if valid.any():
                    ious_b = iou[b][valid]
                    iou_list.append(ious_b.mean())
                    for ti, thr in enumerate(thr_list):
                        ap_hits[ti] += (ious_b >= thr).float().mean()
                    ap_cnt += 1
            mean_iou = float(torch.stack(iou_list).mean().item()) if iou_list else 0.0
            aps = [(h / ap_cnt if ap_cnt > 0 else 0.0) for h in
                   [float(x.item()) if isinstance(x, torch.Tensor) else x for x in ap_hits]]
            mAP = float(sum(aps) / len(aps)) if aps else 0.0
            return mean_iou, mAP

        # CPU fallback.
        B, P, H, W = pred.shape
        pp = torch.sigmoid(pred).detach().cpu().numpy()
        tb = tgt.detach().cpu().numpy()
        mk = pmask.detach().cpu().numpy().astype(bool)

        binp = pp > 0.5
        binl = tb > 0.5

        inter = np.logical_and(binp, binl).sum(axis=(2, 3))
        union = np.logical_or(binp, binl).sum(axis=(2, 3))
        both_empty = (binp.sum(axis=(2, 3)) == 0) & (binl.sum(axis=(2, 3)) == 0)

        iou = np.where(both_empty, 1.0, inter / np.maximum(union, 1))
        iou_list, ap_hits, ap_cnt = [], [0] * len(thr_list), 0
        for b in range(B):
            valid = mk[b]
            if valid.any():
                ious_b = iou[b, valid]
                iou_list.append(ious_b.mean())
                for ti, thr in enumerate(thr_list):
                    ap_hits[ti] += (ious_b >= thr).mean()
                ap_cnt += 1
        mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
        aps = [(h / ap_cnt if ap_cnt > 0 else 0.0) for h in ap_hits]
        mAP = float(np.mean(aps)) if aps else 0.0
        return mean_iou, mAP

# ============================================================
# The remaining functions implement:
# - optional physical statistics for test (radii profiles, velocity, NAS, H2, TCI, angular correlations)
# - distance-based boundary metrics (HD, HD95, ASSD)
# - train/eval/test pipelines
# NOTE: These are kept intact; comments below focus on research-relevant logic.
# ============================================================
def _smooth_circular(v: np.ndarray, win_bins: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if win_bins <= 1:
        return v
    k = np.ones(int(win_bins), dtype=np.float32) / float(win_bins)
    p = int(win_bins) - 1
    vv = np.concatenate([v[-p:], v, v[:p]])
    out = np.convolve(vv, k, mode='valid')
    return out.astype(np.float32)

def _center_of_mass(mask_uint8):
    ys, xs = np.nonzero(mask_uint8)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())

def _radii_by_angle(mask_uint8, cx, cy, n_ang=72, smooth_win_deg=10.0):
    rr = np.zeros(n_ang, dtype=np.float32)
    if mask_uint8.max() == 0:
        return rr
    cnts, _ = cv2.findContours(mask_uint8.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) == 0:
        return rr
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

    xs, ys = cnt[:, 0], cnt[:, 1]
    ang = np.arctan2(ys - cy, xs - cx)
    ang[ang < 0] += 2 * np.pi
    rad = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    theta_grid = np.linspace(0, 2 * np.pi, num=n_ang, endpoint=False, dtype=np.float32)
    order = np.argsort(ang)
    ang_sorted = ang[order]
    rad_sorted = rad[order]
    ang_ext = np.concatenate([ang_sorted - 2 * np.pi, ang_sorted, ang_sorted + 2 * np.pi])
    rad_ext = np.concatenate([rad_sorted, rad_sorted, rad_sorted])
    rr = np.interp(theta_grid, ang_ext, rad_ext).astype(np.float32)

    if smooth_win_deg and smooth_win_deg > 0:
        bins_per_deg = n_ang / 360.0
        win_bins = max(1, int(round(smooth_win_deg * bins_per_deg)))
        rr = _smooth_circular(rr, win_bins)
    return rr

def _velocity_from_radii(r_cur, r_prev, dt=1.0):
    return (r_cur - r_prev) / float(dt)

def _anisotropy_index(v_theta, eps=1e-8):
    v = np.asarray(v_theta, dtype=np.float32)
    vpos = v[v > 0]
    if vpos.size < 3:
        return np.nan
    m = float(np.mean(vpos))
    s = float(np.std(vpos))
    if m <= eps:
        return np.nan
    return s / m

def tci_local_window_strict(v_gt, v_pr, eps=1e-8, tau0=1e-6):
    std_gt = np.std(v_gt, axis=0, ddof=1)
    std_pr = np.std(v_pr, axis=0, ddof=1)
    valid_mask = (std_gt + std_pr) > tau0
    if not np.any(valid_mask):
        return np.nan
    diff = np.abs(std_pr[valid_mask] - std_gt[valid_mask])
    denom = std_pr[valid_mask] + std_gt[valid_mask] + eps
    tci_valid = 1.0 - diff / denom
    tci_valid = np.clip(tci_valid, 0.0, 1.0)
    return float(np.mean(tci_valid))

def tci_strict_paper(r_gt, r_pr, dt=1.0, tau0=1e-6):
    T, A = r_gt.shape
    if T < 4:
        return np.nan
    window_tcis = []
    for t in range(T - 3):
        r_gt_w = r_gt[t:t + 4]
        r_pr_w = r_pr[t:t + 4]
        v_gt_w = np.diff(r_gt_w, axis=0) / dt
        v_pr_w = np.diff(r_pr_w, axis=0) / dt
        tci_w = tci_local_window_strict(v_gt_w, v_pr_w, tau0=tau0)
        if np.isfinite(tci_w):
            window_tcis.append(tci_w)
    if len(window_tcis) == 0:
        return np.nan
    return float(np.mean(window_tcis))

def _fourier_h2_energy(v_theta):
    vt = np.asarray(v_theta, dtype=np.float32)
    spec = np.fft.rfft(vt)
    power = (spec.real ** 2 + spec.imag ** 2)
    denom = power.sum()
    if denom <= 0:
        return np.nan
    return float(power[2] / denom) if len(power) > 2 else np.nan

def _rmse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def angular_shift_corr(v_gt, v_pr):
    gt = v_gt - np.mean(v_gt)
    pr = v_pr - np.mean(v_pr)
    corr = np.fft.irfft(np.fft.rfft(gt) * np.conj(np.fft.rfft(pr)))
    shift_idx = np.argmax(np.roll(corr, len(corr)//2))
    bins = len(v_gt)
    shift_deg = abs((shift_idx - bins//2) * 360.0 / bins)
    if shift_deg > 180:
        shift_deg = 360 - shift_deg
    return shift_deg

def angular_energy_corr(v_gt, v_pr):
    e_gt = np.abs(np.fft.rfft(v_gt))**2
    e_pr = np.abs(np.fft.rfft(v_pr))**2
    num = np.sum((e_gt - e_gt.mean()) * (e_pr - e_pr.mean()))
    den = np.sqrt(np.sum((e_gt - e_gt.mean())**2) * np.sum((e_pr - e_pr.mean())**2))
    return num / den if den > 1e-6 else 0.0

def _pearsonr_safe(a, b, eps=1e-8, zero_mask_thr=1e-3):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = ~((np.abs(a) < zero_mask_thr) & (np.abs(b) < zero_mask_thr))
    if not np.any(m):
        return np.nan
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= eps or nb <= eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def dynamic_corr(v_gt: np.ndarray, v_pr: np.ndarray, max_shift_deg: float = 10.0) -> float:
    n = v_gt.size
    if n < 4:
        return np.nan
    best_r = -np.inf
    max_shift_bins = int(round(max_shift_deg / (360.0 / n)))
    for s in range(-max_shift_bins, max_shift_bins + 1):
        pr_s = np.roll(v_pr, s)
        gt_c = v_gt - v_gt.mean()
        pr_c = pr_s - pr_s.mean()
        denom = np.sqrt((gt_c**2).sum() * (pr_c**2).sum()) + 1e-8
        if denom == 0:
            continue
        r = (gt_c * pr_c).sum() / denom
        if r > best_r:
            best_r = r
    return best_r if np.isfinite(best_r) else np.nan

def _ccc_weighted(a: np.ndarray, b: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)

    sum_w = float(w.sum())
    if not np.isfinite(sum_w) or sum_w <= eps:
        return np.nan
    w_norm = w / sum_w

    mu_a = float((w_norm * a).sum())
    mu_b = float((w_norm * b).sum())

    da = a - mu_a
    db = b - mu_b
    var_a = float((w_norm * (da * da)).sum())
    var_b = float((w_norm * (db * db)).sum())
    cov_ab = float((w_norm * (da * db)).sum())

    denom = var_a + var_b + (mu_a - mu_b) ** 2 + eps
    return float((2.0 * cov_ab) / denom)

def vcc_from_profiles(v_gt: np.ndarray,
                      v_pr: np.ndarray,
                      max_shift_bins: int = 6,
                      sigma_bins: int = 3,
                      zero_mask_thr: float = 1e-3,
                      weight_mode: str = "geo") -> tuple[float, int]:
    gt = np.asarray(v_gt, dtype=np.float32)
    pr = np.asarray(v_pr, dtype=np.float32)
    n = gt.size
    if n < 3 or n != pr.size:
        return np.nan, 0

    best_score = -np.inf
    best_ccc = np.nan

    if sigma_bins is None or sigma_bins <= 0:
        def _penalty(_):
            return 1.0
    else:
        def _penalty(s):
            return math.exp(-0.5 * (float(s) / float(sigma_bins)) ** 2)

    for s in range(-int(max_shift_bins), int(max_shift_bins) + 1):
        pr_s = np.roll(pr, s)
        valid = ~((np.abs(gt) < zero_mask_thr) & (np.abs(pr_s) < zero_mask_thr))
        if not np.any(valid):
            continue

        a = gt[valid]
        b = pr_s[valid]

        if weight_mode == "min":
            w = np.minimum(np.abs(a), np.abs(b))
        elif weight_mode == "absb":
            w = np.abs(b)
        elif weight_mode == "absa":
            w = np.abs(a)
        else:
            w = np.sqrt(np.abs(a) * np.abs(b))

        ccc = _ccc_weighted(a, b, w)
        score = ccc * _penalty(s)

        if score > best_score:
            best_score = score
            best_ccc = ccc * _penalty(s)

    if not np.isfinite(best_score):
        return np.nan, 0
    return float(best_ccc), 0

def _mask_to_surface(mask_uint8):
    if mask_uint8.max() == 0:
        return np.zeros_like(mask_uint8, dtype=np.uint8)
    eroded = ndi.binary_erosion(mask_uint8.astype(bool), structure=np.ones((3, 3), dtype=bool), border_value=0)
    surface = np.logical_xor(mask_uint8.astype(bool), eroded)
    return surface.astype(np.uint8)

def _hd95_assd_pair(pred_bin, true_bin):
    pred_has = pred_bin.max() > 0
    true_has = true_bin.max() > 0
    if not pred_has and not true_has:
        return 0.0, 0.0
    if pred_has and true_has:
        sp = _mask_to_surface(pred_bin)
        st = _mask_to_surface(true_bin)

        if st.any():
            dt_true = ndi.distance_transform_edt(1 - st)
            d_pt = dt_true[sp.astype(bool)]
        else:
            d_pt = np.array([], dtype=np.float64)
        if sp.any():
            dt_pred = ndi.distance_transform_edt(1 - sp)
            d_tp = dt_pred[st.astype(bool)]
        else:
            d_tp = np.array([], dtype=np.float64)

        if d_pt.size == 0 and d_tp.size == 0:
            return float('nan'), float('nan')

        all_d = np.concatenate([d_pt, d_tp]) if (d_pt.size and d_tp.size) else (d_pt if d_pt.size else d_tp)
        hd95 = float(np.percentile(all_d, 95))
        assd = float(all_d.mean())
        return hd95, assd
    return float('nan'), float('nan')

def compute_hd95_assd_over_batch(pred_logits, tgt_bin, pmask):
    if not Config.eval_compute_hd_assd:
        return float('nan'), float('nan'), [], []

    B, P, H, W = pred_logits.shape
    pp = (torch.sigmoid(pred_logits).detach().cpu().numpy() > 0.5).astype(np.uint8)
    tb = (tgt_bin.detach().cpu().numpy() > 0.5).astype(np.uint8)
    mk = pmask.detach().cpu().numpy().astype(bool)

    hd_list, as_list = [], []
    for b in range(B):
        valid = mk[b]
        for p in np.where(valid)[0]:
            hd95, assd = _hd95_assd_pair(pp[b, p], tb[b, p])
            hd_list.append(hd95)
            as_list.append(assd)
    mean_hd = float(np.nanmean(hd_list)) if len(hd_list) and np.any(~np.isnan(hd_list)) else float('nan')
    mean_as = float(np.nanmean(as_list)) if len(as_list) and np.any(~np.isnan(as_list)) else float('nan')
    return mean_hd, mean_as, hd_list, as_list

def _hd_pair(pred_bin, true_bin):
    pred_has = pred_bin.max() > 0
    true_has = true_bin.max() > 0
    if not pred_has and not true_has:
        return 0.0
    if pred_has and true_has:
        sp = _mask_to_surface(pred_bin)
        st = _mask_to_surface(true_bin)

        if st.any():
            dt_true = ndi.distance_transform_edt(1 - st)
            d_pt = dt_true[sp.astype(bool)]
        else:
            d_pt = np.array([], dtype=np.float64)
        if sp.any():
            dt_pred = ndi.distance_transform_edt(1 - sp)
            d_tp = dt_pred[st.astype(bool)]
        else:
            d_tp = np.array([], dtype=np.float64)

        if d_pt.size == 0 and d_tp.size == 0:
            return float('nan')

        if d_pt.size and d_tp.size:
            all_d = np.concatenate([d_pt, d_tp])
        else:
            all_d = d_pt if d_pt.size else d_tp
        return float(np.max(all_d))
    return float('nan')

def compute_hd_over_batch(pred_logits, tgt_bin, pmask):
    if not Config.eval_compute_hd_assd:
        return float('nan'), []

    B, P, H, W = pred_logits.shape
    pp = (torch.sigmoid(pred_logits).detach().cpu().numpy() > 0.5).astype(np.uint8)
    tb = (tgt_bin.detach().cpu().numpy() > 0.5).astype(np.uint8)
    mk = pmask.detach().cpu().numpy().astype(bool)

    hd_list = []
    for b in range(B):
        valid = mk[b]
        for p in np.where(valid)[0]:
            hd = _hd_pair(pp[b, p], tb[b, p])
            hd_list.append(hd)
    mean_hd = float(np.nanmean(hd_list)) if len(hd_list) and np.any(~np.isnan(hd_list)) else float('nan')
    return mean_hd, hd_list

# ============================================================
# Evaluation loop (validation)
# - Computes average loss components + metrics across val loader
# ============================================================
def evaluate(model, loader):
    model.eval()
    tot_loss = tot_fl = tot_si = tot_bi = 0.0
    miou_list, map_list = [], []
    hd_list_all = []
    hd95_list_all = []
    assd_list_all = []

    with torch.no_grad():
        for seq, fmask in loader:
            seq = seq.to(Config.device, non_blocking=True)
            fmask = fmask.to(Config.device, non_blocking=True)
            if Config.device.type == 'cuda':
                seq = seq.half()
            with autocast(enabled=(Config.device.type == 'cuda')):
                pred, tgt, pmask = model(seq, fmask)
                loss, fl, si, bi = total_loss_fn(pred, tgt, pmask)

            tot_loss += float(loss.item())
            tot_fl += float(fl.item())
            tot_si += float(si.item())
            tot_bi += float(bi.item())

            miou, mAP = compute_batch_metrics(pred, tgt, pmask)
            miou_list.append(miou)
            map_list.append(mAP)

            mean_hd95, mean_as, _, _ = compute_hd95_assd_over_batch(pred, tgt, pmask)
            if not math.isnan(mean_hd95):
                hd95_list_all.append(mean_hd95)
            if not math.isnan(mean_as):
                assd_list_all.append(mean_as)

            mean_hd, _ = compute_hd_over_batch(pred, tgt, pmask)
            if not math.isnan(mean_hd):
                hd_list_all.append(mean_hd)

    n = len(loader)
    return (
        tot_loss / n,
        tot_fl / n,
        tot_si / n,
        tot_bi / n,
        float(np.mean(miou_list)) if miou_list else 0.0,
        float(np.mean(map_list)) if map_list else 0.0,
        float(np.mean(hd_list_all)) if hd_list_all else float('nan'),
        float(np.mean(hd95_list_all)) if hd95_list_all else float('nan'),
        float(np.mean(assd_list_all)) if assd_list_all else float('nan'),
    )

# ============================================================
# Training + validation entry
# - Builds model by `arch`
# - Trains for epochs_per_fold epochs
# - Saves best model by validation mIoU
# ============================================================
def run_trainval(arch="gru", save_name="best.pth", save_path=None, log_csv=None):
    arch_l = arch.lower()

    if arch_l == 'transformer':
        model = MorpherTransformer().to(Config.device)
    elif arch_l == 'lstm':
        model = MorpherLSTM().to(Config.device)
    elif arch_l == 'rnn':
        model = MorpherRNN().to(Config.device)
    else:
        model = Morpher().to(Config.device)

    tr_ds = MultiSequenceDataset(Config.train_path, step=Config.step, augment=True)
    val_ds = MultiSequenceDataset(Config.val_path, step=Config.step, augment=False)

    trL = DataLoader(
        tr_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=Config.num_workers_train,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=Config.prefetch_factor_train
    )

    valL = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=Config.num_workers_val,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=Config.prefetch_factor_val
    )

    if Config.device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    if Config.use_torch_compile:
        try:
            model = torch.compile(model, mode=Config.torch_compile_mode)
        except Exception:
            pass

    opt = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(Config.device.type == 'cuda'))

    total_steps = Config.epochs_per_fold * len(trL)
    warmup_steps = int(total_steps * Config.warmup_ratio)

    # Cosine schedule with linear warmup.
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = LambdaLR(opt, lr_lambda)

    best_miou = -1.0
    os.makedirs(Config.results_dir, exist_ok=True)

    if save_path is None:
        save_path = os.path.join(Config.results_dir, save_name)

    csv_f = None
    csv_w = None
    if log_csv is not None:
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        csv_f = open(log_csv, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["epoch", "lr", "train_loss", "train_mIoU", "val_loss", "val_mIoU", "mAP_50_95", "HD", "HD95", "ASSD"])

    step = 0
    for ep in range(1, Config.epochs_per_fold + 1):
        model.train()
        sum_loss = 0.0
        train_mious = []

        for seq, fmask in tqdm(trL, desc=f"Train Ep{ep}"):
            seq = seq.to(Config.device, non_blocking=True)
            fmask = fmask.to(Config.device, non_blocking=True)
            if Config.device.type == 'cuda':
                seq = seq.half()

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(Config.device.type == 'cuda')):
                pred, tgt, pmask = model(seq, fmask)
                loss, fl, si, bi = total_loss_fn(pred, tgt, pmask)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            sched.step()
            step += 1

            sum_loss += float(loss.item())

            miou, _ = compute_batch_metrics(pred, tgt, pmask)
            train_mious.append(miou)

        train_loss = sum_loss / len(trL)
        train_miou = float(np.mean(train_mious)) if len(train_mious) else 0.0

        v = evaluate(model, valL)
        if v[4] > best_miou:
            best_miou = v[4]
            torch.save(model.state_dict(), save_path)
            print(f"→ Saved best mIoU={best_miou:.4f} to {save_path}")

        lr_cur = opt.param_groups[0]['lr']
        print(f"Ep{ep} lr={lr_cur:.2e} TrainLoss={train_loss:.4f} Train_mIoU={train_miou:.4f} "
              f"ValLoss={v[0]:.4f} Val_mIoU={v[4]:.4f} mAP@[.50:.95]={v[5]:.4f} "
              f"HD={v[6]:.4f} HD95={v[7]:.4f} ASSD={v[8]:.4f}")

        if csv_w is not None:
            csv_w.writerow([ep, lr_cur, train_loss, train_miou, v[0], v[4], v[5], v[6], v[7], v[8]])
            csv_f.flush()

    if csv_f is not None:
        csv_f.close()

def _resample_angle_uniform(v_theta: np.ndarray, target_bins: int) -> np.ndarray:
    v = np.asarray(v_theta, dtype=np.float32).reshape(-1)
    n = v.size
    if target_bins <= 0:
        return v.copy()
    if n == target_bins:
        return v.copy()
    x_old = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_bins, endpoint=False, dtype=np.float32)
    v_ext = np.concatenate([v, v[:1]], axis=0)
    x_ext = np.concatenate([x_old, np.array([1.0], dtype=np.float32)], axis=0)
    out = np.interp(x_new, x_ext, v_ext).astype(np.float32)
    return out

# ============================================================
# Test entry (loads weights, runs inference, writes CSV outputs)
# - phys_stats=True enables additional per-sequence & per-frame statistics
# ============================================================
def run_test(weights_path, out_csv=None, arch='gru', phys_stats=False, dt=1.0, burst_eval_bins=36):
    import csv as _csv
    import os as _os
    import math as _math

    save_dir = _os.path.join(Config.results_dir, "test_outputs")
    _os.makedirs(save_dir, exist_ok=True)

    csv_path = _os.path.join(save_dir, "frame_metrics.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = _csv.writer(csv_f)

    csv_w.writerow([
        "sample_idx",
        "pred_frame_idx",
        "IoU",
        "HD95",
        "ASSD",
        "vel_RMSE",
        "AI_dev",
        "H2_dev"
    ])

    if out_csv is None:
        out_csv = _os.path.join(Config.results_dir, "test_metrics.csv")
    _os.makedirs(_os.path.dirname(out_csv), exist_ok=True)

    ds = MultiSequenceDataset(Config.val_path, step=Config.step, augment=False)
    dl = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=pad_collate,
        num_workers=Config.num_workers_val, pin_memory=True,
        persistent_workers=False, prefetch_factor=Config.prefetch_factor_val
    )

    arch_l = arch.lower()

    if arch_l == 'transformer':
        model = MorpherTransformer().to(Config.device)
    elif arch_l == 'lstm':
        model = MorpherLSTM().to(Config.device)
    elif arch_l == 'rnn':
        model = MorpherRNN().to(Config.device)
    else:
        model = Morpher().to(Config.device)

    if Config.device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    state = torch.load(weights_path, map_location=Config.device)

    # (Per your request) This remapping utility is left without extra comments.
    def remap_group_to_morphon(state):
        new_state = {}
        for k, v in state.items():
            nk = k
            nk = nk.replace("attn_mlp.", "morphon_score_mlp.")
            nk = nk.replace("query_mlp.", "morphon_query_mlp.")
            nk = nk.replace("group_attn.", "morphon_attn.")
            new_state[nk] = v
        return new_state

    state = remap_group_to_morphon(state)
    model.load_state_dict(state, strict=True)

    if Config.use_torch_compile:
        try:
            model = torch.compile(model, mode=Config.torch_compile_mode)
        except Exception:
            pass

    model.eval()

    rows, all_miou, all_hd, all_hd95, all_assd = [], [], [], [], []

    g_vel_rmse_num = g_vel_rmse_den = 0
    g_nas_err_num = g_nas_err_den = 0
    g_h2_err_num = g_h2_err_den = 0
    g_tci_num = g_tci_den = 0
    g_shift_deg_vals = []
    g_energy_corr_vals = []

    with torch.no_grad():
        for idx, (seq, fmask) in enumerate(tqdm(dl, desc="Test")):
            seq = seq.to(Config.device, non_blocking=True)
            fmask = fmask.to(Config.device, non_blocking=True)
            if Config.device.type == 'cuda':
                seq = seq.half()

            with autocast(enabled=(Config.device.type == 'cuda')):
                pred, tgt, pmask = model(seq, fmask)

            miou, _ = compute_batch_metrics(pred, tgt, pmask)
            all_miou.append(miou)

            mean_hd95, mean_as, _, _ = compute_hd95_assd_over_batch(pred, tgt, pmask)
            all_hd95.append(mean_hd95)
            all_assd.append(mean_as)

            mean_hd, _ = compute_hd_over_batch(pred, tgt, pmask)
            all_hd.append(mean_hd)

            rows.append([idx, int(fmask.sum().item()), miou,
                         ("" if _math.isnan(mean_hd) else mean_hd),
                         ("" if _math.isnan(mean_hd95) else mean_hd95),
                         ("" if _math.isnan(mean_as) else mean_as)])

            pp = (torch.sigmoid(pred).detach().cpu().numpy() > 0.5).astype(np.uint8)[0]
            gg = (tgt.detach().cpu().numpy() > 0.5).astype(np.uint8)[0]
            mk = pmask.detach().cpu().numpy().astype(bool)[0]

            if phys_stats:
                vel_rmse_list = []
                NAS_abs_err_list = []
                h2_abs_err_list = []
                angular_shift_list = []
                angular_energy_list = []

                EVAL_BINS = int(burst_eval_bins)
                RAW_BINS = EVAL_BINS

                radii_gt, radii_pr = [], []
                for p_i, valid in enumerate(mk):
                    if not valid:
                        radii_gt.append(None)
                        radii_pr.append(None)
                        continue
                    cxcy = _center_of_mass(gg[p_i])
                    if cxcy is None:
                        radii_gt.append(None)
                        radii_pr.append(None)
                        continue
                    cx, cy = cxcy
                    r_gt_raw = _radii_by_angle(gg[p_i], cx, cy, n_ang=RAW_BINS)
                    r_pr_raw = _radii_by_angle(pp[p_i], cx, cy, n_ang=RAW_BINS)

                    if RAW_BINS != EVAL_BINS:
                        r_gt = _resample_angle_uniform(r_gt_raw, EVAL_BINS)
                        r_pr = _resample_angle_uniform(r_pr_raw, EVAL_BINS)
                    else:
                        r_gt, r_pr = r_gt_raw, r_pr_raw
                    radii_gt.append(r_gt)
                    radii_pr.append(r_pr)

                bins_per_deg = EVAL_BINS / 360.0

                for j in range(1, len(radii_gt)):
                    if not mk[j] or not mk[j - 1]:
                        continue
                    if radii_gt[j] is None or radii_gt[j - 1] is None:
                        continue

                    if (j + 1) < len(radii_gt) and mk[j + 1] and (radii_gt[j + 1] is not None):
                        v_gt = _velocity_from_radii(radii_gt[j + 1], radii_gt[j - 1], dt=2 * dt)
                    else:
                        v_gt = _velocity_from_radii(radii_gt[j], radii_gt[j - 1], dt=dt)

                    if (radii_pr[j] is None) or (radii_pr[j - 1] is None):
                        continue
                    if (j + 1) < len(radii_pr) and (radii_pr[j + 1] is not None):
                        v_pr = _velocity_from_radii(radii_pr[j + 1], radii_pr[j - 1], dt=2 * dt)
                    else:
                        v_pr = _velocity_from_radii(radii_pr[j], radii_pr[j - 1], dt=dt)

                    def _winsorize(x, p=0.01):
                        x = np.asarray(x, dtype=np.float32)
                        lo, hi = np.quantile(x, p), np.quantile(x, 1 - p)
                        return np.clip(x, lo, hi)

                    v_gt = _winsorize(v_gt, 0.01)
                    v_pr = _winsorize(v_pr, 0.01)

                    max_shift_bins = max(0, int(round(20.0 * bins_per_deg)))
                    sigma_bins = max(1, int(round(8.0 * bins_per_deg)))

                    vel_rmse_list.append(_rmse(v_pr, v_gt))

                    ai_gt = _anisotropy_index(v_gt)
                    ai_pr = _anisotropy_index(v_pr)
                    if not np.isnan(ai_gt) and not np.isnan(ai_pr):
                        NAS_abs_err_list.append(abs(ai_pr - ai_gt))

                    h2_gt = _fourier_h2_energy(v_gt)
                    h2_pr = _fourier_h2_energy(v_pr)
                    if not np.isnan(h2_gt) and not np.isnan(h2_pr):
                        h2_abs_err_list.append(abs(h2_pr - h2_gt))

                    try:
                        shift_deg = angular_shift_corr(v_gt, v_pr)
                        if np.isfinite(shift_deg):
                            angular_shift_list.append(float(shift_deg))
                    except Exception:
                        pass
                    try:
                        energy_corr = angular_energy_corr(v_gt, v_pr)
                        if np.isfinite(energy_corr):
                            angular_energy_list.append(float(energy_corr))
                    except Exception:
                        pass

                valid_idx = [
                    t for t, ok in enumerate(mk)
                    if ok and (radii_gt[t] is not None) and (radii_pr[t] is not None)
                ]

                if len(valid_idx) >= 4:
                    r_gt_seq = np.stack([radii_gt[t] for t in valid_idx], axis=0)
                    r_pr_seq = np.stack([radii_pr[t] for t in valid_idx], axis=0)
                    seq_tci = tci_strict_paper(r_gt_seq, r_pr_seq, dt=dt)
                else:
                    seq_tci = np.nan

                seq_vel_rmse = float(np.mean(vel_rmse_list)) if len(vel_rmse_list) else ""

                seq_nas_err = float(np.mean(NAS_abs_err_list)) if len(NAS_abs_err_list) else ""
                seq_h2_err = float(np.mean(h2_abs_err_list)) if len(h2_abs_err_list) else ""

                seq_shift_mean = float(np.mean(angular_shift_list)) if len(angular_shift_list) else ""
                seq_energy_mean = float(np.mean(angular_energy_list)) if len(angular_energy_list) else ""

                if isinstance(seq_tci, float) and np.isfinite(seq_tci):
                    seq_tci_out = float(seq_tci)
                else:
                    seq_tci_out = ""

                rows[-1].extend([
                    seq_vel_rmse,
                    seq_nas_err,
                    seq_h2_err,
                    seq_shift_mean,
                    seq_energy_mean,
                    seq_tci_out
                ])

                num_windows = max(0, len(valid_idx) - 3)
                if isinstance(seq_tci, float) and np.isfinite(seq_tci):
                    g_tci_num += seq_tci * num_windows
                    g_tci_den += num_windows

                if isinstance(seq_vel_rmse, float):
                    g_vel_rmse_num += seq_vel_rmse * len(vel_rmse_list)
                    g_vel_rmse_den += len(vel_rmse_list)

                if isinstance(seq_nas_err, float):
                    g_nas_err_num += seq_nas_err * len(NAS_abs_err_list)
                    g_nas_err_den += len(NAS_abs_err_list)

                if isinstance(seq_h2_err, float):
                    g_h2_err_num += seq_h2_err * len(h2_abs_err_list)
                    g_h2_err_den += len(h2_abs_err_list)

                if isinstance(seq_shift_mean, float):
                    g_shift_deg_vals.append(seq_shift_mean)

                if isinstance(seq_energy_mean, float):
                    g_energy_corr_vals.append(seq_energy_mean)


                for p_i, valid in enumerate(mk):
                    if not valid:
                        continue

                    pred_mask = pp[p_i]
                    gt_mask = gg[p_i]

                    inter = np.logical_and(pred_mask, gt_mask).sum()
                    union = np.logical_or(pred_mask, gt_mask).sum()
                    iou = 1.0 if union == 0 else inter / union

                    hd95, assd = _hd95_assd_pair(pred_mask, gt_mask)

                    vel_rmse = vel_rmse_list[p_i - 1] if p_i > 0 and p_i - 1 < len(vel_rmse_list) else ""
                    ai_dev = NAS_abs_err_list[p_i - 1] if p_i > 0 and p_i - 1 < len(NAS_abs_err_list) else ""
                    h2_dev = h2_abs_err_list[p_i - 1] if p_i > 0 and p_i - 1 < len(h2_abs_err_list) else ""

                    csv_w.writerow([
                        idx,
                        p_i,
                        iou,
                        "" if _math.isnan(hd95) else hd95,
                        "" if _math.isnan(assd) else assd,
                        vel_rmse,
                        ai_dev,
                        h2_dev
                    ])

    csv_f.close()
    print("[Frame CSV saved to]", csv_path)

    mean_miou = float(np.mean(all_miou)) if all_miou else 0.0
    mean_hd = float(np.nanmean(all_hd)) if np.any(~np.isnan(all_hd)) else float('nan')
    mean_hd95 = float(np.nanmean(all_hd95)) if np.any(~np.isnan(all_hd95)) else float('nan')
    mean_assd = float(np.nanmean(all_assd)) if np.any(~np.isnan(all_assd)) else float('nan')

    g_vel_rmse_mean = (g_vel_rmse_num / g_vel_rmse_den) if g_vel_rmse_den > 0 else float('nan')
    g_nas_err_mean = (g_nas_err_num / g_nas_err_den) if g_nas_err_den > 0 else float('nan')
    g_h2_err_mean = (g_h2_err_num / g_h2_err_den) if g_h2_err_den > 0 else float('nan')
    g_shift_mean = float(np.mean(g_shift_deg_vals)) if len(g_shift_deg_vals) else float('nan')
    g_energy_mean = float(np.mean(g_energy_corr_vals)) if len(g_energy_corr_vals) else float('nan')
    g_tci_mean = (g_tci_num / g_tci_den) if g_tci_den > 0 else float('nan')

    with open(out_csv, 'w', newline='', encoding="utf-8") as f:
        w = _csv.writer(f)
        header = ["sample_idx", "valid_pred_frames", "mIoU", "HD(px)", "HD95(px)", "ASSD(px)"]
        if phys_stats:
            header += ["vel_RMSE", "NAS_abs_err", "H2_abs_err",
                       "AngularShift_deg", "AngularEnergyCorr", "TCI"]
        w.writerow(header)

        for r in rows:
            w.writerow(r)

        w.writerow([])
        w.writerow(["mean_mIoU", mean_miou])
        w.writerow(["mean_HD(px)", mean_hd if not _math.isnan(mean_hd) else "NaN"])
        w.writerow(["mean_HD95(px)", mean_hd95 if not _math.isnan(mean_hd95) else "NaN"])
        w.writerow(["mean_ASSD(px)", mean_assd if not _math.isnan(mean_assd) else "NaN"])
        if phys_stats:
            w.writerow(["mean_vel_RMSE", g_vel_rmse_mean if not _math.isnan(g_vel_rmse_mean) else "NaN"])
            w.writerow(["mean_TCI", g_tci_mean if not _math.isnan(g_tci_mean) else "NaN"])
            w.writerow(["mean_NAS_abs_err", g_nas_err_mean if not _math.isnan(g_nas_err_mean) else "NaN"])
            w.writerow(["mean_H2_abs_err", g_h2_err_mean if not _math.isnan(g_h2_err_mean) else "NaN"])
            w.writerow(["mean_AngularShift_deg", g_shift_mean if not _math.isnan(g_shift_mean) else "NaN"])
            w.writerow(["mean_AngularEnergyCorr", g_energy_mean if not _math.isnan(g_energy_mean) else "NaN"])

    if phys_stats:
        print(
            f"[Test Done] mIoU={mean_miou:.6f}  HD={mean_hd:.6f}  HD95={mean_hd95:.6f}  ASSD={mean_assd:.6f}  "
            f"vel_RMSE={g_vel_rmse_mean:.6f}  TCI={g_tci_mean:.6f}  "
            f"NAS_abs_err={g_nas_err_mean:.6f}  H2_abs_err={g_h2_err_mean:.6f}"
        )

# ============================================================
# CLI: train / test
# (Utility plumbing; kept minimally commented as requested.)
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm", "rnn", "transformer"])
    tr.add_argument("--train_path", type=str, default=Config.train_path)
    tr.add_argument("--val_path", type=str, default=Config.val_path)
    tr.add_argument("--img_size", type=int, default=Config.img_size)
    tr.add_argument("--step", type=int, default=Config.step)
    tr.add_argument("--obs_ratio", type=float, default=Config.obs_ratio)
    tr.add_argument("--batch_size", type=int, default=Config.batch_size)
    tr.add_argument("--epochs", type=int, default=Config.epochs_per_fold)
    tr.add_argument("--lr", type=float, default=Config.lr)
    tr.add_argument("--device", type=str, default=str(Config.device))
    tr.add_argument("--results_dir", type=str, default=Config.results_dir)
    tr.add_argument("--save_name", type=str, default="best.pth")
    tr.add_argument("--save_path", type=str, default=None)
    tr.add_argument("--log_csv", type=str, default=None)
    tr.add_argument("--torch_compile", action="store_true", default=Config.use_torch_compile)
    tr.add_argument("--torch_compile_mode", type=str, default=Config.torch_compile_mode)

    te = sub.add_parser("test")
    te.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm", "rnn", "transformer"])
    te.add_argument("--weights", type=str, required=True)
    te.add_argument("--test_path", type=str, default=Config.val_path)
    te.add_argument("--img_size", type=int, default=Config.img_size)
    te.add_argument("--step", type=int, default=Config.step)
    te.add_argument("--obs_ratio", type=float, default=Config.obs_ratio)
    te.add_argument("--device", type=str, default=str(Config.device))
    te.add_argument("--results_dir", type=str, default=Config.results_dir)
    te.add_argument("--out_csv", type=str, default=None)
    te.add_argument("--phys_stats", action="store_true")
    te.add_argument("--dt", type=float, default=1.0)
    te.add_argument("--burst_eval_bins", type=int, default=36)
    te.add_argument("--torch_compile", action="store_true", default=Config.use_torch_compile)
    te.add_argument("--torch_compile_mode", type=str, default=Config.torch_compile_mode)

    return p

def apply_runtime_args(args):
    if hasattr(args, "results_dir") and args.results_dir is not None:
        Config.results_dir = args.results_dir

    if hasattr(args, "device") and args.device is not None:
        Config.device = torch.device(args.device)

    if hasattr(args, "img_size") and args.img_size is not None:
        Config.img_size = int(args.img_size)
        Config.pool_size = Config.img_size // 32

    if hasattr(args, "step") and args.step is not None:
        Config.step = int(args.step)

    if hasattr(args, "obs_ratio") and args.obs_ratio is not None:
        Config.obs_ratio = float(args.obs_ratio)

    if hasattr(args, "batch_size") and args.batch_size is not None:
        Config.batch_size = int(args.batch_size)

    if hasattr(args, "epochs") and args.epochs is not None:
        Config.epochs_per_fold = int(args.epochs)

    if hasattr(args, "lr") and args.lr is not None:
        Config.lr = float(args.lr)

    if hasattr(args, "train_path") and args.train_path is not None:
        Config.train_path = args.train_path

    if hasattr(args, "val_path") and args.val_path is not None:
        Config.val_path = args.val_path

    if hasattr(args, "test_path") and args.test_path is not None:
        Config.val_path = args.test_path

    if hasattr(args, "torch_compile") and args.torch_compile is not None:
        Config.use_torch_compile = bool(args.torch_compile)

    if hasattr(args, "torch_compile_mode") and args.torch_compile_mode is not None:
        Config.torch_compile_mode = str(args.torch_compile_mode)

def main():
    parser = build_argparser()
    args = parser.parse_args()
    apply_runtime_args(args)

    if args.cmd == "train":
        run_trainval(arch=args.arch, save_name=args.save_name, save_path=args.save_path, log_csv=args.log_csv)
    elif args.cmd == "test":
        run_test(
            weights_path=args.weights,
            out_csv=args.out_csv,
            arch=args.arch,
            phys_stats=args.phys_stats,
            dt=args.dt,
            burst_eval_bins=args.burst_eval_bins
        )

if __name__ == "__main__":
    main()
