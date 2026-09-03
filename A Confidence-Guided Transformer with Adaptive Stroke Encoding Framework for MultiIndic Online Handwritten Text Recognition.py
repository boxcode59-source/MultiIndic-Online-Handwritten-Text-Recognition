"""
Confidence-Guided Transformer with Adaptive Stroke Encoding Framework
for Multi-Indic Online Handwritten Text Recognition

This is a full working implementation of the proposed framework with
simplified components for demonstration purposes. Real datasets and
advanced architectures can be plugged in where indicated.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import math
import time
from collections import defaultdict

# ------------------------------
# Step 1: Online Handwriting Data Acquisition
# ------------------------------

class OnlineHandwritingDataset(Dataset):
    """
    Dataset class for online handwriting trajectories.
    If UCI digits dataset is available, it loads it; otherwise generates
    synthetic random trajectories for demonstration.
    """
    def __init__(self, num_samples=100, max_len=100, use_uci=False):
        self.use_uci = use_uci
        if use_uci:
            # Attempt to load UCI Pen-Based Recognition of Handwritten Digits
            try:
                from ucimlrepo import fetch_ucirepo
                pen_digits = fetch_ucirepo(id=81)
                X = pen_digits.data.features
                y = pen_digits.data.targets
                # Convert to trajectory format (simplified)
                self.trajectories = []
                self.labels = []
                for i in range(len(X)):
                    traj = self._uci_to_trajectory(X.iloc[i])
                    self.trajectories.append(traj)
                    self.labels.append(int(y.iloc[i]))
            except ImportError:
                print("ucimlrepo not installed. Using synthetic data.")
                self.use_uci = False
        if not self.use_uci:
            # Generate synthetic trajectories: each sample is a sequence of strokes
            self.trajectories = [self._generate_synthetic_trajectory(max_len) for _ in range(num_samples)]
            self.labels = np.random.randint(0, 10, num_samples)  # 10 digit classes
    
    def _uci_to_trajectory(self, feature_row):
        # The UCI dataset has 16 features per sample (8 pairs of x,y coordinates).
        # We treat them as a simple stroke sequence.
        coords = []
        for i in range(8):
            x = feature_row[f'x{i+1}']
            y = feature_row[f'y{i+1}']
            coords.append((x, y))
        # Add pen-up events between coordinates (simulate multiple strokes)
        traj = []
        for i, (x, y) in enumerate(coords):
            traj.append({'x': x, 'y': y, 'pressure': 1.0, 'pen_up': i > 0})
        return traj
    
    def _generate_synthetic_trajectory(self, max_len):
        """Generate a random sequence of points with pen-up/down events."""
        length = np.random.randint(5, max_len)
        traj = []
        pen_up = False
        for i in range(length):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            pressure = np.random.uniform(0.5, 1.0)
            if i > 0 and np.random.rand() > 0.7:
                pen_up = True
            else:
                pen_up = False
            traj.append({'x': x, 'y': y, 'pressure': pressure, 'pen_up': pen_up})
        return traj
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]

# ------------------------------
# Step 2: Trajectory Preprocessing
# ------------------------------

def remove_duplicate_points(traj):
    """Remove consecutive duplicate points."""
    if not traj:
        return traj
    new_traj = [traj[0]]
    for pt in traj[1:]:
        if pt['x'] != new_traj[-1]['x'] or pt['y'] != new_traj[-1]['y']:
            new_traj.append(pt)
    return new_traj

def savitzky_golay_smooth(traj, window_size=5, poly_order=2):
    """Apply Savitzky-Golay filter to x and y coordinates."""
    if len(traj) < window_size:
        return traj
    x = np.array([pt['x'] for pt in traj])
    y = np.array([pt['y'] for pt in traj])
    x_smooth = savgol_filter(x, window_size, poly_order)
    y_smooth = savgol_filter(y, window_size, poly_order)
    for i, pt in enumerate(traj):
        pt['x'] = x_smooth[i]
        pt['y'] = y_smooth[i]
    return traj

def min_max_normalize(traj):
    """Normalize coordinates to [0,1] range."""
    xs = [pt['x'] for pt in traj]
    ys = [pt['y'] for pt in traj]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x = min_x + 1e-6
    if max_y == min_y:
        max_y = min_y + 1e-6
    for pt in traj:
        pt['x'] = (pt['x'] - min_x) / (max_x - min_x)
        pt['y'] = (pt['y'] - min_y) / (max_y - min_y)
    return traj

def linear_interpolation_resample(traj, target_points=50):
    """Resample trajectory to a fixed number of points using linear interpolation."""
    if len(traj) < 2:
        return traj
    # Compute cumulative distance along the trajectory
    xs = np.array([pt['x'] for pt in traj])
    ys = np.array([pt['y'] for pt in traj])
    pressures = np.array([pt['pressure'] for pt in traj])
    pen_ups = np.array([pt['pen_up'] for pt in traj])
    distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    cum_dist = np.concatenate([[0], np.cumsum(distances)])
    total_dist = cum_dist[-1]
    if total_dist == 0:
        return traj
    # New sample points
    new_dist = np.linspace(0, total_dist, target_points)
    # Interpolate x, y, pressure
    interp_x = interp1d(cum_dist, xs, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(cum_dist, ys, kind='linear', fill_value='extrapolate')
    interp_p = interp1d(cum_dist, pressures, kind='linear', fill_value='extrapolate')
    new_x = interp_x(new_dist)
    new_y = interp_y(new_dist)
    new_p = interp_p(new_dist)
    # Determine pen_up for new points (nearest neighbor)
    pen_up_idx = np.searchsorted(cum_dist, new_dist, side='left')
    pen_up_idx = np.clip(pen_up_idx, 0, len(pen_ups)-1)
    new_pen_up = pen_ups[pen_up_idx]
    new_traj = []
    for i in range(target_points):
        new_traj.append({'x': new_x[i], 'y': new_y[i], 'pressure': new_p[i], 'pen_up': bool(new_pen_up[i])})
    return new_traj

def preprocess_trajectory(traj):
    """Apply all preprocessing steps."""
    traj = remove_duplicate_points(traj)
    traj = savitzky_golay_smooth(traj)
    traj = min_max_normalize(traj)
    traj = linear_interpolation_resample(traj)
    return traj

# ------------------------------
# Step 3: Stroke Segmentation
# ------------------------------

def segment_strokes_penup(traj):
    """Segment strokes based on pen-up/pen-down events."""
    strokes = []
    current_stroke = []
    for pt in traj:
        if pt['pen_up'] and current_stroke:
            strokes.append(current_stroke)
            current_stroke = []
        current_stroke.append(pt)
    if current_stroke:
        strokes.append(current_stroke)
    return strokes

def segment_strokes_velocity(traj, threshold=0.1):
    """Segment strokes based on velocity change (for datasets without pen info)."""
    strokes = []
    current_stroke = [traj[0]]
    for i in range(1, len(traj)):
        prev = traj[i-1]
        curr = traj[i]
        vel = np.sqrt((curr['x']-prev['x'])**2 + (curr['y']-prev['y'])**2)
        if vel > threshold:
            current_stroke.append(curr)
        else:
            if len(current_stroke) > 1:
                strokes.append(current_stroke)
            current_stroke = [curr]
    if current_stroke:
        strokes.append(current_stroke)
    return strokes

def segment_trajectory(traj, use_penup=True):
    if use_penup:
        return segment_strokes_penup(traj)
    else:
        return segment_strokes_velocity(traj)

# ------------------------------
# Step 4: Dynamic Feature Extraction
# ------------------------------

def compute_features(stroke):
    """Compute spatial, velocity, direction, curvature, pressure variation, temporal features."""
    if len(stroke) < 2:
        return None
    xs = np.array([pt['x'] for pt in stroke])
    ys = np.array([pt['y'] for pt in stroke])
    pressures = np.array([pt['pressure'] for pt in stroke])
    # Spatial coordinates (already normalized)
    coords = np.stack([xs, ys], axis=1)
    # Velocity (first difference)
    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = 1.0  # assuming uniform time steps after resampling
    velocities = np.sqrt(dx**2 + dy**2) / dt
    # Writing direction (angle)
    directions = np.arctan2(dy, dx)
    # Curvature (change in direction)
    curvatures = np.diff(directions)
    # Pressure variation (first difference)
    pressure_var = np.diff(pressures)
    # Temporal features (time index normalized)
    temporal = np.linspace(0, 1, len(stroke))
    # Combine all features per point (excluding first point for derivatives)
    # We'll create a feature vector for each point except the first
    features = []
    for i in range(1, len(stroke)):
        feat = [
            xs[i], ys[i],                     # spatial
            velocities[i-1],                  # velocity
            directions[i-1],                  # direction
            curvatures[i-1] if i>1 else 0.0,  # curvature (0 for second point)
            pressure_var[i-1],                # pressure variation
            temporal[i]                       # temporal
        ]
        features.append(feat)
    return np.array(features)  # shape (L-1, 7)

def extract_dynamic_features(strokes):
    """Extract features for all strokes and pad to a fixed length."""
    all_features = []
    for stroke in strokes:
        feat = compute_features(stroke)
        if feat is not None:
            all_features.append(feat)
    if not all_features:
        return np.zeros((1, 7))
    # Pad/truncate each stroke to max stroke length (e.g., 50)
    max_len = max(f.shape[0] for f in all_features)
    padded = []
    for f in all_features:
        if f.shape[0] < max_len:
            pad = np.zeros((max_len - f.shape[0], f.shape[1]))
            f = np.vstack([f, pad])
        else:
            f = f[:max_len]
        padded.append(f)
    # Stack strokes: shape (num_strokes, max_len, feature_dim)
    return np.stack(padded)

# ------------------------------
# Step 5: Adaptive Stroke Tokenization (AST)
# ------------------------------

def adaptive_stroke_tokenization(strokes):
    """
    Refine stroke boundaries using direction/velocity/curvature change analysis.
    Returns a list of token sequences (each token is a sub-stroke segment).
    """
    tokens = []
    for stroke in strokes:
        # Compute changes
        points = stroke
        if len(points) < 3:
            tokens.append(points)
            continue
        # Direction change detection
        directions = []
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            directions.append(math.atan2(dy, dx))
        direction_changes = [abs(directions[i] - directions[i-1]) for i in range(1, len(directions))]
        # Velocity change detection
        velocities = [math.sqrt((points[i]['x']-points[i-1]['x'])**2 + 
                               (points[i]['y']-points[i-1]['y'])**2) for i in range(1, len(points))]
        vel_changes = [abs(velocities[i] - velocities[i-1]) for i in range(1, len(velocities))]
        # Curvature change (simplified as second derivative of direction)
        curv_changes = [abs(direction_changes[i] - direction_changes[i-1]) for i in range(1, len(direction_changes))]
        # Combine change scores
        change_scores = []
        for i in range(len(curv_changes)):
            # index i corresponds to point i+2 (0-based)
            score = direction_changes[i] + vel_changes[i] + curv_changes[i]
            change_scores.append(score)
        # Threshold-based boundary detection (adaptive threshold: mean + std)
        threshold = np.mean(change_scores) + 0.5 * np.std(change_scores)
        boundaries = [i+2 for i, s in enumerate(change_scores) if s > threshold]  # point indices
        # Split stroke into sub-strokes at boundaries
        start = 0
        for b in boundaries + [len(points)]:
            if b > start:
                tokens.append(points[start:b])
                start = b
    return tokens

# ------------------------------
# Step 6: Adaptive Hierarchical Stroke Encoding Network (AHSEN)
# ------------------------------

class TCNBlock(nn.Module):
    """Simplified Temporal Convolutional Network block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.transpose(1, 2)  # (batch, seq_len, channels)
        out = self.norm(out)
        out = out.transpose(1, 2)
        return out

class SimplifiedPixelFormer(nn.Module):
    """Simplified Transformer encoder used as a stand-in for PixelFormer+NeWCRFs."""
    def __init__(self, d_model, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return self.transformer(x)

class AHSEN(nn.Module):
    """
    Adaptive Hierarchical Stroke Encoding Network.
    Processes stroke tokens with TCN (local) and Transformer (global).
    """
    def __init__(self, input_dim=7, d_model=64, num_strokes=10):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.tcn = TCNBlock(d_model, d_model)
        self.pixelformer = SimplifiedPixelFormer(d_model)
        self.stroke_pool = nn.AdaptiveAvgPool1d(1)  # pool over stroke length
    
    def forward(self, stroke_features):
        """
        stroke_features: tensor of shape (batch, num_strokes, seq_len, input_dim)
        Returns global representation: (batch, d_model)
        """
        batch, num_strokes, seq_len, feat_dim = stroke_features.shape
        # Flatten strokes for TCN processing
        x = stroke_features.view(batch*num_strokes, seq_len, feat_dim)
        x = self.input_proj(x)  # (batch*num_strokes, seq_len, d_model)
        x = x.transpose(1, 2)   # (batch*num_strokes, d_model, seq_len)
        x = self.tcn(x)         # (batch*num_strokes, d_model, seq_len)
        x = x.transpose(1, 2)   # (batch*num_strokes, seq_len, d_model)
        # Global encoding with PixelFormer
        x = self.pixelformer(x) # (batch*num_strokes, seq_len, d_model)
        # Pool over sequence length to get stroke-level representation
        x = x.mean(dim=1)       # (batch*num_strokes, d_model)
        # Reshape back to strokes and combine
        x = x.view(batch, num_strokes, -1)  # (batch, num_strokes, d_model)
        # Pool over strokes to get a single vector per sample
        x = x.mean(dim=1)       # (batch, d_model)
        return x

# ------------------------------
# Step 7: Cross-Script Feature Learning (GAT + Multi-Head Cross-Attention)
# ------------------------------

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for stroke relationship modeling."""
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def forward(self, x, adj):
        # x: (batch, num_nodes, in_features)
        # adj: (batch, num_nodes, num_nodes) adjacency matrix
        h = self.W(x)  # (batch, N, out_features)
        N = h.size(1)
        # Compute attention coefficients
        # Repeat for all pairs
        h_i = h.unsqueeze(2).repeat(1,1,N,1)  # (batch, N, N, out)
        h_j = h.unsqueeze(1).repeat(1,N,1,1)  # (batch, N, N, out)
        concat = torch.cat([h_i, h_j], dim=-1)  # (batch, N, N, 2*out)
        e = self.leakyrelu(self.a(concat)).squeeze(-1)  # (batch, N, N)
        # Masked attention (only where adj=1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        h_prime = torch.matmul(attention, h)  # (batch, N, out)
        return h_prime

class CrossScriptFeatureLearning(nn.Module):
    """Combines GAT over strokes and multi-head cross-attention."""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.gat = GraphAttentionLayer(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, adjacency):
        # x: (batch, num_strokes, d_model)
        gat_out = self.gat(x, adjacency)  # (batch, num_strokes, d_model)
        # Self-attention as cross-attention between script-specific and script-independent
        attn_out, _ = self.cross_attn(gat_out, gat_out, gat_out)
        out = self.norm(gat_out + attn_out)
        return out  # (batch, num_strokes, d_model)

# ------------------------------
# Step 8: Confidence-Guided Transformer Recognition Network (CGTR-Net)
# ------------------------------

class CGTRNet(nn.Module):
    """
    Transformer encoder with cross-stage skip connections and CTC decoder.
    """
    def __init__(self, d_model=64, num_classes=11, nhead=4, num_layers=3):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        # Cross-stage skip connections: add residual from earlier layers
        self.skip_layers = [1, 2]  # connect layer 1 output to layer 3 input
        self.classifier = nn.Linear(d_model, num_classes)  # +1 for CTC blank
        self.ctc_loss = nn.CTCLoss(blank=num_classes-1, zero_infinity=True)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model) unified feature representation.
        For simplicity, we use the global AHSEN output replicated to seq_len.
        In a real scenario, x would be stroke-level features.
        """
        # Add positional encoding
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
        pos = pos / seq_len
        x = x + pos
        # Transformer encoder with skip connections
        hidden = x
        for i, layer in enumerate(self.transformer.layers):
            hidden = layer(hidden)
            if i in self.skip_layers:
                # Add skip connection (residual)
                hidden = hidden + x  # skip from input
        logits = self.classifier(hidden)  # (batch, seq_len, num_classes)
        return logits

# ------------------------------
# Step 9: Confidence-Guided Recognition Refinement
# ------------------------------

def confidence_guided_refinement(logits, threshold=0.6):
    """
    Refine low-confidence predictions using contextual re-ranking.
    This is a simplified version; real implementation would use a candidate generator
    and context-aware scoring.
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, preds = probs.max(dim=-1)
    # Identify low-confidence positions
    low_conf_mask = max_probs < threshold
    if low_conf_mask.any():
        # Re-rank using top-k candidates and a simple context score
        topk = 3
        topk_probs, topk_indices = probs.topk(topk, dim=-1)
        for t in range(logits.size(1)):
            if low_conf_mask[:, t].any():
                # For each low-confidence sample, adjust based on neighboring contexts
                # Simple heuristic: prefer candidate that appears in neighbors
                # This is a placeholder.
                pass  # In a real implementation, a language model or context scorer would be used
    return preds

# ------------------------------
# Step 10: Indic Text Reconstruction
# ------------------------------

def ctc_beam_search_decode(logits, beam_width=5, blank_idx=10):
    """CTC beam search decoding."""
    log_probs = F.log_softmax(logits, dim=-1)
    batch_size, seq_len, _ = log_probs.shape
    decoded = []
    for b in range(batch_size):
        # Simple greedy decoding for demonstration (beam search would be more involved)
        probs = log_probs[b].exp()
        preds = probs.argmax(dim=-1)
        # Collapse repeated characters and remove blanks
        result = []
        prev = blank_idx
        for p in preds:
            if p != blank_idx and p != prev:
                result.append(p.item())
            prev = p
        decoded.append(result)
    return decoded

# ------------------------------
# Step 11: Model Training
# ------------------------------

def train_model(model, dataloader, num_epochs=5, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    ctc_loss_fn = nn.CTCLoss(blank=10, zero_infinity=True)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (trajs, labels) in enumerate(dataloader):
            # Preprocess and extract features for each trajectory
            all_stroke_feats = []
            all_adj = []
            for traj in trajs:
                # Preprocess
                traj = preprocess_trajectory(traj)
                # Segment strokes
                strokes = segment_trajectory(traj, use_penup=True)
                # Tokenize adaptively
                tokens = adaptive_stroke_tokenization(strokes)
                # Convert tokens to features
                # For simplicity, we use the original strokes' features
                feats = extract_dynamic_features(tokens if tokens else strokes)
                all_stroke_feats.append(feats)
            # Pad variable stroke counts
            max_strokes = max(f.shape[0] for f in all_stroke_feats)
            stroke_batch = torch.zeros(len(all_stroke_feats), max_strokes, all_stroke_feats[0].shape[1], all_stroke_feats[0].shape[2])
            adj_batch = torch.zeros(len(all_stroke_feats), max_strokes, max_strokes)
            for i, f in enumerate(all_stroke_feats):
                num_s = f.shape[0]
                stroke_batch[i, :num_s] = torch.tensor(f, dtype=torch.float32)
                # Fully connected adjacency among existing strokes
                adj_batch[i, :num_s, :num_s] = 1.0
            stroke_batch = stroke_batch.to(device)
            adj_batch = adj_batch.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            # Forward pass
            # AHSEN: global encoding
            ahsen_out = model.ahsen(stroke_batch)  # (batch, d_model)
            # Cross-script: need stroke-level representation; we use a simple projection
            # In a full model, AHSEN would output per-stroke features. Here we approximate.
            stroke_feats = stroke_batch.mean(dim=2)  # (batch, num_strokes, input_dim)
            stroke_feats = model.stroke_proj(stroke_feats)  # (batch, num_strokes, d_model)
            cross_out = model.cross_script(stroke_feats, adj_batch)  # (batch, num_strokes, d_model)
            # Combine global and cross-script features
            unified = cross_out.mean(dim=1) + ahsen_out  # (batch, d_model)
            # CGTR-Net: replicate unified to sequence length
            seq_len = 20
            unified_seq = unified.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, d_model)
            logits = model.cgtr(unified_seq)  # (batch, seq_len, num_classes)
            
            # Compute losses
            # CTC loss (requires input_lengths, target_lengths)
            input_lengths = torch.full((logits.size(0),), seq_len, dtype=torch.long)
            # Target sequences: each label as a sequence of length 1 (or more if we had real text)
            targets = labels.unsqueeze(1)  # (batch, 1)
            target_lengths = torch.ones(labels.size(0), dtype=torch.long)
            ctc_loss = ctc_loss_fn(logits.transpose(0,1), targets, input_lengths, target_lengths)
            
            # Cross-entropy for script classification (dummy; we treat labels as script ID)
            script_logits = model.script_classifier(unified)  # (batch, num_scripts)
            ce_loss = ce_loss_fn(script_logits, labels % 2)  # dummy script labels
            
            # Confidence calibration loss (simple: encourage high confidence on correct)
            probs = F.softmax(logits, dim=-1)
            # For each sample, compute NLL of target at the first position
            nll = -torch.log(probs[:, 0, labels] + 1e-8).mean()
            total_batch_loss = ctc_loss + 0.1*ce_loss + 0.1*nll
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ------------------------------
# Step 12: Performance Evaluation
# ------------------------------

def evaluate(model, dataloader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    with torch.no_grad():
        for trajs, labels in dataloader:
            start_time = time.time()
            # Similar preprocessing as training
            all_stroke_feats = []
            for traj in trajs:
                traj = preprocess_trajectory(traj)
                strokes = segment_trajectory(traj, use_penup=True)
                tokens = adaptive_stroke_tokenization(strokes)
                feats = extract_dynamic_features(tokens if tokens else strokes)
                all_stroke_feats.append(feats)
            # Pad
            max_strokes = max(f.shape[0] for f in all_stroke_feats)
            stroke_batch = torch.zeros(len(all_stroke_feats), max_strokes, all_stroke_feats[0].shape[1], all_stroke_feats[0].shape[2])
            adj_batch = torch.zeros(len(all_stroke_feats), max_strokes, max_strokes)
            for i, f in enumerate(all_stroke_feats):
                num_s = f.shape[0]
                stroke_batch[i, :num_s] = torch.tensor(f, dtype=torch.float32)
                adj_batch[i, :num_s, :num_s] = 1.0
            stroke_batch = stroke_batch.to(device)
            adj_batch = adj_batch.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            # Forward
            ahsen_out = model.ahsen(stroke_batch)
            stroke_feats = stroke_batch.mean(dim=2)
            stroke_feats = model.stroke_proj(stroke_feats)
            cross_out = model.cross_script(stroke_feats, adj_batch)
            unified = cross_out.mean(dim=1) + ahsen_out
            unified_seq = unified.unsqueeze(1).repeat(1, 20, 1)
            logits = model.cgtr(unified_seq)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Decode (greedy for simplicity)
            probs = F.softmax(logits, dim=-1)
            preds = probs[:, 0, :].argmax(dim=-1)  # use first position as classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    avg_inference_time = np.mean(inference_times)
    
    print(f"Character Recognition Accuracy (CRA): {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.4f} sec")
    # Other metrics like CER, WER, FLOPs, params would require more detailed setup.
    return accuracy, precision, recall, f1

# ------------------------------
# Full Framework Model
# ------------------------------

class MultiIndicOnlineHTR(nn.Module):
    def __init__(self, input_dim=7, d_model=64, num_classes=11, num_scripts=10):
        super().__init__()
        self.ahsen = AHSEN(input_dim=input_dim, d_model=d_model)
        self.stroke_proj = nn.Linear(input_dim, d_model)  # projection for cross-script
        self.cross_script = CrossScriptFeatureLearning(d_model)
        self.cgtr = CGTRNet(d_model=d_model, num_classes=num_classes)
        self.script_classifier = nn.Linear(d_model, num_scripts)  # for CE loss
    
    def forward(self, stroke_batch, adj_batch, seq_len=20):
        ahsen_out = self.ahsen(stroke_batch)
        stroke_feats = stroke_batch.mean(dim=2)
        stroke_feats = self.stroke_proj(stroke_feats)
        cross_out = self.cross_script(stroke_feats, adj_batch)
        unified = cross_out.mean(dim=1) + ahsen_out
        unified_seq = unified.unsqueeze(1).repeat(1, seq_len, 1)
        logits = self.cgtr(unified_seq)
        script_logits = self.script_classifier(unified)
        return logits, script_logits, unified

# ------------------------------
# Main Execution
# ------------------------------

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Data acquisition
    print("Step 1: Loading data...")
    dataset = OnlineHandwritingDataset(num_samples=200, max_len=80, use_uci=False)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=lambda x: list(zip(*x)))
    
    # Initialize model
    print("Initializing model...")
    model = MultiIndicOnlineHTR().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Step 11: Training
    print("Starting training...")
    train_model(model, train_loader, num_epochs=3, device=device)
    
    # Step 12: Evaluation
    print("Evaluating...")
    evaluate(model, test_loader, device=device)

if __name__ == "__main__":
    main()