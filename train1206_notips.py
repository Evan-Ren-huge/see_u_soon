# -*- coding: utf-8 -*-
#
# Function (YAML + global normalization + train/validation split):
# - Load Abaqus npz batch data (disp, s, rf2, coords, connectivity, ...)
# - DualGraph + GConvGRU: node graph + element graph
# - Predict: U(t) (node), s_elem(t) (element), RF2(t) (global scalar)
# - Autoregressive roll-out with feedback of U/Vel/s_prev
# - Use real batch training: multiple cases in parallel on GPU
# - Use YAML config + optional global or per-case normalization
# - Split cases into train/validation; validation only forward (no backprop) for model selection

import os, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
import yaml

# ===================== 0. Default config (can be overridden by YAML) =====================

# Data paths
NPZ_DIR      = "/root/autodl-tmp/file"
NPZ_PATH     = ""              # If you want to train on a single npz file, set its path; otherwise leave empty.

# Model / training
SAVE_PATH    = "/root/autodl-tmp/joint_us_dual_rf2_batch16.pth"
HIDDEN_NODE  = 64
HIDDEN_ELEM  = 64
EPOCHS       = 2000
LR           = 1e-3
CLIP_NORM    = 0.5
SEED         = 42

# Loss weights
LAMBDA_S     = 0.0    # If you want to include s loss, set this > 0 (e.g., 0.1 or 1.0).
LAMBDA_LAP   = 0.0    # Laplacian smoothing for displacement
LAMBDA_RF2   = 0.0    # Global RF2 loss weight

# Teacher forcing (note: P_TF_START=0 means fully disabled)
K_STEPS_TF   = 100    # Number of epochs to decay teacher forcing
P_TF_START   = 0.0    # Initial teacher forcing probability (0 = off)

USE_CUDA     = True

# Graph and feature modes
EDGE_MODE_NODE = "abaqus"   # "abaqus" or "complete"
ELEM_ADJ_MODE  = "face"     # "face" | "node" | "none"
S_PREV_MODE    = "mean"     # "mean" | "rms" | "none"
USE_SMOOTHL1   = True
K_HOP_NODE     = 1
K_HOP_ELEM     = 1

# Normalization & frame sampling
# Set to "global" for global normalization across all cases
NORM_SCOPE   = "global"     # "global" or "per_case"
FRAME_STRIDE = 2            # Frame downsampling stride

# Others
NO_S_FEEDBACK = True        # True disables s_prev feedback; only train displacement
BATCH_CASES   = 16          # Batch size (number of cases per batch)
IN_NODE_F     = 12          # Node feature dimension [coord(3), u_prev(3), alpha(1), flag(1), v_prev(3), s_prev(1)]

# Validation settings
VAL_RATIO    = 0.1          # Fraction of cases used for validation (0.1 = 10%; 0 means no validation split)
VAL_INTERVAL = 20           # Validate every N epochs
SAVE_BEST    = True         # Save best model according to validation loss


# ===================== 0.1 YAML loading =====================

def load_config_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_config(cfg: dict):
    """Override global default parameters using YAML config."""
    global NPZ_DIR, NPZ_PATH, SAVE_PATH
    global HIDDEN_NODE, HIDDEN_ELEM, EPOCHS, LR, CLIP_NORM, SEED
    global LAMBDA_S, LAMBDA_LAP, LAMBDA_RF2
    global K_STEPS_TF, P_TF_START, USE_CUDA
    global EDGE_MODE_NODE, ELEM_ADJ_MODE, S_PREV_MODE, USE_SMOOTHL1
    global K_HOP_NODE, K_HOP_ELEM
    global NORM_SCOPE, FRAME_STRIDE
    global NO_S_FEEDBACK, BATCH_CASES, IN_NODE_F
    global VAL_RATIO, VAL_INTERVAL, SAVE_BEST

    NPZ_DIR      = cfg.get("NPZ_DIR", NPZ_DIR)
    NPZ_PATH     = cfg.get("NPZ_PATH", NPZ_PATH)
    SAVE_PATH    = cfg.get("SAVE_PATH", SAVE_PATH)

    HIDDEN_NODE  = cfg.get("HIDDEN_NODE", HIDDEN_NODE)
    HIDDEN_ELEM  = cfg.get("HIDDEN_ELEM", HIDDEN_ELEM)
    EPOCHS       = cfg.get("EPOCHS", EPOCHS)
    LR           = cfg.get("LR", LR)
    CLIP_NORM    = cfg.get("CLIP_NORM", CLIP_NORM)
    SEED         = cfg.get("SEED", SEED)

    LAMBDA_S     = cfg.get("LAMBDA_S", LAMBDA_S)
    LAMBDA_LAP   = cfg.get("LAMBDA_LAP", LAMBDA_LAP)
    LAMBDA_RF2   = cfg.get("LAMBDA_RF2", LAMBDA_RF2)

    K_STEPS_TF   = cfg.get("K_STEPS_TF", K_STEPS_TF)
    P_TF_START   = cfg.get("P_TF_START", P_TF_START)
    USE_CUDA     = cfg.get("USE_CUDA", USE_CUDA)

    EDGE_MODE_NODE = cfg.get("EDGE_MODE_NODE", EDGE_MODE_NODE)
    ELEM_ADJ_MODE  = cfg.get("ELEM_ADJ_MODE", ELEM_ADJ_MODE)
    S_PREV_MODE    = cfg.get("S_PREV_MODE", S_PREV_MODE)
    USE_SMOOTHL1   = cfg.get("USE_SMOOTHL1", USE_SMOOTHL1)
    K_HOP_NODE     = cfg.get("K_HOP_NODE", K_HOP_NODE)
    K_HOP_ELEM     = cfg.get("K_HOP_ELEM", K_HOP_ELEM)

    NORM_SCOPE   = cfg.get("NORM_SCOPE", NORM_SCOPE)
    FRAME_STRIDE = cfg.get("FRAME_STRIDE", FRAME_STRIDE)

    NO_S_FEEDBACK = cfg.get("NO_S_FEEDBACK", NO_S_FEEDBACK)
    BATCH_CASES   = cfg.get("BATCH_CASES", BATCH_CASES)
    IN_NODE_F     = cfg.get("IN_NODE_F", IN_NODE_F)

    VAL_RATIO    = cfg.get("VAL_RATIO", VAL_RATIO)
    VAL_INTERVAL = cfg.get("VAL_INTERVAL", VAL_INTERVAL)
    SAVE_BEST    = cfg.get("SAVE_BEST", SAVE_BEST)


# ===================== 1. Utility functions =====================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def list_npz(npz_dir, npz_path):
    """Collect npz files from directory and/or single path (no duplicates)."""
    files = []
    if npz_dir and os.path.isdir(npz_dir):
        for f in os.listdir(npz_dir):
            if f.lower().endswith(".npz"):
                files.append(os.path.join(npz_dir, f))
    if npz_path and os.path.isfile(npz_path):
        files.append(npz_path)
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    out.sort()
    return out

def label_to_index_map(node_labels: np.ndarray):
    """Map (Abaqus) node labels to 0-based indices."""
    return {int(lbl): i for i, lbl in enumerate(node_labels.tolist())}

def compute_vel_from_disp(disp: np.ndarray):
    """Compute velocity as first difference of displacement along time."""
    T, N, _ = disp.shape
    vel = np.zeros_like(disp, dtype=np.float32)
    vel[1:] = disp[1:] - disp[:-1]
    return vel

def make_flags(N: int, node_labels: np.ndarray, surf_labels: np.ndarray):
    """Binary flag for nodes on the loading surface."""
    surf = set(int(x) for x in surf_labels.tolist())
    flags = np.array([1.0 if int(lbl) in surf else 0.0 for lbl in node_labels.tolist()],
                     dtype=np.float32)
    return flags.reshape(N, 1)

def _pairs_for_elem_C3D8():
    """Return 12 edges (pairs of local indices) for a C3D8 element."""
    return [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

def build_node_edge_index(connectivity_idx: np.ndarray, mode="abaqus"):
    """Build node graph edges from element connectivity."""
    edges = set()
    if mode == "abaqus":
        pairs = _pairs_for_elem_C3D8()
        for elem in connectivity_idx:
            for a, b in pairs:
                i, j = int(elem[a]), int(elem[b])
                if i != j:
                    edges.add((i, j))
                    edges.add((j, i))
    else:
        for elem in connectivity_idx:
            for a in range(8):
                for b in range(a+1, 8):
                    i, j = int(elem[a]), int(elem[b])
                    if i != j:
                        edges.add((i, j))
                        edges.add((j, i))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return ei

def build_elem_edge_index(connectivity_idx: np.ndarray, mode="face"):
    """Build element graph edges based on shared faces or shared nodes."""
    Ne = connectivity_idx.shape[0]
    if mode == "none" or Ne == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edges = set()
    if mode == "node":
        # Connect elements that share nodes
        node2elems = {}
        for e, nodes in enumerate(connectivity_idx):
            for n in nodes.tolist():
                node2elems.setdefault(int(n), []).append(e)
        for lst in node2elems.values():
            m = len(lst)
            for i in range(m):
                for j in range(i+1, m):
                    a, b = lst[i], lst[j]
                    edges.add((a, b))
                    edges.add((b, a))
    else:
        # Connect elements that share faces
        faces_local = [
            (0,1,2,3), (4,5,6,7),
            (0,1,5,4), (1,2,6,5),
            (2,3,7,6), (3,0,4,7)
        ]
        face_map = {}
        for e, nodes in enumerate(connectivity_idx):
            for fc in faces_local:
                face = tuple(sorted([int(nodes[i]) for i in fc]))
                owner = face_map.get(face, -1)
                if owner == -1:
                    face_map[face] = e
                else:
                    a, b = owner, e
                    edges.add((a, b))
                    edges.add((b, a))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return ei

def elems_to_nodes_scalar(s_elem: torch.Tensor, connectivity_idx: torch.Tensor, mode="mean"):
    """Aggregate element scalar to node scalar by mean or RMS over incident elements."""
    Ne = connectivity_idx.size(0)
    if Ne == 0:
        return torch.zeros(0, device=s_elem.device, dtype=s_elem.dtype)
    N = int(connectivity_idx.max().item()) + 1
    idx = connectivity_idx.reshape(-1)
    acc = torch.zeros(N, device=s_elem.device, dtype=s_elem.dtype)
    cnt = torch.zeros(N, device=s_elem.device, dtype=s_elem.dtype)
    if mode == "rms":
        acc.index_add_(0, idx, (s_elem.repeat_interleave(8))**2)
        cnt.index_add_(0, idx, torch.ones_like(idx, dtype=cnt.dtype))
        return torch.sqrt(acc / (cnt + 1e-6))
    else:
        acc.index_add_(0, idx, s_elem.repeat_interleave(8))
        cnt.index_add_(0, idx, torch.ones_like(idx, dtype=cnt.dtype))
        return acc / (cnt + 1e-6)

def huber_loss(input: torch.Tensor, target: torch.Tensor, beta: float = 0.5, reduction: str = "mean"):
    """Standard Huber loss."""
    x = input - target
    absx = torch.abs(x)
    loss = torch.where(absx < beta, 0.5 * (x**2) / beta, absx - 0.5 * beta)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def laplacian_smooth_loss(u_t: torch.Tensor, edge_index: torch.Tensor):
    """Simple Laplacian smoothness loss on displacement field."""
    if edge_index.numel() == 0:
        return u_t.new_zeros(())
    src, dst = edge_index
    diff = u_t[dst] - u_t[src]         # (E,3)
    return (diff * diff).mean()

def case_stats_from_arrays(coord, disp, s_elem):
    """Compute per-case statistics for coord / disp / vel / s_elem."""
    vel = compute_vel_from_disp(disp)
    mu_coord = coord.mean(axis=0).astype(np.float32)
    sig_coord = coord.std(axis=0).astype(np.float32) + 1e-8
    mu_u = disp.mean(axis=(0, 1)).astype(np.float32)
    sig_u = disp.std(axis=(0, 1)).astype(np.float32) + 1e-8
    mu_v = vel.mean(axis=(0, 1)).astype(np.float32)
    sig_v = vel.std(axis=(0, 1)).astype(np.float32) + 1e-8
    mu_s = float(s_elem.mean())
    sig_s = float(s_elem.std() + 1e-8)
    return dict(mu_coord=mu_coord, sig_coord=sig_coord,
                mu_u=mu_u, sig_u=sig_u,
                mu_v=mu_v, sig_v=sig_v,
                mu_s=mu_s, sig_s=sig_s)

# === Global normalization stats (global normalization) ===
def compute_global_stats(npz_files, frame_stride=1, s_elem_mode="mean"):
    """
    Compute global mean/std over all cases for coord / disp / vel / s_elem.
    """
    sum_coord = np.zeros(3, dtype=np.float64)
    sumsq_coord = np.zeros(3, dtype=np.float64)
    count_coord = 0

    sum_u = np.zeros(3, dtype=np.float64)
    sumsq_u = np.zeros(3, dtype=np.float64)
    count_u = 0

    sum_v = np.zeros(3, dtype=np.float64)
    sumsq_v = np.zeros(3, dtype=np.float64)
    count_v = 0

    sum_s = 0.0
    sumsq_s = 0.0
    count_s = 0

    for path in npz_files:
        dat = np.load(path, allow_pickle=True)

        disp  = dat["disp"].astype(np.float32)          # (T,N,3)
        coord = dat["node_coords"].astype(np.float32)   # (N,3)
        conn  = dat["connectivity"].astype(np.int64)    # (Ne,8)
        nlab  = dat["node_labels"].astype(np.int64)

        disp  = disp[::frame_stride]

        if "s_elem" in dat:
            s_elem = dat["s_elem"].astype(np.float32)[::frame_stride]
        else:
            s_node = dat["s"].astype(np.float32)[::frame_stride]   # (T,N)
            l2i = label_to_index_map(nlab)
            conn_idx_np = np.vectorize(lambda x: l2i[int(x)])(conn)  # (Ne,8)
            s_elem_list = []
            for t in range(s_node.shape[0]):
                vals = s_node[t][conn_idx_np]   # (Ne,8)
                if s_elem_mode == "mean":
                    s_elem_list.append(vals.mean(axis=1))
                else:
                    s_elem_list.append(np.sqrt((vals * vals).mean(axis=1) + 1e-12))
            s_elem = np.stack(s_elem_list, axis=0)  # (T,Ne)

        vel = compute_vel_from_disp(disp)

        # coord
        sum_coord += coord.sum(axis=0, dtype=np.float64)
        sumsq_coord += (coord.astype(np.float64)**2).sum(axis=0)
        count_coord += coord.shape[0]

        # disp
        disp_flat = disp.reshape(-1, 3).astype(np.float64)
        sum_u += disp_flat.sum(axis=0)
        sumsq_u += (disp_flat**2).sum(axis=0)
        count_u += disp_flat.shape[0]

        # vel
        vel_flat = vel.reshape(-1, 3).astype(np.float64)
        sum_v += vel_flat.sum(axis=0)
        sumsq_v += (vel_flat**2).sum(axis=0)
        count_v += vel_flat.shape[0]

        # s_elem
        s_flat = s_elem.reshape(-1).astype(np.float64)
        sum_s += s_flat.sum()
        sumsq_s += (s_flat**2).sum()
        count_s += s_flat.shape[0]

    def finalize(sum_, sumsq_, cnt):
        mean = (sum_ / max(cnt, 1)).astype(np.float32)
        var = (sumsq_ / max(cnt, 1)) - (sum_ / max(cnt, 1))**2
        std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32) + 1e-8
        return mean, std

    mu_coord, sig_coord = finalize(sum_coord, sumsq_coord, count_coord)
    mu_u, sig_u = finalize(sum_u, sumsq_u, count_u)
    mu_v, sig_v = finalize(sum_v, sumsq_v, count_v)

    mu_s, sig_s = finalize(np.array([sum_s]), np.array([sumsq_s]), count_s)
    mu_s = float(mu_s[0])
    sig_s = float(sig_s[0])

    stats = dict(
        mu_coord=mu_coord, sig_coord=sig_coord,
        mu_u=mu_u, sig_u=sig_u,
        mu_v=mu_v, sig_v=sig_v,
        mu_s=mu_s, sig_s=sig_s
    )
    return stats

# === Batched edge_index (for both node and element graphs) ===
def build_batched_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int):
    """
    edge_index: (2,E), node indices 0..num_nodes-1
    return: (2, batch_size*E), edges for all batches with node offsets
    """
    if edge_index.numel() == 0 or batch_size == 1:
        return edge_index
    ei_list = []
    for b in range(batch_size):
        offset = b * num_nodes
        offset_vec = torch.tensor(
            [offset, offset],
            device=edge_index.device,
            dtype=edge_index.dtype
        ).view(2, 1)
        ei_list.append(edge_index + offset_vec)
    return torch.cat(ei_list, dim=1)

# Batched elem_nodes_idx for aggregating node features to element features
def build_batched_elem_nodes_idx(elem_nodes_idx: torch.Tensor, batch_size: int, num_nodes: int):
    """
    elem_nodes_idx: (Ne,8)  element connectivity with node indices (0..num_nodes-1)
    return: (batch_size*Ne,8), connectivity with batch-wise node offsets
    """
    if batch_size == 1:
        return elem_nodes_idx
    idx_list = []
    for b in range(batch_size):
        offset = b * num_nodes
        idx_list.append(elem_nodes_idx + offset)
    return torch.cat(idx_list, dim=0)


# ===================== 2. Model: DualGraph + RF2 =====================

class DualGraphModel(nn.Module):
    def __init__(self, in_node_f=IN_NODE_F, hidden_node=HIDDEN_NODE,
                 hidden_elem=HIDDEN_ELEM, K_node=K_HOP_NODE, K_elem=K_HOP_ELEM):
        super().__init__()
        # Node side: two-layer GConvGRU
        self.node_gru1 = GConvGRU(in_node_f,   hidden_node, K=K_node)
        self.node_gru2 = GConvGRU(hidden_node, hidden_node, K=K_node)

        # Element side: two-layer GConvGRU
        self.elem_gru1 = GConvGRU(hidden_node, hidden_elem, K=K_elem)
        self.elem_gru2 = GConvGRU(hidden_elem, hidden_elem, K=K_elem)

        self.head_u      = nn.Linear(hidden_node, 3)
        self.head_s_elem = nn.Linear(hidden_elem, 1)
        self.head_rf2    = nn.Linear(hidden_elem, 1)

    # ---------- Single-case version (for inference) ----------
    def forward(self, X_node_seq, node_edge_index, elem_edge_index, elem_nodes_idx,
                Y_u=None, Y_s_elem=None, teacher_forcing_prob=0.0,
                s_prev_mode="mean"):
        """
        X_node_seq: (T,N,Fin)
        """
        T, N, F = X_node_seq.shape
        Ne = elem_nodes_idx.size(0)
        X = [x.clone() for x in X_node_seq]

        Hn1 = None
        Hn2 = None
        He1 = None
        He2 = None
        outs_u = []
        outs_se = []
        outs_rf2 = []

        for t in range(T):
            # Node GNN: two-layer GConvGRU
            Hn1 = self.node_gru1(X[t], edge_index=node_edge_index, H=Hn1)  # (N,hidden_node)
            Hn2 = self.node_gru2(Hn1,  edge_index=node_edge_index, H=Hn2)  # (N,hidden_node)
            u_t = self.head_u(Hn2)                                         # (N,3)

            # Element GNN: two-layer GConvGRU
            if Ne > 0:
                H_e_in = Hn2[elem_nodes_idx].mean(dim=1)                   # (Ne,hidden_node)
                He1 = self.elem_gru1(H_e_in, edge_index=elem_edge_index, H=He1)  # (Ne,hidden_elem)
                He2 = self.elem_gru2(He1,    edge_index=elem_edge_index, H=He2)  # (Ne,hidden_elem)
                s_e_t = self.head_s_elem(He2).squeeze(-1)                         # (Ne,)
                h_rf2 = He2.mean(dim=0, keepdim=True)                             # (1,H)
                rf2_t = self.head_rf2(h_rf2).squeeze(-1).squeeze(-1)             # scalar
            else:
                s_e_t = u_t.new_zeros((0,), dtype=u_t.dtype)
                rf2_t = u_t.new_zeros((), dtype=u_t.dtype)

            outs_u.append(u_t)
            outs_se.append(s_e_t)
            outs_rf2.append(rf2_t)

            # Autoregressive feedback
            if t + 1 < T:
                use_tf = (
                    (teacher_forcing_prob > 0)
                    and (np.random.rand() < teacher_forcing_prob)
                    and (Y_u is not None)
                    and (Y_s_elem is not None)
                )

                if use_tf:
                    u_feed = Y_u[t]
                    s_e_feed = Y_s_elem[t]
                else:
                    u_feed = u_t
                    s_e_feed = s_e_t

                u_prev = X[t][:, 3:6]
                v_feed = u_feed - u_prev

                X[t+1][:, 3:6]  = u_feed
                X[t+1][:, 8:11] = v_feed

                if s_prev_mode != "none" and Ne > 0:
                    s_node = elems_to_nodes_scalar(s_e_feed, elem_nodes_idx, mode=s_prev_mode).detach()
                    X[t+1][:, 11] = s_node

        u_hat    = torch.stack(outs_u)   # (T,N,3)
        s_elem_h = torch.stack(outs_se)  # (T,Ne)
        rf2_hat  = torch.stack(outs_rf2) # (T,)

        return u_hat, s_elem_h, rf2_hat

    # ---------- Batched version (for training / validation) ----------
    def forward_batch(self, X_node_seq_batch, node_edge_index, elem_edge_index, elem_nodes_idx,
                      Y_u=None, Y_s_elem=None, teacher_forcing_prob=0.0,
                      s_prev_mode="mean"):
        """
        X_node_seq_batch: (T,B,N,Fin)
        """
        T, B, N, F = X_node_seq_batch.shape
        Ne = elem_nodes_idx.size(0)

        X = [X_node_seq_batch[t].clone() for t in range(T)]  # each (B,N,F)

        # Build batched edges
        node_edge_b = build_batched_edge_index(node_edge_index, B, N)     # (2,B*En)
        elem_edge_b = build_batched_edge_index(elem_edge_index, B, Ne)    # (2,B*Ee)
        elem_nodes_b = build_batched_elem_nodes_idx(elem_nodes_idx, B, N) # (B*Ne,8)

        Hn1 = None
        Hn2 = None
        He1 = None
        He2 = None
        outs_u = []
        outs_se = []
        outs_rf2 = []

        for t in range(T):
            # (B,N,F) -> (B*N,F)
            x_t = X[t].reshape(B * N, F)

            # Node side (two-layer GConvGRU)
            Hn1 = self.node_gru1(x_t, edge_index=node_edge_b, H=Hn1)     # (B*N,hidden_node)
            Hn2 = self.node_gru2(Hn1, edge_index=node_edge_b, H=Hn2)     # (B*N,hidden_node)
            u_t_flat = self.head_u(Hn2)                                  # (B*N,3)
            u_t = u_t_flat.reshape(B, N, 3)                              # (B,N,3)

            # Element side (two-layer GConvGRU)
            if Ne > 0:
                H_e_in = Hn2[elem_nodes_b].mean(dim=1)                   # (B*Ne,hidden_node)
                He1 = self.elem_gru1(H_e_in, edge_index=elem_edge_b, H=He1)  # (B*Ne,hidden_elem)
                He2 = self.elem_gru2(He1,    edge_index=elem_edge_b, H=He2)  # (B*Ne,hidden_elem)
                s_e_flat = self.head_s_elem(He2).squeeze(-1)                 # (B*Ne,)
                s_e_t = s_e_flat.reshape(B, Ne)                               # (B,Ne)

                He2_b = He2.reshape(B, Ne, -1)                                # (B,Ne,H)
                h_rf2 = He2_b.mean(dim=1)                                     # (B,H)
                rf2_t = self.head_rf2(h_rf2).squeeze(-1)                      # (B,)
            else:
                s_e_t = u_t.new_zeros((B, 0), dtype=u_t.dtype)
                rf2_t = u_t.new_zeros((B,), dtype=u_t.dtype)

            outs_u.append(u_t)        # (B,N,3)
            outs_se.append(s_e_t)     # (B,Ne)
            outs_rf2.append(rf2_t)    # (B,)

            # Autoregressive feedback
            if t + 1 < T:
                use_tf = (
                    (teacher_forcing_prob > 0)
                    and (np.random.rand() < teacher_forcing_prob)
                    and (Y_u is not None)
                    and (Y_s_elem is not None)
                )

                if use_tf:
                    u_feed = Y_u[t]        # (B,N,3)
                    s_e_feed = Y_s_elem[t] # (B,Ne)
                else:
                    u_feed = u_t
                    s_e_feed = s_e_t

                u_prev = X[t][:, :, 3:6]     # (B,N,3)
                v_feed = u_feed - u_prev     # (B,N,3)

                X[t+1][:, :, 3:6]  = u_feed
                X[t+1][:, :, 8:11] = v_feed

                if s_prev_mode != "none" and Ne > 0:
                    s_prev_list = []
                    for b in range(B):
                        s_elem_b = s_e_feed[b]                       # (Ne,)
                        s_node_b = elems_to_nodes_scalar(s_elem_b, elem_nodes_idx, mode=s_prev_mode)
                        s_prev_list.append(s_node_b.view(1, -1))     # (1,N)
                    s_prev = torch.cat(s_prev_list, dim=0)           # (B,N)
                    X[t+1][:, :, 11] = s_prev

        u_hat    = torch.stack(outs_u,   dim=0)  # (T,B,N,3)
        s_elem_h = torch.stack(outs_se,  dim=0)  # (T,B,Ne)
        rf2_hat  = torch.stack(outs_rf2, dim=0)  # (T,B)

        return u_hat, s_elem_h, rf2_hat


# ===================== 3. Load a single case (with rf2) =====================

def load_case(npz_path: str, device: torch.device, stats_dict: dict,
              edge_mode_node=EDGE_MODE_NODE, elem_adj_mode=ELEM_ADJ_MODE,
              s_elem_mode="mean", frame_stride=1, norm_scope="global"):
    """Load a single npz case and build all tensors for training/inference."""
    dat = np.load(npz_path, allow_pickle=True)

    disp  = dat["disp"].astype(np.float32)          # (T,N,3)
    coord = dat["node_coords"].astype(np.float32)   # (N,3)
    conn  = dat["connectivity"].astype(np.int64)    # (Ne,8)
    nlab  = dat["node_labels"].astype(np.int64)     # (N,)
    surf  = dat["SURF1_NODE_LABELS"].astype(np.int64)
    times = dat["frame_times"].astype(np.float32)   # (T,)

    rf2   = dat["rf2"].astype(np.float32)           # (T,)

    # Frame downsampling
    disp  = disp[::frame_stride]
    times = times[::frame_stride]
    rf2   = rf2[::frame_stride]

    if "s_elem" in dat:
        s_elem = dat["s_elem"].astype(np.float32)[::frame_stride]
    else:
        s_node = dat["s"].astype(np.float32)[::frame_stride]   # (T,N)
        l2i = label_to_index_map(nlab)
        conn_idx_np = np.vectorize(lambda x: l2i[int(x)])(conn)  # (Ne,8)
        s_elem_list = []
        for t in range(s_node.shape[0]):
            vals = s_node[t][conn_idx_np]   # (Ne,8)
            if s_elem_mode == "mean":
                s_elem_list.append(vals.mean(axis=1))
            else:
                s_elem_list.append(np.sqrt((vals * vals).mean(axis=1) + 1e-12))
        s_elem = np.stack(s_elem_list, axis=0)  # (T,Ne)

    # Convert connectivity to 0-based index
    l2i = label_to_index_map(nlab)
    conn_idx = np.vectorize(lambda x: l2i[int(x)])(conn)  # (Ne,8)

    # Graph edges
    node_edge = build_node_edge_index(conn_idx, mode=edge_mode_node).to(device)
    elem_edge = build_elem_edge_index(conn_idx, mode=elem_adj_mode).to(device)
    elem_nodes_idx = torch.tensor(conn_idx, dtype=torch.long, device=device)

    # Node flags on loaded surface
    flag = make_flags(coord.shape[0], nlab, surf).astype(np.float32)

    # Normalization
    if norm_scope == "per_case":
        stats_used = case_stats_from_arrays(coord, disp, s_elem)
    else:
        assert stats_dict is not None, "Global normalization requires non-empty stats_dict."
        stats_used = stats_dict

    mu_c, sg_c = stats_used["mu_coord"], stats_used["sig_coord"]
    mu_u, sg_u = stats_used["mu_u"],    stats_used["sig_u"]
    mu_v, sg_v = stats_used["mu_v"],    stats_used["sig_v"]
    mu_s, sg_s = stats_used["mu_s"],    stats_used["sig_s"]

    vel = compute_vel_from_disp(disp)

    coord_n = (coord - mu_c) / sg_c
    disp_n  = (disp  - mu_u) / sg_u
    vel_n   = (vel   - mu_v) / sg_v
    s_elem_n = (s_elem - mu_s) / sg_s

    T, N, _ = disp.shape
    X = torch.zeros((T, N, IN_NODE_F), dtype=torch.float32, device=device)
    coord_t = torch.tensor(coord_n, device=device)
    flag_t  = torch.tensor(flag, device=device)
    disp_t  = torch.tensor(disp_n, device=device)
    vel_t   = torch.tensor(vel_n,  device=device)
    s_elem_t = torch.tensor(s_elem_n, device=device)
    conn_idx_t = torch.tensor(conn_idx, dtype=torch.long, device=device)
    rf2_t  = torch.tensor(rf2, dtype=torch.float32, device=device)

    t0, t1 = float(times[0]), float(times[-1])
    alpha_all = ((times - t0) / (t1 - t0 + 1e-12)).astype(np.float32)

    for t in range(T):
        alpha = torch.full((N, 1), alpha_all[t], device=device)
        if t == 0:
            u_prev = torch.zeros((N, 3), device=device)
            v_prev = torch.zeros((N, 3), device=device)
            s_prev_col = torch.zeros((N, 1), device=device)
        else:
            u_prev = disp_t[t-1]
            v_prev = vel_t[t-1]
            s_prev_node = elems_to_nodes_scalar(s_elem_t[t-1], conn_idx_t, mode="mean")
            s_prev_col = s_prev_node.view(-1, 1)
        X[t] = torch.cat([coord_t, u_prev, alpha, flag_t, v_prev, s_prev_col], dim=1)

    return dict(
        X_node=X,
        node_edge=node_edge,
        elem_edge=elem_edge,
        elem_nodes_idx=elem_nodes_idx,
        Y_u=disp_t,
        Y_s_elem=s_elem_t,
        Y_rf2=rf2_t,
        times=times,
        conn_idx=conn_idx,
        stats_used=stats_used
    )


# ===================== 4. Validation function (forward only, no update) =====================

@torch.no_grad()
def evaluate(model: DualGraphModel,
             files,
             device: torch.device,
             gstats: dict,
             node_e0: torch.Tensor,
             elem_e0: torch.Tensor,
             enodes0: torch.Tensor):
    """
    Run full rollout validation over given files.
    Returns: val_L, val_Lu, val_Ls, val_Lrf2, val_Llap
    """
    if len(files) == 0:
        return None

    model.eval()

    total_L = total_Lu = total_Ls = total_Lrf2 = total_Llap = 0.0
    steps = 0

    for start in range(0, len(files), BATCH_CASES):
        batch_files = files[start:start + BATCH_CASES]
        B = len(batch_files)

        X_list   = []
        Y_u_list = []
        Y_se_list= []
        Y_rf2_list = []

        for f in batch_files:
            case = load_case(f, device, gstats,
                             edge_mode_node=EDGE_MODE_NODE,
                             elem_adj_mode=ELEM_ADJ_MODE,
                             s_elem_mode=S_PREV_MODE,
                             frame_stride=FRAME_STRIDE,
                             norm_scope=NORM_SCOPE)
            X, Y_u, Y_se, Y_rf2 = case["X_node"], case["Y_u"], case["Y_s_elem"], case["Y_rf2"]

            X_list.append(X)          # (T,N,F)
            Y_u_list.append(Y_u)      # (T,N,3)
            Y_se_list.append(Y_se)    # (T,Ne)
            Y_rf2_list.append(Y_rf2)  # (T,)

        X_batch    = torch.stack(X_list,   dim=1)   # (T,B,N,F)
        Y_u_batch  = torch.stack(Y_u_list, dim=1)   # (T,B,N,3)
        Y_se_batch = torch.stack(Y_se_list,dim=1)   # (T,B,Ne)
        Y_rf2_batch= torch.stack(Y_rf2_list,dim=1)  # (T,B)

        # Validation: no teacher forcing, full autoregressive rollout
        eff_s_prev_mode = "none" if (NO_S_FEEDBACK or S_PREV_MODE == "none") else S_PREV_MODE
        if NO_S_FEEDBACK or S_PREV_MODE == "none":
            X_batch[:, :, :, 11] = 0.0

        u_hat, s_elem_hat, rf2_hat = model.forward_batch(
            X_batch,
            node_e0, elem_e0, enodes0,
            Y_u=None, Y_s_elem=None,
            teacher_forcing_prob=0.0,
            s_prev_mode=eff_s_prev_mode
        )

        Lu = F.mse_loss(u_hat, Y_u_batch, reduction="mean")
        if USE_SMOOTHL1:
            Ls = huber_loss(s_elem_hat, Y_se_batch, beta=0.5, reduction="mean")
        else:
            Ls = F.mse_loss(s_elem_hat, Y_se_batch, reduction="mean")

        Lrf2 = F.mse_loss(rf2_hat, Y_rf2_batch, reduction="mean")

        if LAMBDA_LAP > 0.0:
            Llap = 0.0
            T_b = u_hat.size(0)
            node_edge_b = build_batched_edge_index(node_e0, B, node_e0.max().item() + 1)
            for t in range(T_b):
                u_t_flat = u_hat[t].reshape(B * (node_e0.max().item() + 1), 3)
                Llap = Llap + laplacian_smooth_loss(u_t_flat, node_edge_b)
            Llap = Llap / T_b
        else:
            Llap = torch.zeros((), device=device)

        L = Lu + LAMBDA_S*Ls + LAMBDA_LAP*Llap + LAMBDA_RF2*Lrf2

        total_L     += float(L.detach().cpu())
        total_Lu    += float(Lu.detach().cpu())
        total_Ls    += float(Ls.detach().cpu())
        total_Lrf2  += float(Lrf2.detach().cpu())
        total_Llap  += float(Llap.detach().cpu())
        steps       += 1

    val_L    = total_L    / max(1, steps)
    val_Lu   = total_Lu   / max(1, steps)
    val_Ls   = total_Ls   / max(1, steps)
    val_Lrf2 = total_Lrf2 / max(1, steps)
    val_Llap = total_Llap / max(1, steps)

    return val_L, val_Lu, val_Ls, val_Lrf2, val_Llap


# ===================== 5. Training main function (batch + global norm + validation) =====================

def train():
    set_seed(SEED)
    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    files = list_npz(NPZ_DIR, NPZ_PATH)
    assert len(files) > 0, "No .npz files found. Please check NPZ_DIR / NPZ_PATH."

    print("Found %d cases:" % len(files))
    for f in files[:5]:
        print("  ", f)
    if len(files) > 5:
        print("  ...")

    # ===== Global normalization over all cases =====
    if NORM_SCOPE == "global":
        print(">>> Using GLOBAL normalization, computing stats over ALL cases ...")
        gstats = compute_global_stats(files, frame_stride=FRAME_STRIDE, s_elem_mode=S_PREV_MODE)
        print("Global stats:")
        print("  mu_coord:", gstats["mu_coord"])
        print("  sig_coord:", gstats["sig_coord"])
        print("  mu_u:", gstats["mu_u"])
        print("  sig_u:", gstats["sig_u"])
        print("  mu_v:", gstats["mu_v"])
        print("  sig_v:", gstats["sig_v"])
        print("  mu_s:", gstats["mu_s"])
        print("  sig_s:", gstats["sig_s"])
    else:
        print(">>> Using PER-CASE normalization")
        gstats = None

    # ===== Train/validation split =====
    if VAL_RATIO > 0.0 and len(files) >= 2:
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_RATIO))
        val_files = files[:n_val]
        train_files = files[n_val:]
        print(f"Train cases: {len(train_files)}, Val cases: {len(val_files)} (VAL_RATIO={VAL_RATIO})")
    else:
        train_files = files
        val_files = []
        print("No separate validation set (VAL_RATIO <= 0 or too few cases).")

    # Use the first training case to initialize graphs and model (assume static mesh/graph)
    case0 = load_case(train_files[0], device, gstats,
                      edge_mode_node=EDGE_MODE_NODE,
                      elem_adj_mode=ELEM_ADJ_MODE,
                      s_elem_mode=S_PREV_MODE,
                      frame_stride=FRAME_STRIDE,
                      norm_scope=NORM_SCOPE)
    X0, node_e0, elem_e0, enodes0 = case0["X_node"], case0["node_edge"], case0["elem_edge"], case0["elem_nodes_idx"]
    T0, N0, F0 = X0.shape
    Ne0 = enodes0.size(0)
    print(f"[Init] T={T0}, N={N0}, Ne={Ne0}, node_edges={node_e0.shape[1]}, elem_edges={elem_e0.shape[1]}")

    model = DualGraphModel(in_node_f=F0,
                           hidden_node=HIDDEN_NODE,
                           hidden_elem=HIDDEN_ELEM,
                           K_node=K_HOP_NODE,
                           K_elem=K_HOP_ELEM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    loss_hist = []
    best_val = None
    base, ext = os.path.splitext(SAVE_PATH)
    best_path = base + "_best" + ext

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_files)
        ep_L = ep_Lu = ep_Ls = ep_Llap = ep_Lrf2 = 0.0
        steps = 0          # number of batches
        ep_cases = 0       # number of cases

        # Linear decay of teacher forcing (if P_TF_START=0, p_tf is always 0)
        if K_STEPS_TF > 0 and epoch <= K_STEPS_TF:
            p_tf = P_TF_START * (1.0 - (epoch - 1) / K_STEPS_TF)
        else:
            p_tf = 0.0

        # ===== Batch training =====
        model.train()
        for start in range(0, len(train_files), BATCH_CASES):
            batch_files = train_files[start:start + BATCH_CASES]
            B = len(batch_files)

            # Load this group of cases and stack into batch
            X_list   = []
            Y_u_list = []
            Y_se_list= []
            Y_rf2_list = []

            for f in batch_files:
                case = load_case(f, device, gstats,
                                 edge_mode_node=EDGE_MODE_NODE,
                                 elem_adj_mode=ELEM_ADJ_MODE,
                                 s_elem_mode=S_PREV_MODE,
                                 frame_stride=FRAME_STRIDE,
                                 norm_scope=NORM_SCOPE)
                X, Y_u, Y_se, Y_rf2 = case["X_node"], case["Y_u"], case["Y_s_elem"], case["Y_rf2"]

                X_list.append(X)          # (T,N,F)
                Y_u_list.append(Y_u)      # (T,N,3)
                Y_se_list.append(Y_se)    # (T,Ne)
                Y_rf2_list.append(Y_rf2)  # (T,)

            # Stack into batch tensors: (T,B,N,...) / (T,B,Ne) / (T,B)
            X_batch    = torch.stack(X_list,   dim=1)   # (T,B,N,F)
            Y_u_batch  = torch.stack(Y_u_list, dim=1)   # (T,B,N,3)
            Y_se_batch = torch.stack(Y_se_list,dim=1)   # (T,B,Ne)
            Y_rf2_batch= torch.stack(Y_rf2_list,dim=1)  # (T,B)

            # Handle s_prev flag
            if NO_S_FEEDBACK or S_PREV_MODE == "none":
                X_batch[:, :, :, 11] = 0.0
                eff_s_prev_mode = "none"
            else:
                eff_s_prev_mode = S_PREV_MODE

            opt.zero_grad()

            # One forward pass for B cases in parallel
            u_hat, s_elem_hat, rf2_hat = model.forward_batch(
                X_batch,
                node_e0, elem_e0, enodes0,
                Y_u=Y_u_batch, Y_s_elem=Y_se_batch,
                teacher_forcing_prob=p_tf,
                s_prev_mode=eff_s_prev_mode
            )

            # Loss: mean over T,B,N
            Lu = F.mse_loss(u_hat, Y_u_batch, reduction="mean")
            if USE_SMOOTHL1:
                Ls = huber_loss(s_elem_hat, Y_se_batch, beta=0.5, reduction="mean")
            else:
                Ls = F.mse_loss(s_elem_hat, Y_se_batch, reduction="mean")

            Lrf2 = F.mse_loss(rf2_hat, Y_rf2_batch, reduction="mean")

            if LAMBDA_LAP > 0.0:
                Llap = 0.0
                # u_hat: (T,B,N,3) -> compute Laplacian loss per time step
                T_b = u_hat.size(0)
                node_edge_b = build_batched_edge_index(node_e0, B, N0)
                for t in range(T_b):
                    u_t_flat = u_hat[t].reshape(B * N0, 3)
                    Llap = Llap + laplacian_smooth_loss(u_t_flat, node_edge_b)
                Llap = Llap / T_b
            else:
                Llap = torch.zeros((), device=device)

            L = Lu + LAMBDA_S*Ls + LAMBDA_LAP*Llap + LAMBDA_RF2*Lrf2
            L.backward()

            if CLIP_NORM and CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

            ep_L   += float(L.detach().cpu())
            ep_Lu  += float(Lu.detach().cpu())
            ep_Ls  += float(Ls.detach().cpu())
            ep_Llap+= float(Llap.detach().cpu())
            ep_Lrf2+= float(Lrf2.detach().cpu())
            steps  += 1
            ep_cases += B

        train_L   = ep_L / max(1, steps)
        train_Lu  = ep_Lu / max(1, steps)
        train_Ls  = ep_Ls / max(1, steps)
        train_Lrf2= ep_Lrf2 / max(1, steps)
        train_Llap= ep_Llap / max(1, steps)
        loss_hist.append(train_L)

        msg = (
            f"[{epoch:4d}/{EPOCHS}] "
            f"Train L={train_L:.6e}  "
            f"Lu={train_Lu:.6e}  "
            f"Ls={train_Ls:.6e}  "
            f"Lrf2={train_Lrf2:.6e}  "
            f"Llap={train_Llap:.6e}  "
            f"p_tf={p_tf:.3f}  cases={ep_cases}"
        )

        # ===== Validation =====
        if len(val_files) > 0 and VAL_INTERVAL > 0 and (epoch % VAL_INTERVAL == 0):
            val_res = evaluate(model, val_files, device, gstats, node_e0, elem_e0, enodes0)
            if val_res is not None:
                val_L, val_Lu, val_Ls, val_Lrf2, val_Llap = val_res
                msg += (
                    f"  |  Val L={val_L:.6e} "
                    f"Lu={val_Lu:.6e} "
                    f"Ls={val_Ls:.6e} "
                    f"Lrf2={val_Lrf2:.6e} "
                    f"Llap={val_Llap:.6e}"
                )
                # Save best model
                if SAVE_BEST:
                    if (best_val is None) or (val_L < best_val):
                        best_val = val_L
                        ckpt_best = dict(
                            model_state=model.state_dict(),
                            in_node_f=F0,
                            hidden_node=HIDDEN_NODE,
                            hidden_elem=HIDDEN_ELEM,
                            K_node=K_HOP_NODE,
                            K_elem=K_HOP_ELEM
                        )
                        torch.save(ckpt_best, best_path)
                        msg += f"  [BEST model saved -> {best_path}]"

        print(msg)

    # Save final model (last epoch)
    ckpt = dict(
        model_state=model.state_dict(),
        in_node_f=F0,
        hidden_node=HIDDEN_NODE,
        hidden_elem=HIDDEN_ELEM,
        K_node=K_HOP_NODE,
        K_elem=K_HOP_ELEM
    )
    torch.save(ckpt, SAVE_PATH)
    print("Final model saved:", SAVE_PATH)
    if SAVE_BEST and best_val is not None:
        print(f"Best val loss = {best_val:.6e}, best model at: {best_path}")


# ===================== 6. main: read YAML and start training =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualGraph GNN with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="joint_us_config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    cfg = load_config_yaml(args.config)
    apply_config(cfg)

    print("Loaded config from:", args.config)
    print("NPZ_DIR   =", NPZ_DIR)
    print("NORM_SCOPE=", NORM_SCOPE)
    print("EPOCHS    =", EPOCHS)
    print("LR        =", LR)
    print("VAL_RATIO =", VAL_RATIO)
    print("VAL_INTERVAL =", VAL_INTERVAL)
    print("SAVE_BEST =", SAVE_BEST)

    train()
