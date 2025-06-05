#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU

# ===== 0. Path Settings =====
NPZ_PATH = r"D:\BaiduNetdiskDownload\npz\Job-t2.npz"
MODEL_U  = r"C:\py\uvel_selected.pth"
OUT_NPZ  = NPZ_PATH.replace('.npz', '_pred_selected.npz')

# ===== 1. Initialization =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ===== 2. Helper: Build Graph Edges (edge_index) =====
def build_edge_index(conn, labels):
    idx = {lbl: i for i, lbl in enumerate(labels)}
    edges = set()
    for elem in conn:
        if len(elem) == 8:
            b, t = elem[:4], elem[4:]
            for k in range(4):
                edges |= {
                    (idx[b[k]], idx[b[(k+1) % 4]]),
                    (idx[t[k]], idx[t[(k+1) % 4]]),
                    (idx[b[k]], idx[t[k]])
                }
        else:
            m = len(elem)
            for k in range(m):
                i0, i1 = elem[k], elem[(k+1) % m]
                edges.add((idx[i0], idx[i1]))
    edges |= {(j, i) for i, j in edges}
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

# ===== 3. Load Original NPZ =====
d = np.load(NPZ_PATH)
coord       = d["node_coords"].astype(np.float32)
conn        = d["connectivity"].astype(np.int32)
labels      = d["node_labels"].astype(np.int32)
times_full  = d["frame_times"].astype(np.float32)
disp_full   = d["disp"].astype(np.float32)
surf_nodes  = d.get("SURF1_NODE_LABELS", None)

N, T_full = coord.shape[0], disp_full.shape[0]
edge = build_edge_index(conn, labels).to(device)

# ===== 4. Compute True Velocity (vel_full) =====
dt_full  = np.diff(times_full, prepend=times_full[0])
vel_full = np.zeros_like(disp_full, dtype=np.float32)
vel_full[1:] = (disp_full[1:] - disp_full[:-1]) / dt_full[1:, None, None]

# ===== 5. Build force_flag =====
if surf_nodes is None:
    flag = np.zeros(N, dtype=np.float32)
else:
    surf_set = set(int(x) for x in surf_nodes)
    flag = np.array([1.0 if int(lbl) in surf_set else 0.0 for lbl in labels], dtype=np.float32)
flag_t = torch.tensor(flag, device=device).unsqueeze(1)

# ===== 6. Keep Only Frames Used in Training =====
selected_indices = [0, 4, 8, 12, 16, 20]
selected_indices = [i for i in selected_indices if i < T_full]

times_sel = times_full[selected_indices]
disp_sel  = disp_full[selected_indices]
vel_sel   = vel_full[selected_indices]
T_sel     = len(times_sel)

# ===== 7. Build Input X_sel: (T_sel, N, 11) =====
X_sel = torch.zeros((T_sel, N, 11), dtype=torch.float32, device=device)
coord_t = torch.tensor(coord, device=device)
for ti in range(T_sel):
    alpha_col = torch.full((N, 1), times_sel[ti], device=device)
    if ti == 0:
        u_prev = torch.zeros((N, 3), device=device)
        v_prev = torch.zeros((N, 3), device=device)
    else:
        u_prev = torch.tensor(disp_sel[ti - 1], device=device)
        v_prev = torch.tensor(vel_sel[ti - 1], device=device)
    X_sel[ti] = torch.cat([coord_t, u_prev, alpha_col, flag_t, v_prev], dim=1)

# ===== 8. Define Model: Single-layer GConvGRU Version (Same as Training) =====
class AutoregU(nn.Module):
    def __init__(self, in_f=11, h=256, out_f=3, K=1):
        super().__init__()
        self.gru  = GConvGRU(in_f, h, K=K)
        self.head = nn.Linear(h, out_f)

    def forward(self, X_seq, edge, Y_u=None, p_tf=0.0):
        X, h, outs = [x.clone() for x in X_seq], None, []
        T = len(X)
        for t in range(T):
            h = self.gru(X[t], edge_index=edge, H=h)
            u = self.head(h)
            outs.append(u)
            if t < T - 1:
                dt_t = X_seq[t + 1][:, 6] - X_seq[t][:, 6]
                X[t + 1][:, 3:6] = u.detach()
                v = (u.detach() - X[t][:, 3:6]) / dt_t.unsqueeze(1)
                X[t + 1][:, 8:11] = v
        return torch.stack(outs)

model = AutoregU().to(device)

# ===== 9. Load Trained Model Weights & Normalization Parameters =====
ckpt = torch.load(MODEL_U, map_location=device)
model.load_state_dict(ckpt["model"])
mu_x, std_x = ckpt["mu_x"].to(device), ckpt["std_x"].to(device)
mu_u, std_u = ckpt["mu_u"].to(device), ckpt["std_u"].to(device)

# ===== 10. Normalize Input X_sel =====
X_sel_n = (X_sel - mu_x) / std_x

# ===== 11. Predict: Output Only for These Frames (T_sel) =====
model.eval()
with torch.no_grad():
    X_list = [X_sel_n[t] for t in range(T_sel)]
    pred_sel_n = model(X_list, edge, None, p_tf=0.0)

pred_sel = pred_sel_n * std_u + mu_u

# ===== 12. Save Results =====
out_dict = {
    "node_coords": coord,
    "connectivity": conn,
    "frame_times": times_sel,
    "disp": disp_sel,
    "node_labels": labels,
    "SURF1_NODE_LABELS": surf_nodes,
    "pred_u": pred_sel.cpu().numpy()
}
np.savez_compressed(OUT_NPZ, **out_dict)
print("save at:", OUT_NPZ)
