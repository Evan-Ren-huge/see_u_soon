Two-layer GConvGRU training script (based on uvel.py), using only frames 0, 4, 8, 12, 16, 20 for autoregressive prediction.
Input features: (x, y, z, U_prev_x,y,z, α, force_flag, V_prev_x,y,z)
Output: (Ux, Uy, Uz)
Only these 6 frames are selected: indices 0, 4, 8, 12, 16, 20 (out‐of‐range indices are filtered out).
All other training logic follows uvel.py.


import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

# ===== 0. Hyperparameters & Paths =====
NPZ_DIR   = "/root/autodl-tmp"
HIDDEN    = 128
EPOCHS    = 2500
N_DECAY   = 0
CLIP_NORM = 0.5
LR        = 1e-3
SAVE_PATH = "/root/autodl-tmp/uvel_selected.pth"
SEED      = 42
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===== 1. Helper Functions =====
def load_npz(path):
    d = np.load(path)
    return (
        d["node_coords"].astype(np.float32),
        d["connectivity"].astype(np.int32),
        d["frame_times"].astype(np.float32),
        d["disp"].astype(np.float32),
        d["node_labels"].astype(np.int32),
        d.get("SURF1_NODE_LABELS", None)
    )

def build_edge_index(conn, labels):
    idx = {lbl: i for i, lbl in enumerate(labels)}
    edges = set()
    for elem in conn:
        m = len(elem)
        if m == 8:
            b, t = elem[:4], elem[4:]
            for k in range(4):
                edges |= {
                    (idx[b[k]], idx[b[(k+1) % 4]]),
                    (idx[t[k]], idx[t[(k+1) % 4]]),
                    (idx[b[k]], idx[t[k]])
                }
        else:
            for k in range(m):
                i0, i1 = elem[k], elem[(k + 1) % m]
                edges.add((idx[i0], idx[i1]))
    edges |= {(j, i) for i, j in edges}
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index

# ===== 2. Load .npz and Assemble Graphs (keeping only frames 0,4,8,12,16,20) =====
graphs = []
for fn in sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz"))):
    coord, conn, times_full, disp_full, labels, surf_nodes = load_npz(fn)
    N, T_full = coord.shape[0], disp_full.shape[0]
    selected_indices = [0, 4, 8, 12, 18, 20]
    selected_indices = [i for i in selected_indices if i < T_full]
    times = times_full[selected_indices]
    disp  = disp_full[selected_indices]
    T     = len(times)
    edge_index = build_edge_index(conn, labels).to(DEVICE)
    dt  = np.zeros(T, dtype=np.float32)
    dt[0] = 0.0
    for i in range(1, T):
        dt[i] = times[i] - times[i - 1]
    vel = np.zeros((T, N, 3), dtype=np.float32)
    for i in range(1, T):
        vel[i] = (disp[i] - disp[i - 1]) / dt[i]
    if surf_nodes is None:
        flag = np.zeros(N, dtype=np.float32)
    else:
        surf_set = set(int(x) for x in surf_nodes)
        flag = np.array([1.0 if int(lbl) in surf_set else 0.0 for lbl in labels], dtype=np.float32)
    flag_t = torch.tensor(flag, device=DEVICE).unsqueeze(1)
    X_sel = torch.zeros((T, N, 11), dtype=torch.float32, device=DEVICE)
    coord_t = torch.tensor(coord, device=DEVICE)
    for ti in range(T):
        alpha_col = torch.full((N, 1), times[ti], device=DEVICE)
        if ti == 0:
            u_prev = torch.zeros((N, 3), device=DEVICE)
            v_prev = torch.zeros((N, 3), device=DEVICE)
        else:
            u_prev = torch.tensor(disp[ti - 1], device=DEVICE)
            v_prev = torch.tensor(vel[ti - 1], device=DEVICE)
        X_sel[ti] = torch.cat([coord_t, u_prev, alpha_col, flag_t, v_prev], dim=1)
    Y_sel = torch.tensor(disp, device=DEVICE)
    graphs.append({
        "X": X_sel,
        "edge": edge_index,
        "Y_u": Y_sel,
        "times": times
    })

print(f"Loaded {len(graphs)} npz files; each has T_sel = {len(graphs[0]['times'])} selected frames.")

# ===== 3. Compute μ/σ for Input/Output Normalization =====
feat_list = []
out_list  = []
for g in graphs:
    feat_list.append(g["X"].reshape(-1, 11).cpu())
    out_list.append(g["Y_u"].reshape(-1, 3).cpu())
feat_all = torch.cat(feat_list, dim=0)
out_all  = torch.cat(out_list,  dim=0)
mu_x  = feat_all.mean(0)
std_x = feat_all.std(0)
mu_u  = out_all.mean(0)
std_u = out_all.std(0)
mu_x[6]  = 0.0; std_x[6]  = 1.0
mu_x[7]  = 0.0; std_x[7]  = 1.0
std_x[std_x == 0] = 1.0
std_u[std_u == 0] = 1.0
mu_x  = mu_x.to(DEVICE)
std_x = std_x.to(DEVICE)
mu_u  = mu_u.to(DEVICE)
std_u = std_u.to(DEVICE)
for g in graphs:
    g["X"]   = (g["X"] - mu_x) / std_x
    g["Y_u"] = (g["Y_u"] - mu_u) / std_u

# ===== 4. Define Model: Two-layer GConvGRU + Output Linear Head =====
class AutoregU(nn.Module):
    def __init__(self, in_f=11, h=HIDDEN, out_f=3, K=1):
        super().__init__()
        self.gru  = GConvGRU(in_f, h, K=K)
        self.head = nn.Linear(h, out_f)

    def forward(self, X_seq, edge, Y_u=None, p_tf=0.0):
        X = [x.clone() for x in X_seq]
        h = None
        outs = []
        T_sel = len(X)
        for t in range(T_sel):
            h = self.gru(X[t], edge_index=edge, H=h)
            u = self.head(h)
            outs.append(u)
            if t < T_sel - 1:
                dt_t = X_seq[t + 1][:, 6] - X_seq[t][:, 6]
                X[t + 1][:, 3:6] = u.detach()
                v = (u.detach() - X[t][:, 3:6]) / dt_t.unsqueeze(1)
                X[t + 1][:, 8:11] = v
        return torch.stack(outs)

model = AutoregU().to(DEVICE)

# ===== 5. Optimizer & Scheduler =====
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ===== 6. Training Loop =====
loss_hist = []
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    p_tf = 0.0
    model.train()
    for g in graphs:
        out_u = model([g["X"][t] for t in range(len(g["X"]))],
                      g["edge"],
                      g["Y_u"],
                      p_tf=p_tf)
        loss_u = F.mse_loss(out_u, g["Y_u"])
        opt.zero_grad()
        loss_u.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        epoch_loss += loss_u.item()
    loss_hist.append(epoch_loss)
    if epoch == 1 or epoch % 10 == 0:
        print(f"Ep {epoch:04d} | loss = {epoch_loss:.6f}")

# ===== 7. Per-Frame Average MSE Print =====
print("\nPer-frame average MSE (normalized space):")
model.eval()
with torch.no_grad():
    T_sel = len(graphs[0]["times"])
    pf = np.zeros(T_sel, dtype=np.float32)
    for g in graphs:
        out_u = model([g["X"][t] for t in range(T_sel)], g["edge"], None, p_tf=0.0)
        mse_per_frame = (out_u - g["Y_u"]).pow(2).mean(dim=(1, 2)).cpu().numpy()
        pf += mse_per_frame
    pf /= len(graphs)
    times_sel = graphs[0]["times"]
    for i, mse_val in enumerate(pf):
        print(f" Frame idx {i}  (time={times_sel[i]:.4f})  MSE={mse_val:.6f}")

# ===== 8. Save Model + Normalization Parameters =====
torch.save({
    "model": model.state_dict(),
    "mu_x":  mu_x.cpu(),
    "std_x": std_x.cpu(),
    "mu_u":  mu_u.cpu(),
    "std_u": std_u.cpu()
}, SAVE_PATH)
print("\nModel saved to:", SAVE_PATH)

# ===== 9. Plot and Save Training Loss Curve =====
plt.figure(figsize=(6, 4))
plt.plot(loss_hist, label="Epoch Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Sum MSE (normalized)")
plt.title("Training Loss History")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("uvel_selected_frames_loss.png")
print("Loss curve saved as uvel_selected_frames_loss.png")
