import os, glob, random, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
# -----------------------------

# ===== 0. hyper-params & paths =====
NPZ_DIR        = "/root/autodl-tmp"                # folder with .npz files
HIDDEN         = 128                               # hidden dim
EPOCHS         = 1500
TF_PHASE       = 0                               # 0-100  : p_tf = 1
DECAY_PHASE    = 0                               # 101-300: linear decay
LR             = 1e-3
CLIP_NORM      = 0.5
SAVE_PATH      = "/root/autodl-tmp/uvel_selected_tf.pth"
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED);  np.random.seed(SEED);  random.seed(SEED)

# ===== 1. helper funcs =====
def load_npz(path):
    d = np.load(path)
    return (d["node_coords"].astype(np.float32),
            d["connectivity"].astype(np.int32),
            d["frame_times"].astype(np.float32),
            d["disp"].astype(np.float32),
            d["node_labels"].astype(np.int32),
            d.get("SURF1_NODE_LABELS", None))

def build_edge_index(conn, labels):
    idx = {lbl:i for i,lbl in enumerate(labels)}
    edges = set()
    for elem in conn:
        m = len(elem)
        if m == 8:                                   # C3D8
            b,t = elem[:4], elem[4:]
            for k in range(4):
                edges |= {(idx[b[k]],idx[b[(k+1)%4]]),
                          (idx[t[k]],idx[t[(k+1)%4]]),
                          (idx[b[k]],idx[t[k]])}
        else:                                        # e.g. C3D4
            for k in range(m):
                i0,i1 = elem[k], elem[(k+1)%m]
                edges.add((idx[i0],idx[i1]))
    edges |= {(j,i) for i,j in edges}
    return torch.tensor(list(edges),dtype=torch.long).t().contiguous()

# ===== 2. read data & build graphs =====
sel_idx = [0,2,4,6,8,10,12,14,16,18,20]
graphs=[]
for fp in sorted(glob.glob(os.path.join(NPZ_DIR,"*.npz"))):
    coord,conn,t_full,disp_full,labels,surf = load_npz(fp)
    idx   = [i for i in sel_idx if i < len(t_full)]
    times = t_full[idx]                  # (T,)
    disp  = disp_full[idx]               # (T,N,3)
    T,N   = disp.shape[0], coord.shape[0]

    # velocity
    dt = np.diff(times,prepend=times[0])
    vel = np.zeros_like(disp)
    vel[1:] = (disp[1:]-disp[:-1]) / dt[1:,None,None]

    # flag
    if surf is None:
        flag = np.zeros(N,dtype=np.float32)
    else:
        sset = set(int(x) for x in surf)
        flag = np.array([1.0 if int(l) in sset else 0.0 for l in labels],dtype=np.float32)
    flag_t = torch.tensor(flag,device=DEVICE).unsqueeze(1)

    # build X
    X = torch.zeros((T,N,11),dtype=torch.float32,device=DEVICE)
    coord_t = torch.tensor(coord,device=DEVICE)
    for t in range(T):
        alpha = torch.full((N,1),times[t],device=DEVICE)
        if t==0:
            u_prev=v_prev=torch.zeros((N,3),device=DEVICE)
        else:
            u_prev=torch.tensor(disp[t-1],device=DEVICE)
            v_prev=torch.tensor(vel[t-1],device=DEVICE)
        X[t]=torch.cat([coord_t,u_prev,alpha,flag_t,v_prev],dim=1)

    graphs.append(dict(
        X=X,
        Y_u=torch.tensor(disp,device=DEVICE),
        edge=build_edge_index(conn,labels).to(DEVICE),
        times=times
    ))

print(f"Loaded {len(graphs)} graphs, each with T={graphs[0]['X'].shape[0]} frames.")

# ===== 3. normalization =====
X_all = torch.cat([g["X"].reshape(-1,11).cpu() for g in graphs])
Y_all = torch.cat([g["Y_u"].reshape(-1,3).cpu() for g in graphs])
mu_x,std_x = X_all.mean(0), X_all.std(0);  mu_u,std_u = Y_all.mean(0), Y_all.std(0)
mu_x[6]=mu_x[7]=0;  std_x[6]=std_x[7]=1
std_x[std_x==0]=1;  std_u[std_u==0]=1
mu_x,std_x,mu_u,std_u=[t.to(DEVICE) for t in (mu_x,std_x,mu_u,std_u)]
for g in graphs:
    g["X"]   = (g["X"]-mu_x)/std_x
    g["Y_u"] = (g["Y_u"]-mu_u)/std_u

# ===== 4. model =====
class AutoregU(nn.Module):
    def __init__(self,in_f=11,h=HIDDEN,out_f=3,K=1):
        super().__init__()
        self.gru  = GConvGRU(in_f,h,K=K)
        self.head = nn.Linear(h,out_f)
    def forward(self,X_seq,edge,Y_u=None,p_tf=0.0):
        X=[x.clone() for x in X_seq]; h=None; outs=[]
        T=len(X)
        for t in range(T):
            h=self.gru(X[t],edge_index=edge,H=h)
            u=self.head(h); outs.append(u)
            if t<T-1:
                if (Y_u is not None) and (random.random()<p_tf):
                    next_u=Y_u[t]
                else:
                    next_u=u.detach()
                dt = X_seq[t+1][:,6]-X_seq[t][:,6]
                X[t+1][:,3:6]=next_u
                X[t+1][:,8:11]=(next_u-X[t][:,3:6])/dt.unsqueeze(1)
        return torch.stack(outs)

model=AutoregU().to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)

# ===== 5. training =====
loss_hist=[]
for epoch in range(1,EPOCHS+1):
    # teacher-forcing schedule
    if   epoch<=TF_PHASE:                p_tf=1.0
    elif epoch<=TF_PHASE+DECAY_PHASE:    p_tf=1.0-(epoch-TF_PHASE)/DECAY_PHASE
    else:                                p_tf=0.0

    model.train(); epoch_loss=0.0
    for g in graphs:
        X_list=[g["X"][t] for t in range(len(g["X"]))]
        out=model(X_list,g["edge"],g["Y_u"],p_tf)
        loss=F.mse_loss(out,g["Y_u"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_NORM)
        opt.step(); epoch_loss+=loss.item()
    loss_hist.append(epoch_loss)
    if epoch==1 or epoch%10==0:
        print(f"Ep {epoch:04d} | p_tf={p_tf:.3f} | loss={epoch_loss:.6f}")

# ===== 6. save =====
torch.save(dict(model=model.state_dict(),
                mu_x=mu_x.cpu(),std_x=std_x.cpu(),
                mu_u=mu_u.cpu(),std_u=std_u.cpu()),
           SAVE_PATH)
print("Saved:",SAVE_PATH)

# ===== 7. loss curve =====
plt.figure(figsize=(6,4)); plt.plot(loss_hist); plt.yscale('log')
plt.xlabel("Epoch"); plt.ylabel("Sum MSE (norm)"); plt.grid(True)
plt.tight_layout(); plt.savefig("uvel_selected_tf_loss.png")
print("Loss curve saved as uvel_selected_tf_loss.png")
