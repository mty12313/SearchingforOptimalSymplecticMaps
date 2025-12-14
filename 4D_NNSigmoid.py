# symp4d_sigmoid.py
import argparse, math
import torch
import torch.nn as nn

# ----------------------------
# Config & Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 4D Boundary Sampler (Ellipsoid surface)
# ----------------------------
class Ellipsoid4D:
    def __init__(self, a=7.0, center=(0.0, 0.0, 0.0, 0.0)):
        self.r = torch.tensor([1.0, math.sqrt(a), 1.0, math.sqrt(a)], dtype=torch.float32)
        self.c = torch.tensor(center, dtype=torch.float32)


    @torch.no_grad()
    def boundary_points(self, k=6000, seed=11, device=device):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        u = torch.randn((k, 4), generator=g, device=device)        # [k,4]
        u = u / (u.norm(dim=1, keepdim=True) + 1e-12)              # on S^3
        pts = u * self.r.to(device) + self.c.to(device)            # scale & shift
        # return x1,x2,y1,y2 as 1D tensors
        return pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

class LagrangianTorus4D:
    def __init__(self, r1=0.5, r2=0.5, center1=(0.0, 0.0), center2=(0.0, 0.0)):
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.c1 = torch.tensor(center1, dtype=torch.float32)  # (cx1, cy1)
        self.c2 = torch.tensor(center2, dtype=torch.float32)  # (cx2, cy2)

    @torch.no_grad()
    def boundary_points(self, k=6000, seed=11, device=device):
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        theta1 = 2 * math.pi * torch.rand((k,), generator=g, device=device)
        theta2 = 2 * math.pi * torch.rand((k,), generator=g, device=device)

        x1 = self.r1 * torch.cos(theta1) + self.c1.to(device)[0]
        y1 = self.r1 * torch.sin(theta1) + self.c1.to(device)[1]
        x2 = self.r2 * torch.cos(theta2) + self.c2.to(device)[0]
        y2 = self.r2 * torch.sin(theta2) + self.c2.to(device)[1]
        return x1, x2, y1, y2

class Polydisk4D:
    def __init__(self, a1=1.0, a2=5.0, center1=(0.0, 0.0), center2=(0.0, 0.0)):
        self.R1 = math.sqrt(a1 / math.pi)
        self.R2 = math.sqrt(a2 / math.pi)
        self.c1 = torch.tensor(center1, dtype=torch.float32)  # (cx1, cy1)
        self.c2 = torch.tensor(center2, dtype=torch.float32)  # (cx2, cy2)

    @torch.no_grad()
    def boundary_points(self, k=6000, seed=11, device=device, corner_frac=0.0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        m_corner = int(k * max(0.0, min(1.0, corner_frac)))
        k_remain = k - m_corner
        m_side1 = k_remain // 2
        m_side2 = k_remain - m_side1

        th1 = 2 * math.pi * torch.rand((m_side1,), generator=g, device=device)
        th2 = 2 * math.pi * torch.rand((m_side1,), generator=g, device=device)
        r2  = torch.sqrt(torch.rand((m_side1,), generator=g, device=device)) * self.R2

        x1_1 = self.R1 * torch.cos(th1) + self.c1.to(device)[0]
        y1_1 = self.R1 * torch.sin(th1) + self.c1.to(device)[1]
        x2_1 = r2 * torch.cos(th2) + self.c2.to(device)[0]
        y2_1 = r2 * torch.sin(th2) + self.c2.to(device)[1]

        th1b = 2 * math.pi * torch.rand((m_side2,), generator=g, device=device)
        th2b = 2 * math.pi * torch.rand((m_side2,), generator=g, device=device)
        r1b  = torch.sqrt(torch.rand((m_side2,), generator=g, device=device)) * self.R1

        x1_2 = r1b * torch.cos(th1b) + self.c1.to(device)[0]
        y1_2 = r1b * torch.sin(th1b) + self.c1.to(device)[1]
        x2_2 = self.R2 * torch.cos(th2b) + self.c2.to(device)[0]
        y2_2 = self.R2 * torch.sin(th2b) + self.c2.to(device)[1]

        if m_corner > 0:
            th1c = 2 * math.pi * torch.rand((m_corner,), generator=g, device=device)
            th2c = 2 * math.pi * torch.rand((m_corner,), generator=g, device=device)
            x1_c = self.R1 * torch.cos(th1c) + self.c1.to(device)[0]
            y1_c = self.R1 * torch.sin(th1c) + self.c1.to(device)[1]
            x2_c = self.R2 * torch.cos(th2c) + self.c2.to(device)[0]
            y2_c = self.R2 * torch.sin(th2c) + self.c2.to(device)[1]

            x1 = torch.cat([x1_1, x1_2, x1_c], dim=0)
            y1 = torch.cat([y1_1, y1_2, y1_c], dim=0)
            x2 = torch.cat([x2_1, x2_2, x2_c], dim=0)
            y2 = torch.cat([y2_1, y2_2, y2_c], dim=0)
        else:
            x1 = torch.cat([x1_1, x1_2], dim=0)
            y1 = torch.cat([y1_1, y1_2], dim=0)
            x2 = torch.cat([x2_1, x2_2], dim=0)
            y2 = torch.cat([y2_1, y2_2], dim=0)

        return x1, x2, y1, y2

# ----------------------------
# Tiny MLP with Sigmoid for f,g
# ----------------------------
class TinyMLP(nn.Module):
    # hidden = number of neurons in each hidden layer (controls model capacity)
    # in_dim = input dimension (number of input features)
    def __init__(self, in_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [N,2]
        return self.net(x) # [N,1]

# ---------- Funciton s(a) ----------
def s_piecewise_torch(a_tensor):
    a = a_tensor.clone().detach().to(dtype=torch.float32)
    out = torch.full_like(a, float('nan'))

    amax = torch.nan_to_num(a, nan=0.0).max().item()
    kmax = max(1, int(math.sqrt(max(amax, 0.0)/2.0) + 3)) + 10

    for k in range(1, kmax+1):
        lo = 2.0*(k*k - k + 1)
        hi = 2.0*(k*k + k + 1)
        mask = (a >= lo) & (a <= hi)
        if mask.any():
            out[mask] = (a[mask] - 2.0)/(2.0*k) + (k + 2.0)
    return out


# ----------------------------
# 4-different Maps
# ----------------------------
class ASymplecticR4_NN(nn.Module):
    """A) F(y1,y2): (x1,y1,x2,y2) -> (x1 + dF/dy1, y1, x2 + dF/dy2, y2)"""
    def __init__(self, hidden=32):
        super().__init__()
        self.F = TinyMLP(in_dim=2, hidden=hidden)

    def forward(self, x1, x2, y1, y2):
        Y = torch.stack([y1, y2], dim=1).requires_grad_(True)         # [N,2]
        F_val = self.F(Y)                                              # [N,1] or [N]
        dF_dY = torch.autograd.grad(F_val.sum(), Y, create_graph=True)[0]  # [N,2]
        x1 = x1 + dF_dY[:, 0]
        x2 = x2 + dF_dY[:, 1]
        return x1, x2, y1, y2

class BSymplecticR4_NN(nn.Module):
    """B) F(x1,x2): (x1,y1,x2,y2) -> (x1, y1 - dF/dx1, x2, y2 - dF/dx2)"""
    def __init__(self, hidden=32):
        super().__init__()
        self.F = TinyMLP(in_dim=2, hidden=hidden)

    def forward(self, x1, x2, y1, y2):
        X = torch.stack([x1, x2], dim=1).requires_grad_(True)         # [N,2]
        F_val = self.F(X)                                             # [N,1] or [N]
        dF_dX = torch.autograd.grad(F_val.sum(), X, create_graph=True)[0]  # [N,2]
        y1 = y1 - dF_dX[:, 0]
        y2 = y2 - dF_dX[:, 1]
        return x1, x2, y1, y2

class CSymplecticR4_NN(nn.Module):
    """C) F(x1,y2): (x1,y1,x2,y2) -> (x1, y1 - dF/dx1, x2 + dF/dy2, y2)"""
    def __init__(self, hidden=32):
        super().__init__()
        self.F = TinyMLP(in_dim=2, hidden=hidden)  # input = (x1, y2) -> scalar

    def forward(self, x1, x2, y1, y2):
        XY = torch.stack([x1, y2], dim=1).requires_grad_(True)        # [N,2]
        F_val = self.F(XY)                                            # [N,1] or [N]
        dF = torch.autograd.grad(F_val.sum(), XY, create_graph=True)[0]    # [∂F/∂x1, ∂F/∂y2]
        y1 = y1 - dF[:, 0]   # -∂F/∂x1
        x2 = x2 + dF[:, 1]   # +∂F/∂y2
        return x1, x2, y1, y2

class DSymplecticR4_NN(nn.Module):
    """D) F(y1,x2): (x1,y1,x2,y2) -> (x1 + dF/dy1, y1, x2, y2 - dF/dx2)"""
    def __init__(self, hidden=32):
        super().__init__()
        self.F = TinyMLP(in_dim=2, hidden=hidden)  # input = [x2, y1]

    def forward(self, x1, x2, y1, y2):
        YX = torch.stack([y1, x2], dim=1).requires_grad_(True)        # [N,2]
        F_val = self.F(YX)                                            # [N,1] or [N]
        dF = torch.autograd.grad(F_val.sum(), YX, create_graph=True)[0]     # [∂F/∂y1, ∂F/∂x2]
        x1 = x1 + dF[:, 0]   # +∂F/∂y1
        y2 = y2 - dF[:, 1]   # -∂F/∂x2
        return x1, x2, y1, y2

class SymplecticCompositionR4_NN(nn.Module):
    def __init__(self, k=6, hidden=32):
        super().__init__()
        self.k = k
        self.A = nn.ModuleList([ASymplecticR4_NN(hidden) for _ in range(k)])
        self.B = nn.ModuleList([BSymplecticR4_NN(hidden) for _ in range(k)])
        self.C = nn.ModuleList([CSymplecticR4_NN(hidden) for _ in range(k)])
        self.D = nn.ModuleList([DSymplecticR4_NN(hidden) for _ in range(k)])

    def forward(self, x1,x2,y1,y2):
        for i in range(self.k-1, -1, -1):
            x1,x2,y1,y2 = self.D[i](x1,x2,y1,y2)
            x1,x2,y1,y2 = self.C[i](x1,x2,y1,y2)
            x1,x2,y1,y2 = self.B[i](x1,x2,y1,y2)
            x1,x2,y1,y2 = self.A[i](x1,x2,y1,y2)
        return x1,x2,y1,y2

# ----------------------------
# Loss: hard max radius + small center + L2
# ----------------------------
def loss_max_radius_R4(model, x1, x2, y1, y2, w_center=1e-3, w_reg=1e-6):
    X1, X2, Y1, Y2 = model(x1, x2, y1, y2)
    r = torch.sqrt(X1*X1 + X2*X2 + Y1*Y1 + Y2*Y2)
    R = r.max()  # subgradient is fine

    cx, cy, cp, cq = X1.mean(), X2.mean(), Y1.mean(), Y2.mean()
    center = w_center * (cx*cx + cy*cy + cp*cp + cq*cq)

    reg = torch.zeros((), device=X1.device)
    for p in model.parameters():
        reg = reg + (p.pow(2).sum())
    reg = w_reg * reg

    return R + center + reg, R.detach(), (cx, cy, cp, cq)


# ----------------------------
# Training Loop
# ----------------------------
def _to_torch_1d(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device).view(-1)
    import numpy as np
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device).view(-1)
    return torch.tensor(x, device=device).view(-1)

def sample_boundary(region, k, seed, device):
    if isinstance(region, (list, tuple)):
        n = len(region)
        base, extra = divmod(k, n)
        xs1=[]; xs2=[]; ys1=[]; ys2=[]
        for i, r in enumerate(region):
            ki = base + (1 if i < extra else 0)
            x1, x2, y1, y2 = r.boundary_points(k=ki, seed=seed+11*i, device=device)
            xs1.append(_to_torch_1d(x1, device))
            xs2.append(_to_torch_1d(x2, device))
            ys1.append(_to_torch_1d(y1, device))
            ys2.append(_to_torch_1d(y2, device))
        return torch.cat(xs1), torch.cat(xs2), torch.cat(ys1), torch.cat(ys2)
    else:
        x1, x2, y1, y2 = region.boundary_points(k=k, seed=seed, device=device)
        return (_to_torch_1d(x1, device),
                _to_torch_1d(x2, device),
                _to_torch_1d(y1, device),
                _to_torch_1d(y2, device))

def train(
    k=6,
    hidden=32,
    n_boundary=6000,
    n_iters=500,
    lr=1e-3,
    w_center=1e-3,
    w_reg=1e-6,
    seed=11,
    report_every=50,
    region=None,              
    model=None,
    device=None,
    grad_clip=1e3,
):
    torch.manual_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if region is None:
        region = Ellipsoid4D()

    if model is None:
        model = SymplecticCompositionR4_NN(k=k, hidden=hidden).to(device)
    else:
        model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state, best_loss = None, float("inf")

    for it in range(1, n_iters + 1):
        x1, x2, y1, y2 = sample_boundary(region, n_boundary, seed=seed + it, device=device)

        opt.zero_grad(set_to_none=True)
        L, R, (cx, cy, cp, cq) = loss_max_radius_R4(
            model, x1, x2, y1, y2, w_center=w_center, w_reg=w_reg
        )
        L.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        Lf = float(L.detach().cpu())
        if Lf < best_loss:
            best_loss = Lf
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

        if it % report_every == 0 or it == 1 or it == n_iters:
            cap = math.pi * float(R)**2  # capacity corresponding to this R
            print(f"[{it:4d}] L={Lf:.6f}  R={float(R):.6f}  cap={cap:.6f}  "
                f"cent=({float(cx):+.2e},{float(cy):+.2e},{float(cp):+.2e},{float(cq):+.2e})")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-iters", type=int, default=1000)
    ap.add_argument("--n-boundary", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-center", type=float, default=1e-3)
    ap.add_argument("--w-reg", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args()

    #region = Polydisk4D(a1=1.0, a2=6.0)
    region = Ellipsoid4D(a=3)

    train(k=args.k, hidden=args.hidden,
          n_boundary=args.n_boundary, n_iters=args.n_iters, lr=args.lr,
          w_center=args.w_center, w_reg=args.w_reg,
          seed=args.seed, report_every=args.report_every,
          region=region)


if __name__ == "__main__":
    main()
