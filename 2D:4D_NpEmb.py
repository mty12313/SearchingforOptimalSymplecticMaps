#!/usr/bin/env python3
# Boundary-only: learn a 2D symplectic map (composition of shears) that
# minimizes the maximum radius of the mapped boundary.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import csv
import numpy as np
from numpy.polynomial.polynomial import polyval2d as _np_polyval2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, csv, argparse

class Shape:
    def area(self):
        return 1.0  # default unit area

    def boundary_points(self, k = 1, seed = 0):
        raise NotImplementedError("boundary_points not implemented for base Shape class.")
    
class MultipleCircles(Shape):

    def __init__(self, positions, radii):
        self.positions = positions
        self.radii = radii

    def boundary_points(self, k=1, seed=0):
        rng = np.random.default_rng(seed)
  
        xs = []
        ys = []
        for pos, radius in zip(self.positions, self.radii):
            theta = rng.uniform(0, 2 * np.pi, k // len(self.positions))
            x = radius * np.cos(theta) + pos[0]
            y = radius * np.sin(theta) + pos[1]
            xs = np.concatenate([xs, x])
            ys = np.concatenate([ys, y])
        return xs, ys
    
    def area(self):
        return np.pi * sum(r**2 for r in self.radii)

class Rectangle(Shape):

    def __init__(self, position = [0,0], verthalfwidth = 0.5, horihalfwidth = 0.25):
        self.position = position
        self.verthalfwidth = verthalfwidth
        self.horihalfwidth = horihalfwidth

    def area(self):
        return 4 * self.verthalfwidth * self.horihalfwidth

    def boundary_points(self, n, seed=0):
        x, y = square_boundary_points(n, self.verthalfwidth, self.horihalfwidth, seed=seed)
        x += self.position[0]
        y += self.position[1]
        return x, y
    
class MultipleShapes(Shape):

    def __init__(self, shapes):
        self.shapes = shapes

    def boundary_points(self, k=1, seed=0):
        xs = []
        ys = []
        for shape in self.shapes:
            x, y = shape.boundary_points(k // len(self.shapes), seed=seed)
            xs = np.concatenate([xs, x])
            ys = np.concatenate([ys, y])
        return xs, ys
    
    def area(self):
        return sum(shape.area() for shape in self.shapes)

class Keyhole(Shape):

    def __init__(self, position = [0,0], inner_radius = 1, outer_radius = 2, angle = np.pi/8):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle


    def boundary_points(self, k = 1, seed = 0):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1)
        rng3 = np.random.default_rng(seed + 2)
        rng4 = np.random.default_rng(seed + 3)
        theta1 = rng1.uniform(self.angle, 2 * np.pi - self.angle, k // 3)
        theta2 = rng2.uniform(self.angle, 2 * np.pi - self.angle, k // 3)
        t1 = rng3.uniform(0, 1, k // 6)
        t2 = rng4.uniform(0, 1, k // 6)
        
        x_in = self.inner_radius * np.cos(theta1) + self.position[0]
        y_in = self.inner_radius * np.sin(theta1) + self.position[1] 
        x_out = self.outer_radius * np.cos(theta2) + self.position[0]
        y_out = self.outer_radius * np.sin(theta2) + self.position[1]
        a1 = (self.inner_radius - t1 * (self.inner_radius - self.outer_radius)) * np.cos(self.angle) + self.position[0]
        b1 = (self.inner_radius - t1 * (self.inner_radius - self.outer_radius)) * np.sin(self.angle) + self.position[1]
        a2 = (self.outer_radius - t2 * (self.outer_radius - self.inner_radius)) * np.cos(2 * np.pi - self.angle) + self.position[0]
        b2 = (self.outer_radius - t2 * (self.outer_radius - self.inner_radius)) * np.sin(2 * np.pi - self.angle) + self.position[1]
        xs = np.concatenate([x_in, x_out, a1, a2])
        ys = np.concatenate([y_in, y_out, b1, b2])
        return xs, ys

    def area(self):
        return (self.outer_radius**2 - self.inner_radius**2)*(np.pi - self.angle)

# ---------------------------
# Boundary sampling (square)
# ---------------------------
def square_boundary_points(n, verthalfwidth=0.5, horihalfwidth = 0.25, seed=0):
    rng = np.random.default_rng(seed)
    n_side = [n // 4] * 4
    for i in range(n % 4):
        n_side[i] += 1
    xs, ys = [], []

    # Bottom: y=-h
    x = rng.uniform(-horihalfwidth, horihalfwidth, n_side[0])
    y = np.full_like(x, -verthalfwidth); xs.append(x); ys.append(y)
    # Right: x=+h
    y = rng.uniform(-verthalfwidth, verthalfwidth, n_side[1])
    x = np.full_like(y, +horihalfwidth); xs.append(x); ys.append(y)
    # Top: y=+h
    x = rng.uniform(-horihalfwidth, horihalfwidth, n_side[2])
    y = np.full_like(x, +verthalfwidth); xs.append(x); ys.append(y)
    # Left: x=-h
    y = rng.uniform(-verthalfwidth, verthalfwidth, n_side[3])
    x = np.full_like(y, -horihalfwidth); xs.append(x); ys.append(y)

    X = np.concatenate(xs); Y = np.concatenate(ys)
    return X, Y


class ASymplectic:
    # A_f: (x,y) -> (x, y + f'(x))
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        d = len(self.coeffs) - 1
        dasc = np.array([k * self.coeffs[k] for k in range(1, d + 1)], dtype=float)
        self.dcoeffs_desc = dasc[::-1]  # for np.polyval
    def __call__(self, x, y):
        return x, y + np.polyval(self.dcoeffs_desc, x)

class BSymplectic:
    # B_g: (x,y) -> (x + g'(y), y)
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        d = len(self.coeffs) - 1
        dasc = np.array([k * self.coeffs[k] for k in range(1, d + 1)], dtype=float)
        self.dcoeffs_desc = dasc[::-1]
    def __call__(self, x, y):
        return x + np.polyval(self.dcoeffs_desc, y), y

class SymplecticComposition:
    """
    Phi = A1 ∘ B1 ∘ ... ∘ Ak ∘ Bk.
    Apply to coords as: (x,y) -> Bk -> Ak -> ... -> B1 -> A1
    params: flat array length 2*k*(degree+1)
    """
    def __init__(self, params, degree, k):
        self.degree = degree; self.k = k
        self.set_params(params)

    def set_params(self, params):
        params = np.asarray(params, dtype=float)
        D = self.degree + 1
        assert params.size == 2*self.k*D, "Parameter length mismatch."
        self._params = params.copy()
        self.A, self.B = [], []
        for i in range(self.k):
            a = params[2*i*D : 2*i*D + D]
            b = params[2*i*D + D : 2*(i+1)*D]
            self.A.append(ASymplectic(a))
            self.B.append(BSymplectic(b))

    def params(self): return self._params.copy()

    def forward(self, x, y):
        for i in range(self.k-1, -1, -1):
            x, y = self.B[i](x, y)
            x, y = self.A[i](x, y)
        return x, y

# ---------------------------
# Loss: true hard max on boundary
# ---------------------------
def loss_max_radius_boundary(Phi, xB, yB, w_center=1e-3, w_reg=1e-6):
    xb, yb = Phi.forward(xB, yB)
    r = np.hypot(xb, yb)
    R = float(r.max())  # true discrete max radius on boundary

    # gentle helpers for stability
    cx = float(xb.mean()); cy = float(yb.mean())
    reg = float(np.sum(Phi.params()**2))

    L = R + w_center*(cx*cx + cy*cy) + w_reg*reg
    aux = dict(R=R, cx=cx, cy=cy, rmean=float(r.mean()), rvar=float(r.var()))
    return L, aux, (xb, yb)

# ---------------------------
# Adam + finite-difference grads
# ---------------------------
class Adam:
    def __init__(self, params, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m=np.zeros_like(params); self.v=np.zeros_like(params); self.t=0
    def step(self, params, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*(grad*grad)
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return params - self.lr * mhat / (np.sqrt(vhat) + self.eps)

def finite_diff_grad(f, theta, eps=1e-4):
    g = np.zeros_like(theta, dtype=float)
    f0 = f(theta)
    for i in range(theta.size):
        t = theta.copy(); t[i] += eps
        g[i] = (f(t) - f0) / eps
    return g

def analytic_grad(Phi, xB, yB, w_center=1e-3, w_reg=5e-7):
    D = Phi.degree + 1
    theta = Phi.params()
    k = Phi.k

    preB_x  = [None]*k 
    preB_y  = [None]*k
    postB_x = [None]*k 
    postB_y = [None]*k
    postA_x = [None]*k 
    postA_y = [None]*k

    x, y = xB, yB
    for i in range(k-1, -1, -1):
        preB_x[i], preB_y[i] = x, y
        x, y = Phi.B[i](x, y)
        postB_x[i], postB_y[i] = x, y
        x, y = Phi.A[i](x, y)
        postA_x[i], postA_y[i] = x, y

    x_final, y_final = x, y

    r = np.hypot(x_final, y_final)
    R = r.max()
    mask = (r >= R - 0.0)

    gx = np.zeros_like(x_final)
    gy = np.zeros_like(y_final)
    gx[mask] = x_final[mask] / (r[mask] + 1e-12)
    gy[mask] = y_final[mask] / (r[mask] + 1e-12)

    gx += 2.0 * w_center * (x_final.mean()) / x_final.size
    gy += 2.0 * w_center * (y_final.mean()) / y_final.size
    grad = np.zeros_like(theta)
    for i in range(0, k):
        x_in_A = postB_x[i]
        a = Phi.A[i].coeffs  

        if D >= 3:
            d2_a = np.array([m*(m-1)*a[m] for m in range(2, D)], dtype=float)
            f2 = np.polyval(d2_a[::-1], x_in_A)
        else:
            f2 = 0.0
        base = 2*i*D
        for m in range(1, D):
            grad[base + m] += np.sum(gy * (x_in_A**(m-1)) * m)
        gx = gx + gy * f2
        y_in_B = preB_y[i]
        b = Phi.B[i].coeffs
        if D >= 3:
            d2_b = np.array([m*(m-1)*b[m] for m in range(2, D)], dtype=float)
            g2 = np.polyval(d2_b[::-1], y_in_B)
        else:
            g2 = 0.0
        for m in range(1, D):
            grad[base + D + m] += np.sum(gx * (y_in_B**(m-1)) * m)
        gy = gy + gx * g2
    grad += 2.0 * w_reg * theta

    return grad

# ---------------------------
# Training (boundary only)
# ---------------------------
def train_min_radius_boundary_2d(
    degree=5, k=3,
    n_boundary=3000, region=Shape(),
    n_iters=30000, lr=2e-2, seed=7,
    polynomial_bound=0.005,
    w_center=1e-3, w_reg=5e-7, report_every=25,
    animate=False
):
    rng = np.random.default_rng(seed)
    xB, yB = region.boundary_points(n_boundary, seed=seed)

    area = region.area()
    r_eq = np.sqrt(area / np.pi)

    D = degree + 1
    num_params = 2*k*D
    theta0 = rng.uniform(-polynomial_bound, polynomial_bound, num_params)
    Phi = SymplecticComposition(theta0, degree, k)
    opt = Adam(Phi.params(), lr=lr)

    def fobj(theta):
        Phi.set_params(theta)
        L, _, _ = loss_max_radius_boundary(Phi, xB, yB, w_center=w_center, w_reg=w_reg)
        return L

    history = []
    best = (np.inf, Phi.params())
    bestiter = 0

    # For animation
    frames = []

    for it in range(1, n_iters+1):
        xB, yB = region.boundary_points(n_boundary, seed=seed + it)
        grad = analytic_grad(Phi, xB, yB, w_center=w_center, w_reg=w_reg)

        # stop if gradient contains NaN/Inf
        if not np.all(np.isfinite(grad)):
            print(f"[{it:4d}] Non-finite gradient encountered; stopping.")
            break

        new_params = opt.step(Phi.params(), grad)

        # stop if optimizer produced NaN/Inf parameters
        if not np.all(np.isfinite(new_params)):
            print(f"[{it:4d}] Non-finite parameters produced by optimizer; stopping.")
            break

        Phi.set_params(new_params)

        L, aux, (xb, yb) = loss_max_radius_boundary(Phi, xB, yB, w_center=w_center, w_reg=w_reg)

        # check for non-finite loss 
        if not np.isfinite(L):
            print(f"[{it:4d}] Non-finite loss encountered (L={L}); stopping.")
            break

        history.append(dict(it=it, loss=float(L), R=aux["R"], rmean=aux["rmean"],
                            rvar=aux["rvar"], cx=aux["cx"], cy=aux["cy"]))

        if L < best[0]:
            best = (float(L), Phi.params())
            bestiter = it

        if it % report_every == 0 or it == 1 or it == n_iters:
            print(f"[{it:4d}] L={L:.6f}  R={aux['R']:.6f}  r_eq={r_eq:.6f}  "
                  f"rmean={aux['rmean']:.6f}  var={aux['rvar']:.3e}  "
                  f"cent=({aux['cx']:.2e},{aux['cy']:.2e})")
            if animate:
                frames.append((xb.copy(), yb.copy()))

    Phi.set_params(best[1])
    xb, yb = Phi.forward(xB, yB)
    return Phi, (xB, yB), (xb, yb), r_eq, history, bestiter, frames if animate else None
def _polyval2d_desc(x, y, c_desc):
    c_asc = c_desc[::-1, ::-1]
    return _np_polyval2d(x, y, c_asc)

np.polyval2d = _polyval2d_desc
# Shapes in R^4
class Shape4D:
    def volume(self): return 1.0
    def boundary_points(self, k=1, seed=0): raise NotImplementedError

class Ellipsoid4D(Shape4D):
    def __init__(self, position=(0,0,0,0), radii=(1,1,1,1)):
        self.position = np.asarray(position, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
    def boundary_points(self, k=1, seed=0):
        rng = np.random.default_rng(seed)
        u = rng.normal(0,1,(4,k))
        u /= np.linalg.norm(u, axis=0, keepdims=True)
        pts = (self.radii[:,None] * u) + self.position[:,None]
        return pts[0], pts[1], pts[2], pts[3]
    def volume(self):
        return (np.pi**2/2.0) * float(np.prod(self.radii))

class LagrangianTorus4D(Shape4D):
    def __init__(self, center=(0,0,0,0), radii_xy1=(1,1), radii_xy2=(1,1)):
        self.center = np.asarray(center, dtype=float)
        self.rxy1 = np.asarray(radii_xy1, dtype=float)
        self.rxy2 = np.asarray(radii_xy2, dtype=float)
    def boundary_points(self, k=2048, seed=0):
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(0,2*np.pi,k)
        t2 = rng.uniform(0,2*np.pi,k)
        x1 = self.center[0] + self.rxy1[0]*np.cos(t1)
        y1 = self.center[1] + self.rxy1[1]*np.sin(t1)
        x2 = self.center[2] + self.rxy2[0]*np.cos(t2)
        y2 = self.center[3] + self.rxy2[1]*np.sin(t2)
        return x1, x2, y1, y2

class Union4D(Shape4D):
    def __init__(self, shapes):
        self.shapes = list(shapes)
    def boundary_points(self, k=4000, seed=0):
        xs1=[]; xs2=[]; ys1=[]; ys2=[]
        per = max(1, k//len(self.shapes))
        for i,sh in enumerate(self.shapes):
            x1,x2,y1,y2 = sh.boundary_points(per, seed=seed+11*i)
            xs1.append(x1); xs2.append(x2); ys1.append(y1); ys2.append(y2)
        return np.concatenate(xs1), np.concatenate(xs2), np.concatenate(ys1), np.concatenate(ys2)
    def volume(self):
        return sum(getattr(s,'volume',lambda:0.0)() for s in self.shapes)

#build 2D polynomial derivative coefficient matrices
def poly2d_second_derivs(coeffs):
    c = np.asarray(coeffs, dtype=float)
    D = c.shape[0]
    assert c.shape[0]==c.shape[1]
    p_xx = np.zeros((max(D-2,1), D))
    for i in range(2, D):
        p_xx[i-2,:] = i*(i-1)*c[i,:]
    p_yy = np.zeros((D, max(D-2,1)))
    for j in range(2, D):
        p_yy[:,j-2] = j*(j-1)*c[:,j]
    p_xy = np.zeros((max(D-1,1), max(D-1,1)))
    if D>=2:
        for i in range(1,D):
            for j in range(1,D):
                p_xy[i-1,j-1] = i*j*c[i,j]
    def desc(M): return M[::-1, ::-1] if M.size>1 else M
    return desc(p_xx), desc(p_xy), desc(p_yy)

# Symplectic shears in R^4 (A/B blocks)
class ASymplecticR4:
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, dtype=float)
        assert c.ndim==2 and c.shape[0]==c.shape[1]
        self.coeffs = c
        D = c.shape[0]
        d1 = np.zeros((D-1, D)) if D>1 else np.zeros((1, D))
        d2 = np.zeros((D, D-1)) if D>1 else np.zeros((D, 1))
        for i in range(1,D): d1[i-1,:] = i*c[i,:] 
        for j in range(1,D): d2[:,j-1] = j*c[:,j]
        self.d1_desc = d1[::-1, ::-1] if d1.size>1 else d1
        self.d2_desc = d2[::-1, ::-1] if d2.size>1 else d2
        p_xx, p_xy, p_yy = poly2d_second_derivs(c)
        self.f_xx_desc, self.f_xy_desc, self.f_yy_desc = p_xx, p_xy, p_yy
    def __call__(self, x1,x2,y1,y2):
        y1 = y1 + np.polyval2d(x1, x2, self.d1_desc)
        y2 = y2 + np.polyval2d(x1, x2, self.d2_desc)
        return x1,x2,y1,y2

class BSymplecticR4:
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, dtype=float)
        assert c.ndim==2 and c.shape[0]==c.shape[1]
        self.coeffs = c
        D = c.shape[0]
        d1 = np.zeros((D-1, D)) if D>1 else np.zeros((1, D))
        d2 = np.zeros((D, D-1)) if D>1 else np.zeros((D, 1))
        for i in range(1,D): d1[i-1,:] = i*c[i,:] 
        for j in range(1,D): d2[:,j-1] = j*c[:,j] 
        self.d1_desc = d1[::-1, ::-1] if d1.size>1 else d1
        self.d2_desc = d2[::-1, ::-1] if d2.size>1 else d2
        p_xx, p_xy, p_yy = poly2d_second_derivs(c)   
        self.g_11_desc, self.g_12_desc, self.g_22_desc = p_xx, p_xy, p_yy
    def __call__(self, x1,x2,y1,y2):
        x1 = x1 + np.polyval2d(y1, y2, self.d1_desc)
        x2 = x2 + np.polyval2d(y1, y2, self.d2_desc)
        return x1,x2,y1,y2

class SymplecticCompositionR4:
    def __init__(self, params, degree, k):
        self.degree = degree; self.k = k
        self.set_params(params)
    def set_params(self, params):
        params = np.asarray(params, dtype=float)
        D = self.degree + 1
        need = 2*self.k*D*D
        assert params.size==need, f"need {need} params, got {params.size}"
        self._params = params.copy()
        self.A=[]; self.B=[]
        off=0
        for _ in range(self.k):
            a = params[off:off+D*D].reshape(D,D); off += D*D
            b = params[off:off+D*D].reshape(D,D); off += D*D
            self.A.append(ASymplecticR4(a))
            self.B.append(BSymplecticR4(b))
    def params(self): return self._params.copy()
    def forward(self, x1,x2,y1,y2):
        for i in range(self.k-1, -1, -1):
            x1,x2,y1,y2 = self.B[i](x1,x2,y1,y2)
            x1,x2,y1,y2 = self.A[i](x1,x2,y1,y2)
        return x1,x2,y1,y2
 
# Loss and Analytic Gradient
def loss_max_radius_boundary_R4(Phi, x1B,x2B,y1B,y2B, w_center=1e-3, w_reg=1e-7):
    X1,X2,Y1,Y2 = Phi.forward(x1B,x2B,y1B,y2B)
    r2 = X1*X1 + X2*X2 + Y1*Y1 + Y2*Y2
    r = np.sqrt(r2)
    R = float(r.max())
    cx = float(X1.mean()); cy = float(X2.mean()); cp = float(Y1.mean()); cq = float(Y2.mean())
    reg = float(np.sum(Phi.params()**2))
    L = R + w_center*(cx*cx + cy*cy + cp*cp + cq*cq) + w_reg*reg
    aux = dict(R=R, rmean=float(r.mean()), rvar=float(r.var()), cx=cx, cy=cy, cp=cp, cq=cq)
    return L, aux, (X1,X2,Y1,Y2,r)

def analytic_grad_R4(Phi, x1B,x2B,y1B,y2B, w_center=1e-3, w_reg=1e-7):
    k = Phi.k; D = Phi.degree+1
    preB = [None]*k
    postB = [None]*k
    postA = [None]*k
    x1,x2,y1,y2 = x1B, x2B, y1B, y2B
    for i in range(k-1, -1, -1):
        preB[i]  = (x1,x2,y1,y2)
        x1,x2,y1,y2 = Phi.B[i](x1,x2,y1,y2)
        postB[i] = (x1,x2,y1,y2)
        x1,x2,y1,y2 = Phi.A[i](x1,x2,y1,y2)
        postA[i] = (x1,x2,y1,y2)
    X1,X2,Y1,Y2 = x1,x2,y1,y2
    r2 = X1*X1 + X2*X2 + Y1*Y1 + Y2*Y2
    r = np.sqrt(r2)
    R = r.max()
    mask = (r >= R - 0.0)
    eps = 1e-12
    dL_dX1 = np.zeros_like(X1); dL_dX2 = np.zeros_like(X2)
    dL_dY1 = np.zeros_like(Y1); dL_dY2 = np.zeros_like(Y2)
    dL_dX1[mask] = X1[mask] / (r[mask] + eps)
    dL_dX2[mask] = X2[mask] / (r[mask] + eps)
    dL_dY1[mask] = Y1[mask] / (r[mask] + eps)
    dL_dY2[mask] = Y2[mask] / (r[mask] + eps)
    n = X1.size
    dL_dX1 += 2.0*w_center*(X1.mean())/n
    dL_dX2 += 2.0*w_center*(X2.mean())/n
    dL_dY1 += 2.0*w_center*(Y1.mean())/n
    dL_dY2 += 2.0*w_center*(Y2.mean())/n
    grad = np.zeros_like(Phi.params())
    off = 0
    D2 = D*D
    layer_offsets = [(2*i*D2, 2*i*D2 + D2, 2*i*D2 + 2*D2) for i in range(Phi.k)]
    gx1, gx2, gy1, gy2 = dL_dX1, dL_dX2, dL_dY1, dL_dY2
    for i in range(k-1, -1, -1):
        a_start, b_start, _ = layer_offsets[i]
        x1_in, x2_in, y1_in, y2_in = postB[i]
        a = Phi.A[i].coeffs
        P = np.arange(D)[:,None]
        Q = np.arange(D)[None,:]

        X1p = np.vstack([x1_in**p for p in range(D)])
        X2q = np.vstack([x2_in**q for q in range(D)])
        term1 = np.zeros((D,D))
        term2 = np.zeros((D,D))
        if D>=2:
            term1[1: , :] = (P[1: , :])*0 + 0
        for p in range(1,D):
            for q in range(D):
                term1[p,q] = np.sum(gy1 * (p * X1p[p-1] * X2q[q]))
        for p in range(D):
            for q in range(1,D):
                term2[p,q] = np.sum(gy2 * (q * X1p[p] * X2q[q-1]))
        grad[a_start:a_start+D2] += (term1 + term2).ravel()
        f_xx = np.polyval2d(x1_in, x2_in, Phi.A[i].f_xx_desc)
        f_xy = np.polyval2d(x1_in, x2_in, Phi.A[i].f_xy_desc)
        f_yy = np.polyval2d(x1_in, x2_in, Phi.A[i].f_yy_desc)
        gx1 = gx1 + gy1 * f_xx + gy2 * f_xy
        gx2 = gx2 + gy1 * f_xy + gy2 * f_yy
        x1_pre, x2_pre, y1_pre, y2_pre = preB[i]
        b = Phi.B[i].coeffs
        Y1p = np.vstack([y1_pre**p for p in range(D)])
        Y2q = np.vstack([y2_pre**q for q in range(D)])
        term1 = np.zeros((D,D))
        term2 = np.zeros((D,D))
        for p in range(1,D):
            for q in range(D):
                term1[p,q] = np.sum(gx1 * (p * Y1p[p-1] * Y2q[q]))
        for p in range(D):
            for q in range(1,D):
                term2[p,q] = np.sum(gx2 * (q * Y1p[p] * Y2q[q-1]))
        grad[b_start:b_start+D2] += (term1 + term2).ravel()
        g_11 = np.polyval2d(y1_pre, y2_pre, Phi.B[i].g_11_desc)
        g_12 = np.polyval2d(y1_pre, y2_pre, Phi.B[i].g_12_desc)
        g_22 = np.polyval2d(y1_pre, y2_pre, Phi.B[i].g_22_desc)
        gy1 = gy1 + gx1 * g_11 + gx2 * g_12
        gy2 = gy2 + gx1 * g_12 + gx2 * g_22
    grad += 2.0*w_reg * Phi.params()
    return grad

# Optimizer
class Adam:
    def __init__(self, params, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        update = self.lr * mhat / (np.sqrt(vhat) + self.eps)
        new_params = params - update
        return new_params


# Training loop
def train_min_radius_boundary_R4(
    degree=3, k=6,
    n_boundary=6000,
    region=Ellipsoid4D(radii=(0.25,0.35,0.75,0.55)),
    n_iters=10000, lr=2e-3, seed=11,
    polynomial_bound=5e-3,
    w_center=1e-3, w_reg=1e-7, report_every=25,
    animate=True, max_frames=60
):
    rng = np.random.default_rng(seed)
    D = degree + 1
    num_params = 2*k*D*D
    theta0 = rng.uniform(-polynomial_bound, polynomial_bound, num_params)
    Phi = SymplecticCompositionR4(theta0, degree, k)
    opt = Adam(Phi.params(), lr=lr)

    history = []
    best = (np.inf, Phi.params())
    bestiter = 0
    frames = []

    for it in range(1, n_iters+1):
        x1B,x2B,y1B,y2B = region.boundary_points(n_boundary, seed=seed+it)
        grad = analytic_grad_R4(Phi, x1B,x2B,y1B,y2B, w_center=w_center, w_reg=w_reg)
        if not np.all(np.isfinite(grad)):
            print(f"[{it:4d}] Non-finite gradient; stopping."); break
        new_params = opt.step(Phi.params(), grad)
        if not np.all(np.isfinite(new_params)):
            print(f"[{it:4d}] Non-finite parameters; stopping."); break
        Phi.set_params(new_params)
        L, aux, (X1,X2,Y1,Y2,r) = loss_max_radius_boundary_R4(Phi, x1B,x2B,y1B,y2B, w_center=w_center, w_reg=w_reg)
        if not np.isfinite(L):
            print(f"[{it:4d}] Non-finite loss {L}; stopping."); break
        history.append(dict(it=it, loss=float(L), **aux))
        if L < best[0]:
            best = (float(L), Phi.params().copy())
            bestiter = it
        if animate and (it % report_every == 0 or it==1 or it==n_iters) and len(frames) < max_frames:
            take = min(4000, X1.size)
            idx = np.random.default_rng(1234).choice(X1.size, size=take, replace=False)
            frames.append((X1[idx], Y1[idx], X2[idx], Y2[idx], aux['R']))
        if it % report_every == 0 or it==1 or it==n_iters:
            print(f"[{it:4d}] L={L:.6f}  R={aux['R']:.6f}  rmean={aux['rmean']:.6f}  var={aux['rvar']:.3e}  "
                  f"cent=({aux['cx']:.2e},{aux['cy']:.2e},{aux['cp']:.2e},{aux['cq']:.2e})")

    Phi.set_params(best[1])
    x1B,x2B,y1B,y2B = region.boundary_points(n_boundary, seed=seed+9999)
    X1,X2,Y1,Y2 = Phi.forward(x1B,x2B,y1B,y2B)
    return Phi, (x1B,x2B,y1B,y2B), (X1,X2,Y1,Y2), history, bestiter, frames

# Animation helper
def save_projection_animation(frames, outpath="r4_training.gif"):
    if not frames:
        print("No frames to animate."); return
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    scat1 = axes[0].scatter([], [], s=1)
    scat2 = axes[1].scatter([], [], s=1)
    for ax,title in zip(axes, ["Projection (x1,y1)", "Projection (x2,y2)"]):
        ax.set_aspect("equal")
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    def update(frame):
        X1,Y1,X2,Y2,R = frame
        scat1.set_offsets(np.column_stack([X1, Y1]))
        scat2.set_offsets(np.column_stack([X2, Y2]))
        axes[0].set_title(f"(x1,y1)  R≈{R:.3f}")
        axes[1].set_title(f"(x2,y2)  R≈{R:.3f}")
        return scat1, scat2
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="pillow")
    plt.close(fig)
    print(f"Saved animation to {outpath}")

# CLI
def build_region(args):
    if args.region == "ellipsoid":
        r = tuple(map(float, args.radii.split(",")))
        assert len(r)==4, "--radii must have 4 comma-separated floats"
        return Ellipsoid4D(radii=r)
    elif args.region == "torus":
        r1 = tuple(map(float, args.rxy1.split(",")))
        r2 = tuple(map(float, args.rxy2.split(",")))
        return LagrangianTorus4D(radii_xy1=r1, radii_xy2=r2)
    elif args.region == "union_tori":
        r1 = tuple(map(float, args.rxy1.split(",")))
        r2 = tuple(map(float, args.rxy2.split(",")))
        t1 = LagrangianTorus4D(center=(-0.4,0,0,0), radii_xy1=r1, radii_xy2=r2)
        t2 = LagrangianTorus4D(center=( 0.4,0,0,0), radii_xy1=r1, radii_xy2=r2)
        return Union4D([t1,t2])
    else:
        raise ValueError(f"Unknown region {args.region}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-iters", type=int, default=3000)
    ap.add_argument("--n-boundary", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--verthalfwidth", type=float, default=0.5)
    ap.add_argument("--horihalfwidth", type=float, default=0.25)
    ap.add_argument("--outdir", type=str, default="output_rect2d")
    ap.add_argument("--anim", action="store_true", default=True)
    args = ap.parse_args()

    region = Rectangle(position=(0, 0), verthalfwidth=args.verthalfwidth, horihalfwidth=args.horihalfwidth)

    Phi, startB, endB, r_eq, history, bestiter, frames = train_min_radius_boundary_2d(
        degree=args.degree,
        k=args.k,
        n_boundary=args.n_boundary,
        region=region,
        n_iters=args.n_iters,
        lr=args.lr,
        seed=args.seed,
        polynomial_bound=0.005,
        w_center=2e-3,
        w_reg=5e-7,
        report_every=max(5, args.n_iters // 20),
        animate=args.anim
    )

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["it", "loss", "R", "rmean", "rvar", "cx", "cy"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    (xB, yB) = startB
    (xb, yb) = endB
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].scatter(xB, yB, s=1, label="input boundary")
    axes[1].scatter(xb, yb, s=1, label="mapped boundary")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_title(f"Input boundary (Rectangle {args.horihalfwidth*2:.2f}×{args.verthalfwidth*2:.2f})")
    axes[1].set_title("Mapped boundary (after training)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "snapshot.png"), dpi=150)
    plt.close(fig)

    print(f"\nTraining finished. Best iteration = {bestiter}")
    print(f"Equal-area circle radius = {r_eq:.4f}")
    print(f"Results saved to {args.outdir}/")


if __name__ == "__main__":
    main()