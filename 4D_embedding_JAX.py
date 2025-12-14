import numpy as np
import math
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt
import jax.example_libraries.optimizers as joptimizers
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, csv, argparse, time

def loss_max_radius_boundary_R4_jax(params, x1B, x2B, y1B, y2B, degree, k, w_center=1e-3, w_reg=1e-7, tau=None):
    # forward pass through HenonComp
    X1, X2, Y1, Y2 = henon_comp_forward_jax(params, degree, k, x1B, x2B, y1B, y2B)
    r = jnp.sqrt(X1**2 + X2**2 + Y1**2 + Y2**2)
    if tau is None or tau == 0:
        Lmax = jnp.max(r)
    else:
        t = tau * r
        tmax = jnp.max(t)
        s = jnp.sum(jnp.exp(t - tmax))
        Lmax = (tmax + jnp.log(s)) / tau
    cx = jnp.mean(X1); cy = jnp.mean(X2); cp = jnp.mean(Y1); cq = jnp.mean(Y2)
    reg = jnp.sum(params**2)
    return Lmax + w_center * (cx**2 + cy**2 + cp**2 + cq**2) + w_reg * reg

def polyval2d(y1, y2, coeffs):
    # like np.polyval2d but vectorized and JAX-friendly
    D = coeffs.shape[0]
    val = jnp.zeros_like(y1)
    for i in range(D):
        for j in range(D):
            val += coeffs[i, j] * (y1 ** (D - 1 - i)) * (y2 ** (D - 1 - j))
    return val

def henon_map_apply_jax(x1, x2, y1, y2, coeffs, const):
    D = coeffs.shape[0]
    dV_dy1 = jnp.zeros_like(y1)
    dV_dy2 = jnp.zeros_like(y2)

    for i in range(D):
        for j in range(D):
            c = coeffs[i, j]
            p1 = D - 1 - i
            p2 = D - 1 - j
            if p1 > 0:
                dV_dy1 = dV_dy1 + c * p1 * (y1 ** (p1 - 1)) * (y2 ** p2)
            if p2 > 0:
                dV_dy2 = dV_dy2 + c * p2 * (y1 ** p1) * (y2 ** (p2 - 1))

    return (
        y1 + const[0],
        y2 + const[1],
        -x1 + dV_dy1,
        -x2 + dV_dy2,
    )

def henon_comp_forward_jax(params, degree, k, x1, x2, y1, y2):
    D = degree + 1
    off = 0
    for i in range(k - 1, -1, -1):
        coeffs = params[off : off + D*D].reshape(D, D)
        off += D*D
        const = params[off : off + 2]
        off += 2
        x1, x2, y1, y2 = henon_map_apply_jax(x1, x2, y1, y2, coeffs, const)
    return x1, x2, y1, y2

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
    def __init__(self, a=1.0, b=2.0):
        self.a = float(a)
        self.b = float(b)
    def boundary_points(self, k=2048, seed=0):
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(0, 2*np.pi, k)
        t2 = rng.uniform(0, 2*np.pi, k)
        R1 = np.sqrt(self.a / np.pi)
        R2 = np.sqrt(self.b / np.pi)
        x1 = R1 * np.cos(t1)
        y1 = R1 * np.sin(t1)
        x2 = R2 * np.cos(t2)
        y2 = R2 * np.sin(t2)
        return x1, x2, y1, y2

class PolyDisk4D(Shape4D):
    def __init__(self, a=1.0, b=2.5):
        self.a = float(a)
        self.b = float(b)

    def boundary_points(self, k=2048, seed=0):
        rng = np.random.default_rng(seed)

        length = np.sqrt(rng.uniform(0, 1, (k, 2)))
        angle  = np.pi * rng.uniform(0, 2, (k, 2))

        half = k // 2
        length[:half, 0] = 1.0
        length[half:, 1] = 1.0

        q = length * np.cos(angle)
        p = length * np.sin(angle)

        R1 = np.sqrt(self.a / np.pi)
        R2 = np.sqrt(self.b / np.pi)

        z1 = R1 * np.column_stack([q[:, 0], p[:, 0]])
        z2 = R2 * np.column_stack([q[:, 1], p[:, 1]]) 

        z = np.column_stack([z1[:, 0], z1[:, 1], z2[:, 0], z2[:, 1]]).astype(np.float32)

        x1 = z[:, 0]
        x2 = z[:, 2]
        y1 = z[:, 1]
        y2 = z[:, 3]

        return x1, x2, y1, y2

    
def train_min_radius_boundary_R4(
    degree=3, k=6,
    n_boundary=60000,
    region=Ellipsoid4D(radii=(0.25,0.35,0.75,0.55)),
    n_iters=200, lr=2e-3, seed=11,
    polynomial_bound=5e-3,
    w_center=1e-3, w_reg=1e-7, report_every=25,
    optimizer='adam',
    minibatch_size=None,
    sgd_momentum=0.0,
    animate=True, max_frames=100
):
    timestart = time.time()
    D = degree + 1
    num_params = k * D * D + 2 * k
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=polynomial_bound, size=num_params)

    history = []
    frames = []

    plateau_window = 200
    plateau_eps = 1e-4
    last_kick_it = 0
    last_loss = None

    x1B_all, x2B_all, y1B_all, y2B_all = region.boundary_points(k=n_boundary, seed=seed+1)

    if optimizer == 'adam':
        opt_init, opt_update, get_params = joptimizers.adam(lr)
    elif optimizer == 'sgd':
        opt_init, opt_update, get_params = joptimizers.momentum(lr, mass=sgd_momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    opt_state = opt_init(theta)

    def clip_grads(grads, max_norm=10.0):
        leaves = jtu.tree_leaves(grads)
        g2 = sum([jnp.sum(g**2) for g in leaves])
        g_norm = jnp.sqrt(g2)
        factor = jnp.minimum(1.0, max_norm / (g_norm + 1e-8))
        return jtu.tree_map(lambda g: g * factor, grads)

    @jax.jit
    def compute_R(params, x1B, x2B, y1B, y2B):
        X1, X2, Y1, Y2 = henon_comp_forward_jax(
            params, degree, k, x1B, x2B, y1B, y2B
        )
        r = X1**2 + X2**2 + Y1**2 + Y2**2
        return jnp.max(r)

    @jax.jit
    def step(opt_state, x1B, x2B, y1B, y2B):
        params = get_params(opt_state)
        loss, grads = jax.value_and_grad(loss_max_radius_boundary_R4_jax)(
            params, x1B, x2B, y1B, y2B,
            degree, k,
            w_center=w_center,
            w_reg=w_reg,
            tau = 10.0
        )
        grads = clip_grads(grads, max_norm=10.0)
        opt_state = opt_update(it, grads, opt_state)
        return opt_state, loss

    for it in range(n_iters):
        if minibatch_size is not None:
            indices = rng.choice(n_boundary, size=minibatch_size, replace=False)
            x1B_batch = x1B_all[indices]
            x2B_batch = x2B_all[indices]
            y1B_batch = y1B_all[indices]
            y2B_batch = y2B_all[indices]
        else:
            x1B_batch = x1B_all
            x2B_batch = x2B_all
            y1B_batch = y1B_all
            y2B_batch = y2B_all
        
        opt_state, loss = step(opt_state, x1B_batch, x2B_batch, y1B_batch, y2B_batch)
        history.append(dict(it=it, loss=loss))
        if last_loss is not None:

            if it > 25000:

                is_plateau = abs(float(loss) - last_loss) < plateau_eps
                long_since_last = (it - last_kick_it) > plateau_window

                if is_plateau and long_since_last:
                    print(f">>> Plateau detected at iter {it}, applying KICK")

                    params_now = np.array(get_params(opt_state))
                    noise_scale = 1e-4
                    params_perturbed = params_now + noise_scale * rng.normal(size=params_now.shape)

                    opt_state = opt_init(params_perturbed)

                    last_kick_it = it

        last_loss = float(loss)

        if animate and (it % (n_iters // max_frames) == 0 or it == n_iters - 1):
            params = get_params(opt_state)
            X1B, X2B, Y1B, Y2B = henon_comp_forward_jax(params, degree, k, x1B_all, x2B_all, y1B_all, y2B_all)
            R = np.max(np.sqrt(X1B**2 + Y1B**2 + X2B**2 + Y2B**2))
            frames.append((X1B, Y1B, X2B, Y2B, R, it))
            
        if (it + 1) % report_every == 0 or it == 0:
            params = get_params(opt_state)
            R = compute_R(params, x1B_all, x2B_all, y1B_all, y2B_all)
            print(f"Iter {it+1}, Loss: {float(loss):.6f}, R: {float(R):.6f}")

    timeend = time.time()
    print(f"Training completed in {timeend - timestart:.2f} seconds.")
    final_params = get_params(opt_state)
    return final_params, history, frames

def save_projection_animation(frames, outpath="r4_training.gif"):
    if not frames:
        print("No frames to animate."); return
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    scat1 = axes[0].scatter([], [], s=1)
    scat2 = axes[1].scatter([], [], s=1)
    for ax,title in zip(axes, ["Projection (x1,y1)", "Projection (x2,y2)"]):
        ax.set_aspect("equal")
        ax.set_xlim(-4,4); ax.set_ylim(-4,4)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    def update(frame):
        X1,Y1,X2,Y2,R, it = frame
        scat1.set_offsets(np.column_stack([X1, Y1]))
        scat2.set_offsets(np.column_stack([X2, Y2]))
        axes[0].set_title(f"(x1,y1)  R≈{R:.3f}  iter={it}")
        axes[1].set_title(f"(x2,y2)  R≈{R:.3f}  iter={it}")
        return scat1, scat2
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="ffmpeg")
    plt.close(fig)
    print(f"Saved animation to {outpath}")

def save_radial_projection_animation(frames, outpath="r4_radial_training.gif"):
    """Save an animation projecting 4D points to the plane
    (x1^2 + y1^2, x2^2 + y2^2).
    Expects frames as a list of tuples: (X1, Y1, X2, Y2, R)
    where X1 etc are 1D numpy arrays of the same length for that frame.
    """
    if not frames:
        print("No frames to animate."); return
    # compute reasonable axis limits from pooled data (small number of frames)
    all_r1_min = np.inf; all_r1_max = -np.inf
    all_r2_min = np.inf; all_r2_max = -np.inf
    for X1, Y1, X2, Y2, _, _ in frames:
        r1 = 3 * np.pi * (X1**2 + Y1**2)
        r2 = np.pi * (X2**2 + Y2**2)
        if r1.size:
            all_r1_min = min(all_r1_min, float(r1.min())); all_r1_max = max(all_r1_max, float(r1.max()))
        if r2.size:
            all_r2_min = min(all_r2_min, float(r2.min())); all_r2_max = max(all_r2_max, float(r2.max()))

    pad1 = 0.05 * (all_r1_max - all_r1_min) if all_r1_max>all_r1_min else 0.1
    pad2 = 0.05 * (all_r2_max - all_r2_min) if all_r2_max>all_r2_min else 0.1

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    scat = ax.scatter([], [], s=1)
    ax.set_aspect('equal')
    if all_r1_max > 100 or all_r2_max > 100:
        ax.set_xlim(-1, 100)
        ax.set_ylim(-1, 100)
    else:
        ax.set_xlim(all_r1_min - pad1, all_r1_max + pad1)
        ax.set_ylim(all_r2_min - pad2, all_r2_max + pad2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Radial projection (x1^2+y1^2 vs x2^2+y2^2)")

    def update(frame):
        X1, Y1, X2, Y2, R, it = frame
        r1 = 3 * np.pi * (X1**2 + Y1**2)
        r2 = np.pi * (X2**2 + Y2**2)
        coords = np.column_stack([r1, r2])
        scat.set_offsets(coords)
        ax.set_title(f"Radial projection  R≈{R:.3f}  iter={it}")
        return (scat,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="ffmpeg")
    plt.close(fig)
    print(f"Saved radial projection animation to {outpath}")

def main():
    d = 2
    k = 45
    # radii = (1,np.sqrt(6),1, np.sqrt(6))
    # region = Ellipsoid4D(radii=radii)
    # --------------------------
    #  Lagrangian Torus
    # --------------------------
    region = LagrangianTorus4D(a=1.0, b=6.0)
    # region = PolyDisk4D(a=1.0, b=6.0)
    final_params, history, frames = train_min_radius_boundary_R4(
        degree=d, k=k,
        n_boundary=60000,
        region=region,
        n_iters=1000000, lr=1e-4, seed=11,
        polynomial_bound=5e-3,
        w_center=1e-3, w_reg=1e-7, report_every=100,
        optimizer='adam',
        minibatch_size=None,
        sgd_momentum=0.9,
        animate=True, max_frames=60
    )
    
    with open(os.path.join("output_r4","history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["it","loss"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    if frames:
        save_projection_animation(frames, outpath=os.path.join("output_r4","training_animation.mp4"))
        save_radial_projection_animation(frames, outpath=os.path.join("output_r4","training_radial_animation.mp4"))

    x1B, x2B, y1B, y2B = region.boundary_points(k=6000, seed=12)
    X1B, X2B, Y1B, Y2B = henon_comp_forward_jax(final_params, d, k, x1B, x2B, y1B, y2B)
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    axes[0].scatter(X1B, Y1B, s=1)
    axes[1].scatter(X2B, Y2B, s=1)
    for ax,title in zip(axes, ["Projection (x1,y1)", "Projection (x2,y2)"]):
        ax.set_aspect("equal")
        ax.set_xlim(-2,2); ax.set_ylim(-2,2)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    fig.savefig(os.path.join("output_r4","snapshot.png"), dpi=150)

if __name__ == '__main__':
    main()