import numpy as np
import time
import os
import sys
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SymplecticNet(nn.Module):
    """
    PyTorch version of the TF graph:
    - quintic polynomial weights for A,B,C,D,H
    - 4D phase space z = (q1,q2,p1,p2)
    - num_macro_steps time slices
    """
    def __init__(self, num_macro_steps, n, tot_secs):
        super().__init__()
        self.num_macro_steps = num_macro_steps
        self.n = n
        self.tot_secs = tot_secs

        dim = n + n**2 + n**3 + n**4 + n**5

        self.A_weights_list_2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, dtype=torch.float32)) for _ in range(num_macro_steps)]
        )
        self.B_weights_list_2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, dtype=torch.float32)) for _ in range(num_macro_steps)]
        )
        self.C_weights_list_2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, dtype=torch.float32)) for _ in range(num_macro_steps)]
        )
        self.D_weights_list_2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, dtype=torch.float32)) for _ in range(num_macro_steps)]
        )
        self.H_weights_list_2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, dtype=torch.float32)) for _ in range(num_macro_steps)]
        ) 

    def _compute_poly(self, x, w):
        """
        x: [b, n], w: [dim]
        returns: [b]  = sum_{monomials up to degree 5} w_i * m_i(x)
        """
        b, n = x.shape

        quad = (x.view(b, n, 1) * x.view(b, 1, n)).reshape(b, n ** 2)
        cub = (quad.view(b, n ** 2, 1) * x.view(b, 1, n)).reshape(b, n ** 3)
        quart = (cub.view(b, n ** 3, 1) * x.view(b, 1, n)).reshape(b, n ** 4)
        quint = (quart.view(b, n ** 4, 1) * x.view(b, 1, n)).reshape(b, n ** 5)

        tot = torch.cat([x, quad, cub, quart, quint], dim=1)  # [b, dim]
        return (tot * w.view(1, -1)).sum(dim=1)  # [b]

    def compute_A(self, q_input, A_wt_input):
        return self._compute_poly(q_input, A_wt_input)

    def compute_B(self, p_input, B_wt_input):
        return self._compute_poly(p_input, B_wt_input)

    def compute_C(self, p_input, C_wt_input):
        return self._compute_poly(p_input, C_wt_input)

    def compute_D(self, p_input, D_wt_input):
        return self._compute_poly(p_input, D_wt_input)

    def compute_H(self, p_input, H_wt_input):
        return self._compute_poly(p_input, H_wt_input)

    def flow(self, z_init, num_steps, record_traj=False):
        """
        One pass of the time-dependent Hamiltonian, exactly mirroring the TF loop:

        - dt = tot_secs / (num_macro_steps * num_steps)
        - For m in 0..num_macro_steps-1:
            Map A (half step) -> B -> A (half) -> C (half) -> D -> C (half)
        - If record_traj=True, return trajectory tensor [T, b, 2n].
        """
        z = z_init.to(device)
        b, two_n = z.shape
        assert two_n == 4  # since n=2

        dt = self.tot_secs / (self.num_macro_steps * float(num_steps))

        # Need gradients through p,q,u,v and through the poly weights.
        z = z.clone().detach().requires_grad_(True)

        traj = []
        if record_traj:
            traj.append(z.unsqueeze(0))

        for m in range(self.num_macro_steps):
            # -------- Map A --------
            q = z[:, 0:2]
            p = z[:, 2:4]

            A_val = self.compute_A(p, self.A_weights_list_2[m])
            dA_p = torch.autograd.grad(A_val.sum(), p, create_graph=True)[0]

            q = q + dA_p * 0.5 * dt
            z = torch.cat([q, p], dim=1)
            if record_traj:
                traj.append(z.unsqueeze(0))

            # -------- Map B --------
            q = z[:, 0:2]
            p = z[:, 2:4]

            B_val = self.compute_B(q, self.B_weights_list_2[m])
            dB_q = torch.autograd.grad(B_val.sum(), q, create_graph=True)[0]

            p = p - dB_q * dt
            z = torch.cat([q, p], dim=1)
            if record_traj:
                traj.append(z.unsqueeze(0))

            # -------- Map A (second half) --------
            q = z[:, 0:2]
            p = z[:, 2:4]

            q = q + dA_p * 0.5 * dt 
            z = torch.cat([q, p], dim=1)
            if record_traj:
                traj.append(z.unsqueeze(0))

            # -------- Map C (first half) --------
            u = z[:, 1:3]        # indices 1,2
            v = z[:, [0, 3]]     # gather indices 0 and 3

            C_val = self.compute_C(v, self.C_weights_list_2[m])
            dC_v = torch.autograd.grad(C_val.sum(), v, create_graph=True)[0]

            first_component_negative = -dC_v[:, 0:1]
            second_component = dC_v[:, 1:2]
            dC_v_modified = torch.cat([second_component, first_component_negative], dim=1)

            u = u + dC_v_modified * 0.5 * dt
            v_0 = v[:, 0:1]
            v_3 = v[:, 1:2]
            z = torch.cat([v_0, u, v_3], dim=1)
            if record_traj:
                traj.append(z.unsqueeze(0))

            # -------- Map D --------
            u = z[:, 1:3]
            v = z[:, [0, 3]]

            D_val = self.compute_D(u, self.D_weights_list_2[m])
            dD_u = torch.autograd.grad(D_val.sum(), u, create_graph=True)[0]

            first_component_negative = -dD_u[:, 0:1]
            second_component = dD_u[:, 1:2]
            dD_u_modified = torch.cat([second_component, first_component_negative], dim=1)

            v = v + dD_u_modified * dt
            v_0 = v[:, 0:1]
            v_3 = v[:, 1:2]
            z = torch.cat([v_0, u, v_3], dim=1)
            if record_traj:
                traj.append(z.unsqueeze(0))

            # -------- Map C (second half) --------
            u = z[:, 1:3]
            v = z[:, [0, 3]]

            u = u + dC_v_modified * 0.5 * dt  # reuse same dC_v_modified
            v_0 = v[:, 0:1]
            v_3 = v[:, 1:2]
            z = torch.cat([v_0, u, v_3], dim=1)
            if record_traj:
                traj.append(z.unsqueeze(0))

        if record_traj:
            traj = torch.cat(traj, dim=0)  # [T, b, 4]
        else:
            traj = None

        return z, traj


def main():
    inp_args = {}
    for arg in sys.argv[1:]:
        k, v = arg.split("=")
        inp_args[k] = v

    tic = time.time()

    restore_session = True  # Whether or not to recall the weights from the last training stage or re-initialize.
    local_training_steps = 1000  # N = the number of training steps that will be performed when this script is run.
    current_lr = 0.001  # 0.005 #The learning rate to be used.
    decay_steps = 2000  # Number of training steps before we decay the learning rate.
    decay_rate = 0.9
    # The decay factor of the learning rate.
    num_steps_basic = 1  # The number of numerical integration steps for each time discretization of the Hamiltonian.
    num_macro_steps = 60  # k = the number of time discretization steps.
    b_basic = 1000  # batch size for training

    num_steps_check_accuracy = num_steps_basic  # 10 originally; same as TF line
    b_check_accuracy = 25000

    b_movie = b_basic
    num_steps_movie = 100  # for movie

    save_steps = 500
    check_accuracy_steps = 500
    write_steps = 5  # how often to write summary to tensorboard

    b_training_plot = b_basic
    update_training_plot_save_steps = 5

    n = 2  # Half the total real dimension (so 2n = 4).
    tot_secs = 1.0

    # Parameters for polydisk or ellipsoid (i.e. E(a,b) or P(a,b)).
    aa = 1.0
    bb = 6.0

    domain = 'torus'  # ellipsoid, polydisk, squarexdisk, squaretorus or torus.

    assert (update_training_plot_save_steps * local_training_steps * b_training_plot * 2 * n * 4) / (2.0 ** 30) <= 16

    # ------------- Model -------------
    model = SymplecticNet(num_macro_steps=num_macro_steps, n=n, tot_secs=tot_secs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    global_step = 0

    # ------------- Domain samplers -------------
    if domain == 'ellipsoid':
        def F(z):
            z1sq = z[:, 0] ** 2 + z[:, 2] ** 2
            z2sq = z[:, 1] ** 2 + z[:, 3] ** 2
            return math.pi * (z1sq / aa + z2sq / bb)

        def get_z_init(b=b_basic):
            z_init = np.random.uniform(-5, 5, [b, 4]).astype(np.float32)
            div_fact = np.sqrt(F(z_init).reshape([b, 1]))
            z_init = z_init / div_fact
            return z_init

    elif domain == 'polydisk':
        def get_z_init(b=b_basic):
            length = np.sqrt(np.random.uniform(0, 1, [b, 2]))
            angle = math.pi * np.random.uniform(0, 2, [b, 2])

            for i in range(b // 2):
                length[i, 0] = 1
            for j in range(b // 2, b):
                length[j, 1] = 1

            q = length * np.cos(angle)
            p = length * np.sin(angle)

            z1 = np.sqrt(aa) * np.concatenate([q[:, 0:1], p[:, 0:1]], axis=1) / np.sqrt(math.pi)
            z2 = np.sqrt(bb) * np.concatenate([q[:, 1:], p[:, 1:]], axis=1) / np.sqrt(math.pi)

            out = np.concatenate([z1, z2], axis=-1)[:, [0, 2, 1, 3]]

            return out

    elif domain == 'squarexdisk':

        def get_z_init(b=b_basic):
            """
            Generates b random points on the boundary of the domain R x C,
            where Area(R) = aa and Area(C) = bb.

            R = square [-L, L] x [-L, L]
            C = disk of radius r
            Output: [q1, q2, p1, p2]
            """
            L = np.sqrt(aa) / 2.0
            r = np.sqrt(bb / math.pi)

            points_q_p = np.zeros((b, 4), dtype=np.float32)

            surface_choice = np.random.randint(0, 2, b)

            # --- Case 1: ∂R × C ---
            mask_R = (surface_choice == 0)
            num_R = np.sum(mask_R)
            if num_R > 0:
                coord_fixed = np.random.randint(0, 2, num_R)
                sign_fixed = np.random.choice([-1, 1], num_R)

                q1_R = np.random.uniform(-L, L, num_R)
                q2_R = np.random.uniform(-L, L, num_R)

                q1_R[coord_fixed == 0] = sign_fixed[coord_fixed == 0] * L
                q2_R[coord_fixed == 1] = sign_fixed[coord_fixed == 1] * L

                length_C = r * np.sqrt(np.random.uniform(0, 1, num_R))
                angle_C = np.random.uniform(0, 2 * math.pi, num_R)
                p1_C = length_C * np.cos(angle_C)
                p2_C = length_C * np.sin(angle_C)

                points_q_p[mask_R] = np.column_stack([q1_R, q2_R, p1_C, p2_C])

            # --- Case 2: R × ∂C ---
            mask_C = (surface_choice == 1)
            num_C = np.sum(mask_C)
            if num_C > 0:
                q1_R = np.random.uniform(-L, L, num_C)
                q2_R = np.random.uniform(-L, L, num_C)

                length_C = r * np.ones(num_C)
                angle_C = np.random.uniform(0, 2 * math.pi, num_C)
                p1_C = length_C * np.cos(angle_C)
                p2_C = length_C * np.sin(angle_C)

                points_q_p[mask_C] = np.column_stack([q1_R, q2_R, p1_C, p2_C])

            return points_q_p.astype(np.float32)

    elif domain == 'squaretorus':

        def get_z_init(b=b_basic):
            """
            Generates b random points on the boundary of the domain (Perimeter(R) x Circumference(C)),
            where Area(R) = aa (for square side L) and Area(C) = bb (for disk radius r).

            Output: [q1, q2, p1, p2]
            """
            L = np.sqrt(aa) / 2.0
            r = np.sqrt(bb / math.pi)

            side_choice = np.random.randint(0, 4, b)

            q1 = np.random.uniform(-L, L, b)
            q2 = np.random.uniform(-L, L, b)

            q1[side_choice == 0] = L
            q1[side_choice == 1] = -L
            q2[side_choice == 2] = L
            q2[side_choice == 3] = -L

            length_C = r * np.ones(b)
            angle_C = np.random.uniform(0, 2 * math.pi, b)
            p1 = length_C * np.cos(angle_C)
            p2 = length_C * np.sin(angle_C)

            out = np.column_stack([q1, q2, p1, p2]).astype(np.float32)
            return out

    elif domain == 'torus':

        def F_A(z):
            z1sq = z[:, 0] ** 2 + z[:, 2] ** 2
            return z1sq

        def F_B(z):
            z2sq = z[:, 1] ** 2 + z[:, 3] ** 2
            return z2sq

        def get_z_init(b=b_basic):
            angle = math.pi * np.random.uniform(0, 2, [b, 2])

            R_A = np.full([b, 1], np.sqrt(aa / math.pi), dtype=np.float32)
            R_B = np.full([b, 1], np.sqrt(bb / math.pi), dtype=np.float32)
            length = np.concatenate([R_A, R_B], axis=1)

            q = length * np.cos(angle)
            p = length * np.sin(angle)

            z1 = np.concatenate([q[:, 0:1], p[:, 0:1]], axis=1)
            z2 = np.concatenate([q[:, 1:], p[:, 1:]], axis=1)

            out = np.concatenate([z1, z2], axis=-1)[:, [0, 2, 1, 3]]
            return out.astype(np.float32)

    else:
        raise Exception('Error! Domain not recognized.')

    if restore_session:
        if not os.path.isdir('summaries'):
            raise RuntimeError("No summaries directory found for restore_session=True")
        folders = os.listdir(os.curdir + '/summaries')
        for f in folders:
            assert f[:3] == 'run'
        run_num = np.max([int(f[3:]) for f in folders])
    else:
        if not os.path.isdir('summaries'):
            os.makedirs('summaries')
            run_num = 0
        else:
            folders = os.listdir(os.curdir + '/summaries')
            if len(folders) == 0:
                run_num = 0
            else:
                for f in folders:
                    assert f[:3] == 'run'
                run_num = np.max([int(f[3:]) for f in folders]) + 1

    writer = SummaryWriter(log_dir='summaries/run{}'.format(run_num))
    print('run_num:', run_num)
    print('summaries/run{}'.format(run_num))

    # ------------- Optional weight restore -------------
    if restore_session and os.path.exists('save_variables.pt'):
        print('Restoring session...')
        state = torch.load('save_variables.pt', map_location=device)
        model.load_state_dict(state)
    else:
        print('Initializing variables...')

    # ------------- one-time summaries -------------
    writer.add_scalar('num_steps_basic', num_steps_basic, 0)
    writer.add_scalar('num_macro_steps', num_macro_steps, 0)
    writer.add_scalar('b_basic', b_basic, 0)
    writer.add_scalar('num_steps_check_accuracy', num_steps_check_accuracy, 0)
    writer.add_scalar('b_check_accuracy', b_check_accuracy, 0)

    annotation_string = domain + ' aa = ' + str(aa) + ' bb = ' + str(bb) + '\nsplit quartic Hamiltonian'
    writer.add_text('annotations', annotation_string, 0)

    # ------------- Training snapshots (no plotting) -------------
    training_plot_points_list = []

    # ------------- Training loop -------------
    for step in range(local_training_steps):
        z_np = get_z_init(b_basic)
        z_t = torch.from_numpy(z_np).float()

        last_z, _ = model.flow(z_t, num_steps_basic, record_traj=False)

        # enclosing area = π * max(||z||^2)
        radii_sq = torch.sum(last_z * last_z, dim=-1)
        enclosing_area = math.pi * torch.max(radii_sq)
        loss = enclosing_area

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # For saving / snapshots
        current_lst_z = last_z.detach().cpu().numpy()
        lss = loss.item()
        enc_ar = enclosing_area.item()

        writer.add_scalar('enclosing_area', enc_ar, global_step)
        writer.add_scalar('loss', lss, global_step)
        writer.add_scalar('lr', current_lr, global_step)

        if step % update_training_plot_save_steps == update_training_plot_save_steps - 1:
            training_plot_points_list.append(current_lst_z)

        print('using num steps: %s, local step: %s, global step: %s, current_lr: %s, loss: %s, enclosing area: %s'
              % (num_steps_basic, step, global_step, current_lr, lss, enc_ar))

        # Save variables
        if step % save_steps == save_steps - 1:
            torch.save(model.state_dict(), 'save_variables.pt')
            print('Saved variables.')

        # TensorBoard write_steps (already writing each step via writer)
        if step % write_steps == write_steps - 1:
            # nothing extra to do; kept for structural similarity
            pass

        # LR decay
        if global_step % decay_steps == decay_steps - 1:
            current_lr = decay_rate * current_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # Accuracy check
        if step % check_accuracy_steps == check_accuracy_steps - 1:
            print('using num steps: %s' % num_steps_check_accuracy)
            print('enclosing area for b = %s:' % b_check_accuracy)
            with torch.no_grad():
                z_check_np = get_z_init(b_check_accuracy)
                z_check_t = torch.from_numpy(z_check_np).float()
                last_z_check, _ = model.flow(z_check_t, num_steps_check_accuracy, record_traj=False)
                radii_sq_check = torch.sum(last_z_check * last_z_check, dim=-1)
                more_accurate_enclosing_area = math.pi * torch.max(radii_sq_check)
                lss_acc = more_accurate_enclosing_area.item()
            writer.add_scalar('more_accurate_enclosing_area', lss_acc, global_step)
            print(lss_acc)

    # Final save
    torch.save(model.state_dict(), 'save_variables.pt')
    print('Saved variables.')

    # ------------- Trajectory for movie -------------
    z_movie_np = get_z_init(b_movie)
    z_movie_t = torch.from_numpy(z_movie_np).float().to(device)
    z_movie_t.requires_grad_(True)

    _, trj_t = model.flow(z_movie_t, num_steps_movie, record_traj=True)

    toc = time.time()
    print('Total time elapsed: %s' % (toc - tic))

    trj = trj_t.detach().cpu().numpy()
    trj_name = 'trj_run' + str(run_num) + '.npy'
    np.save(trj_name, trj)
    print('saved ' + trj_name + ' (this is the flow for the time dependent Hamiltonian found at the end of training)')

    trn = np.array(training_plot_points_list)
    trn_name = 'trn_run' + str(run_num) + '.npy'
    np.save(trn_name, trn)
    print('saved ' + trn_name + ' (this is the current best embedding as a function of training time)')
    print('note: trn.npy only records the training progress every %s steps' % update_training_plot_save_steps)

    writer.close()


if __name__ == "__main__":
    main()
