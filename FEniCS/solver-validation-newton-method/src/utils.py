import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.interpolate import griddata
import dolfinx.geometry as geometry

def log_map(domain, uh, t, label, REAL_LX, REAL_LY, REAL_NX, REAL_NY):

    x_test = np.linspace(0, REAL_LX, REAL_NX) 
    y_test = np.linspace(0, REAL_LY, REAL_NY)
    X, Y = np.meshgrid(x_test, y_test)

    pts_to_eval = np.vstack((X.flatten(), Y.flatten())).T
    points_3d = np.zeros((pts_to_eval.shape[0], 3), dtype=np.float64)
    points_3d[:, :2] = pts_to_eval

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    local_values, local_pts = [], []

    for i in range(points_3d.shape[0]):
        cells = colliding_cells.links(i) #indeks komórki, która zawiera punkt
        if len(cells) > 0:
            val = uh.eval(points_3d[i:i+1], np.array([cells[0]], dtype=np.int32))
            local_values.append(val[0])
            local_pts.append(pts_to_eval[i])

    all_values = domain.comm.gather(np.array(local_values), root=0)
    all_pts = domain.comm.gather(np.array(local_pts), root=0)

    if domain.comm.rank == 0:
        flat_values = np.concatenate(all_values)
        flat_pts = np.concatenate(all_pts)

        U_plot = griddata(flat_pts, flat_values, (X, Y), method='linear')

        plt.figure(figsize=(10, 14))

        im = plt.imshow(U_plot, extent=[0, REAL_LX, 0, REAL_LY], cmap='jet', origin='lower')
        
        cbar = plt.colorbar(im)
        cbar.set_label("Ciśnienie u", fontsize=22)
        cbar.ax.tick_params(labelsize=22, rotation=35)
        
        plt.xticks(fontsize=20) 
        plt.yticks(fontsize=20)
        plt.title(f"{label}  t={t:.2f}", fontsize=22)
        # plt.tight_layout()

        wandb.log({"evolution_map": wandb.Image(plt)}, commit=False)

        plt.close()


def load_and_scale_kq(path):
    kq_layer = np.load(path)
    log_kq = np.log10(kq_layer + 1e-6)
    max_scale = 10.0
    l_min, l_max = log_kq.min(), log_kq.max()
    k_scaled = (log_kq - l_min) / (l_max - l_min) * (max_scale - 1.0) + 1.0
    scale = (1.0, max_scale)
    return k_scaled, scale


def save_comparison_data(domain, uh, REAL_LX, REAL_LY, REAL_NX, REAL_NY, kq_data_matrix, filename):
    
    x_test = np.linspace(0, REAL_LX, REAL_NX) 
    y_test = np.linspace(0, REAL_LY, REAL_NY)
    X, Y = np.meshgrid(x_test, y_test)

    pts_to_eval = np.vstack((X.flatten(), Y.flatten())).T
    points_3d = np.zeros((pts_to_eval.shape[0], 3), dtype=np.float64)
    points_3d[:, :2] = pts_to_eval

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    local_values, local_pts = [], []
    for i in range(points_3d.shape[0]):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            val = uh.eval(points_3d[i:i+1], np.array([cells[0]], dtype=np.int32))
            local_values.append(val[0])
            local_pts.append(pts_to_eval[i])

    all_values = domain.comm.gather(np.array(local_values), root=0)
    all_pts = domain.comm.gather(np.array(local_pts), root=0)

    if domain.comm.rank == 0:
        flat_values = np.concatenate(all_values)
        flat_pts = np.concatenate(all_pts)
        U_final = griddata(flat_pts, flat_values, (X, Y), method='linear')
        
        comparison_data = {"x": X, "y": Y, "u_fenics": U_final, "kq_scaled": kq_data_matrix}
        np.save(filename, comparison_data)
