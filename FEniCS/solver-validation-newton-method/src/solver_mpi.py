from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from pathlib import Path
from dolfinx.io import XDMFFile
import csv
from scipy.interpolate import RegularGridInterpolator
import dolfinx.geometry as geometry
from scipy.interpolate import griddata
import wandb
import matplotlib.pyplot as plt

def log_map(domain, uh, t, label):
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
        U_plot = griddata(flat_pts, flat_values, (X, Y), method='linear')
        plt.figure(figsize=(10, 14))
        plt.imshow(U_plot, extent=[0, REAL_LX, 0, REAL_LY], cmap='jet', origin='lower')
        plt.colorbar(label="Ciśnienie u")
        plt.title(f"{label}  t={t:.2f}")
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

kq_data_matrix, scale = load_and_scale_kq('/mnt/c/PINN_mgr/FEniCS/solver-validation-newton-method/data/spe_model2_layer50.npy')

# REAL_NX, REAL_NY, REAL_NZ = 60, 220, 85
ACTUAL_NY, ACTUAL_NX = kq_data_matrix.shape
REAL_NX = ACTUAL_NX
REAL_NY = ACTUAL_NY


FT_TO_M = 0.3048
REAL_LX = REAL_NX * 20 * FT_TO_M 
REAL_LY = REAL_NY * 10 * FT_TO_M 


x_coords = np.linspace(0, REAL_LX, REAL_NX)
y_coords = np.linspace(0, REAL_LY, REAL_NY)
interp_kq = RegularGridInterpolator((y_coords, x_coords), kq_data_matrix, bounds_error=False, fill_value=1.0)

Path("results/fields").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)


if MPI.COMM_WORLD.rank == 0:
    run = wandb.init(project="PINN-FEniCS-Comparison") 
    
    params = wandb.config.set
    dt = params["dt"]
    T = params["T"]
    u_coeff = params["u_coeff"]
    h_type = params["h_type"]
    h_name_for_run = "h=0" if h_type == "h=0" else "h=sinusoidal"
    run.name = f"FEniCS_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}" 
else:
    dt, T, u_coeff, h_type = None, None, None, None

dt = MPI.COMM_WORLD.bcast(dt, root=0)
T = MPI.COMM_WORLD.bcast(T, root=0)
u_coeff = MPI.COMM_WORLD.bcast(u_coeff, root=0)
h_type = MPI.COMM_WORLD.bcast(h_type, root=0)

h_name_for_run = "h_0" if h_type == "h=0" else "h_sinusoidal"
name = f"FEniCS_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}"

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([REAL_LX, REAL_LY])], [REAL_NX, REAL_NY])
V = fem.functionspace(domain, ("Lagrange", 1))

Kq = fem.Function(V)
def kq_map(x):
    pts = np.stack((x[1], x[0]), axis=-1)
    return interp_kq(pts)
Kq.interpolate(kq_map)

u_n = fem.Function(V)
u_n.interpolate(lambda x: np.where(np.sqrt((x[0]-REAL_LX/2)**2 + (x[1]-REAL_LY/2)**2) <= 20.0, 1.0, 0.0))
h = fem.Function(V)

if h_type == "h=0":
    h.interpolate(lambda x: 0*x[0])
else:
    h.interpolate(lambda x: 1.0 + np.sin(2*np.pi*x[0]/REAL_LX) * np.sin(2*np.pi*x[1]/REAL_LY))

num_steps = int(T/dt)
log_interval = max(1, num_steps // 5) 


uh = fem.Function(V)
v = ufl.TestFunction(V)
def K(u, u_coeff): return Kq * ufl.exp(u_coeff * u)

F = ((uh - u_n)/dt * v * ufl.dx + ufl.inner(K(uh, u_coeff) * ufl.grad(uh), ufl.grad(v)) * ufl.dx - h * v * ufl.dx)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt", 
    "snes_rtol": 1e-6, 
    "snes_atol": 1e-6, "ksp_type": 
    "gmres", "pc_type": "hypre", 
    "pc_hypre_type": "boomeramg"}

problem = NonlinearProblem(F, uh, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="TimeStep")


xdmf_file = XDMFFile(MPI.COMM_WORLD, f"results/fields/solution_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}.xdmf", "w")
xdmf_file.write_mesh(domain) 

metrics_file = open(f"results/metrics/metrics_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}.csv", "w", newline="")
csv_writer = csv.writer(metrics_file)
csv_writer.writerow(["time", "energy", "delta_L2", "max_u", "mean_u"])

log_map(domain, u_n, 0.0, "Stan Początkowy")

for n in range(num_steps):
    t = (n + 1) * dt
    problem.solve()
    num_its = problem.solver.getIterationNumber()
    uh.name = "Pressure"
    xdmf_file.write_function(uh, t)
    
    dx = ufl.dx(domain=domain)
    vol = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * dx)), op=MPI.SUM)
    mean_u = domain.comm.allreduce(fem.assemble_scalar(fem.form(uh * dx)), op=MPI.SUM) / vol
    max_u = domain.comm.allreduce(np.max(uh.x.array), op=MPI.MAX)
    energy = domain.comm.allreduce(fem.assemble_scalar(fem.form(uh**2 * dx)), op=MPI.SUM)
    delta_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_n)**2 * dx)), op=MPI.SUM))
    
    if MPI.COMM_WORLD.rank == 0:
        csv_writer.writerow([t, energy, delta_L2, max_u, mean_u]) 
        print(f"Step {n+1}/{num_steps}, t={t:.3f}, max_u: {max_u:.4e}, iterations: {num_its}")
        wandb.log({
            "time": t, 
            "max_u": max_u, 
            "mean_u": mean_u, 
            "energy": energy, 
            "delta_L2": delta_L2, 
            "newton_iterations": num_its
            })

    if (n + 1) % log_interval == 0 or (n + 1) == num_steps:
        log_map(domain, uh, t, "Krok czasowy")

    u_n.x.array[:] = uh.x.array

xdmf_file.close()
metrics_file.close()

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
    np.save(f"results/metrics/fenics_for_pinn_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}.npy", comparison_data)
    wandb.finish()