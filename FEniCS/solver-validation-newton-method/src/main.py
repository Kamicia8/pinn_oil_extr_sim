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
import wandb

from utils import log_map, load_and_scale_kq, save_comparison_data
from physics import get_nonlinear_K, get_variational_form   


def main():
    if MPI.COMM_WORLD.rank == 0:
        run = wandb.init(project="FEniCS-solver-for-comparison") 
        params = wandb.config.set

        config = [params["dt"], params["T"], params["u_coeff"], params["h_type"]]
    else:
        config = [None, None, None, None]

    dt, T, u_coeff, h_type = MPI.COMM_WORLD.bcast(config, root=0)
    h_name_for_run = "h_0" if h_type == "h=0" else "h_sinusoidal"

    if MPI.COMM_WORLD.rank == 0:
        run.name = f"FEniCS_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}"

    DATA_PATH = '/mnt/c/PINN_mgr/FEniCS/solver-validation-newton-method/data/spe_model2_layer50.npy'
    kq_data_matrix, scale = load_and_scale_kq(DATA_PATH)
    ACTUAL_NY, ACTUAL_NX = kq_data_matrix.shape
    REAL_NX = ACTUAL_NX
    REAL_NY = ACTUAL_NY

    FT_TO_M = 0.3048
    REAL_LX = REAL_NX * 20.0 * FT_TO_M
    REAL_LY = REAL_NY * 10.0 * FT_TO_M   

    #ustawienia fenics
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([REAL_LX, REAL_LY])], [REAL_NX, REAL_NY])
    V = fem.functionspace(domain, ("Lagrange", 1))

    #interpolacja kq
    x_coords = np.linspace(0, REAL_LX, REAL_NX)
    y_coords = np.linspace(0, REAL_LY, REAL_NY)
    interp_kq = RegularGridInterpolator((y_coords, x_coords), kq_data_matrix, bounds_error=False, fill_value=1.0)

    Kq_func = fem.Function(V)
    Kq_func.interpolate(lambda x: interp_kq(np.stack((x[1], x[0]), axis=-1)))

    u_n, h, uh = fem.Function(V), fem.Function(V), fem.Function(V)
    u_n.interpolate(lambda x: np.where(np.sqrt((x[0]-REAL_LX/2)**2 + (x[1]-REAL_LY/2)**2) <= 30.0, 1.0, 0.0))

    if h_type == "h=0":
        h.interpolate(lambda x: 0*x[0])
    else:
        h.interpolate(lambda x: 1.0 + np.sin(2*np.pi*x[0]/REAL_LX) * np.sin(2*np.pi*x[1]/REAL_LY))


    #forma variacyjna
    v = ufl.TestFunction(V)
    K_func = get_nonlinear_K(uh, Kq_func, u_coeff)
    F = get_variational_form(uh, u_n, v, dt, K_func, h, domain)   

    #brzegi
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V) 

    problem = NonlinearProblem(F, uh, bcs=[bc], petsc_options={
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt", 
    "snes_rtol": 1e-6, 
    "snes_atol": 1e-6, 
    "ksp_type": "gmres", 
    "pc_type": "hypre", 
    "pc_hypre_type": "boomeramg"}, 
    petsc_options_prefix="TimeStep")


    Path("results/fields").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)


    xdmf_file = XDMFFile(domain.comm, f"results/fields/solution_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}.xdmf", "w") 
    xdmf_file.write_mesh(domain) 
    

    metrics_path = f"results/metrics/metrics_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}.csv"
    m_file = open(metrics_path, "w", newline="") if MPI.COMM_WORLD.rank == 0 else None
    csv_writer = csv.writer(m_file) if MPI.COMM_WORLD.rank == 0 else None
    if MPI.COMM_WORLD.rank == 0: csv_writer.writerow(["time", "energy", "delta_L2", "max_u", "mean_u"])


    num_steps = int(T/dt)
    log_interval = max(1, num_steps // 5) 

    log_map(domain, u_n, 0.0, "Stan początkowy", REAL_LX, REAL_LY, REAL_NX, REAL_NY)


    for n in range(num_steps):
        t = (n + 1) * dt
        problem.solve()

        num_its = problem.solver.getIterationNumber()
        uh.name = "Pressure"
        
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
            log_map(domain, uh, t, "Krok czasowy", REAL_LX, REAL_LY, REAL_NX, REAL_NY)

        u_n.x.array[:] = uh.x.array
        xdmf_file.write_function(uh, t)

    xdmf_file.close()

    if MPI.COMM_WORLD.rank == 0:
        m_file.close() 

    npy_path = f"results/metrics/fenics_for_pinn_T{T}_dt{dt}_u_coeff{u_coeff}_{h_name_for_run}.npy"
    save_comparison_data(domain, uh, REAL_LX, REAL_LY, REAL_NX, REAL_NY, kq_data_matrix, npy_path)

    if MPI.COMM_WORLD.rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()

