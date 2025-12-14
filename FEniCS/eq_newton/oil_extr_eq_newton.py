from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from pathlib import Path
from dolfinx.io import XDMFFile
import csv

# --- Przygotowanie folderów ---
Path("results/fields").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

# --- Parametry siatki i dziedziny ---
nx, ny = 8, 8
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

V = fem.functionspace(domain, ("Lagrange", 1))

# --- Warunki początkowe ---
u_n = fem.Function(V)
x = domain.geometry.x
center = np.array([0.5, 0.5])
radius = 0.2
u_n.x.array[:] = np.array(
    [1.0 if np.linalg.norm(xi[:2] - center) <= radius else 0.0 for xi in x]
)

# --- Funkcja źródła ---
h = fem.Function(V)
h.interpolate(lambda x: 1.0 + np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))

# --- Heterogeniczny współczynnik ---
Kq = fem.Function(V)
Kq.interpolate(lambda x: 1.0 + 9.0 * np.random.rand(len(x[0])))

# --- Parametry czasowe ---
dt = 0.01
T = 0.1
num_steps = int(T/dt)

# --- Warunki brzegowe ---
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)

# --- Funkcja trial i test ---
uh = fem.Function(V)
v = ufl.TestFunction(V)

def K(u):
    return Kq * ufl.exp(10*u)

# --- Równanie nieliniowe ---
F = (
    (uh - u_n)/dt * v * ufl.dx
    + ufl.inner(K(uh) * ufl.grad(uh), ufl.grad(v)) * ufl.dx
    - h * v * ufl.dx
)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_rtol": 1e-6,
    "snes_atol": 1e-6,
    "snes_monitor": None,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

problem = NonlinearProblem(F, uh, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="TimeStep")

# --- Plik do zapisu rozwiązania ---
xdmf_file = XDMFFile(MPI.COMM_WORLD, "results/fields/solution.xdmf", "w")
xdmf_file.write_mesh(domain)

# --- Plik CSV do zapisu metryk ---
metrics_file = open("results/metrics/metrics.csv", "w", newline="")
csv_writer = csv.writer(metrics_file)
csv_writer.writerow(["time", "delta_L2", "max_u", "mean_u"])

# --- Funkcja pomocnicza do normy L2 względem poprzedniego kroku ---
def compute_delta_L2(u, u_prev):
    local_error = fem.assemble_scalar(fem.form((u - u_prev)**2 * ufl.dx))
    return np.sqrt(domain.comm.allreduce(local_error, op=MPI.SUM))

# --- Pętla czasowa ---
for n in range(num_steps):
    problem.solve()
    
    # --- zapis XDMF ---
    uh.name = f"u_{n+1}"
    xdmf_file.write_function(uh, float(n+1)*dt)
    
    # --- metryki ---
    delta_L2 = compute_delta_L2(uh, u_n)
    max_u = domain.comm.allreduce(np.max(uh.x.array), op=MPI.MAX)
    mean_u = domain.comm.allreduce(np.mean(uh.x.array), op=MPI.SUM)  # globalna średnia
    
    if MPI.COMM_WORLD.rank == 0:
        csv_writer.writerow([dt*(n+1), delta_L2, max_u, mean_u])
        print(f"Step {n+1}, ΔL2: {delta_L2:.3e}, max: {max_u:.3e}, mean: {mean_u:.3e}")
    
    u_n.x.array[:] = uh.x.array

xdmf_file.close()
metrics_file.close()


#wizualizacja
# from dolfinx import plot
# import pyvista

# topology, cell_types, geometry = plot.vtk_mesh(V)

# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# grid.point_data["u"] = uh.x.array.real
# grid.set_active_scalars("u")

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True, cmap="viridis")
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     plotter.show()

