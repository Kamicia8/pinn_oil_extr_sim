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

#dane SPE
REAL_NX, REAL_NY = 100, 20
REAL_LX, REAL_LY = 762.0, 15.24

#skalowanie danych do zakresu [1, 10] 
def load_and_scale_kq(path):
    raw_data = np.loadtxt(path).flatten()[:2000]
    k_min, k_max = raw_data.min(), raw_data.max()
    k_scaled = (raw_data - k_min) / (k_max - k_min) * (10.0 - 1.0) + 1.0
    return k_scaled.reshape((REAL_NY, REAL_NX))

kq_data_matrix = load_and_scale_kq(r'/mnt/c/PINN_mgr/perm_case1.dat')

#interpolator
x_coords = np.linspace(0, REAL_LX, REAL_NX)
y_coords = np.linspace(0, REAL_LY, REAL_NY)
interp_kq = RegularGridInterpolator((y_coords, x_coords), kq_data_matrix, 
                                    bounds_error=False, fill_value=1.0)


Path("results/fields").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

# nx, ny = 8, 8
# domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
domain = mesh.create_rectangle(MPI.COMM_WORLD, 
                               [np.array([0, 0]), np.array([REAL_LX, REAL_LY])], 
                               [REAL_NX, REAL_NY])

V = fem.functionspace(domain, ("Lagrange", 1))

Kq = fem.Function(V)
def kq_map(x):
    # Mapowanie punktów siatki MES na wartości z interpolatora
    pts = np.stack((x[1], x[0]), axis=-1)
    return interp_kq(pts)

Kq.interpolate(kq_map)

u_n = fem.Function(V)
u_n.interpolate(lambda x: np.where(np.sqrt((x[0]-REAL_LX/2)**2 + (x[1]-REAL_LY/2)**2) <= 2.0, 1.0, 0.0))

h = fem.Function(V)
#funkcja źródła h(x) w 2D
h.interpolate(lambda x: 1.0 + np.sin(2*np.pi*x[0]/REAL_LX) * np.sin(2*np.pi*x[1]/REAL_LY))

dt = 0.001
T = 2.0
num_steps = int(T/dt)

uh = fem.Function(V)
v = ufl.TestFunction(V)

def K(u):
    return Kq * ufl.exp(10 * u)

F = (
    (uh - u_n)/dt * v * ufl.dx
    + ufl.inner(K(uh) * ufl.grad(uh), ufl.grad(v)) * ufl.dx
    - h * v * ufl.dx
)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt", 
    "snes_rtol": 1e-6,
    "snes_atol": 1e-6,
    "ksp_type": "gmres",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

problem = NonlinearProblem(F, uh, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="TimeStep")


xdmf_file = XDMFFile(MPI.COMM_WORLD, "results/fields/solution.xdmf", "w")
xdmf_file.write_mesh(domain)

metrics_file = open("results/metrics/metrics.csv", "w", newline="")
csv_writer = csv.writer(metrics_file)
csv_writer.writerow(["time", "delta_L2", "max_u", "mean_u"])

for n in range(num_steps):
    t = (n + 1) * dt
    problem.solve()
    snes = problem.solver
    num_its = snes.getIterationNumber()

    uh.name = "Pressure"
    xdmf_file.write_function(uh, t)
    
    #metryki
    dx = ufl.dx(domain=domain)

    vol = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(1.0 * dx)), op=MPI.SUM
    )

    mean_u = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(uh * dx)), op=MPI.SUM
    ) / vol

    max_u = domain.comm.allreduce(np.max(uh.x.array), op=MPI.MAX)

    delta_L2_local = fem.assemble_scalar(
    fem.form((uh - u_n)**2 * dx)
    )

    delta_L2 = np.sqrt(
        domain.comm.allreduce(delta_L2_local, op=MPI.SUM)
    )
    
    if MPI.COMM_WORLD.rank == 0:
        csv_writer.writerow([t, delta_L2, max_u, mean_u]) 
        print(f"Step {n+1}/{num_steps}, t={t:.3f}, max_u: {max_u:.4e}, iterations: {num_its}")
    
    u_n.x.array[:] = uh.x.array

xdmf_file.close()
metrics_file.close()


def get_values_at_points(points, function):
    from dolfinx import geometry

    # points: (N, 2) → (N, 3)
    points_3d = np.zeros((points.shape[0], 3), dtype=np.float64)
    points_3d[:, :2] = points

    # 1. Bounding box tree – UWAGA: przekazujemy DOMAIN, nie topology
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    # 2. Szukamy kandydatów komórek
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)

    # 3. Sprawdzamy faktyczne kolizje
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    values = np.full(points.shape[0], np.nan)

    for i in range(points.shape[0]):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            values[i] = function.eval(points_3d[i:i+1], np.array([cells[0]], dtype=np.int32))[0]

    return values


#tworzymy siatkę punktów do porównania 100x100
x_test = np.linspace(0, REAL_LX, 100)
y_test = np.linspace(0, REAL_LY, 100)
X, Y = np.meshgrid(x_test, y_test)
pts_to_eval = np.vstack((X.flatten(), Y.flatten())).T

# 2. Pobieramy wartości końcowego rozwiązania uh
if MPI.COMM_WORLD.rank == 0:
    u_fenics_values = get_values_at_points(pts_to_eval, uh)
else:
    u_fenics_values = None

# 3. Zapisujemy do pliku, który łatwo wczytasz w Pythonie/PINN
if MPI.COMM_WORLD.rank == 0:
    comparison_data = {
        "x": X,
        "y": Y,
        "u_fenics": u_fenics_values.reshape(X.shape)
    }
    np.save("results/metrics/fenics_for_pinn_comparison.npy", comparison_data)