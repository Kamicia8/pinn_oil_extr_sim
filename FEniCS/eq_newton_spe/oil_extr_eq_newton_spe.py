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

#dane SPE model 1, więcej info w spe_data.ipynb
REAL_NX, REAL_NY = 100, 20
REAL_LX, REAL_LY = 762.0, 15.24

#skalowanie danych do zakresu [1, 10] 
def load_and_scale_kq(path):
    raw_data = np.loadtxt(path).flatten()[:2000] #zamieniay na długi wektor i bierzemy 200 pierwszych elementów bo 100*20 to 2000
    k_min, k_max = raw_data.min(), raw_data.max()
    k_scaled = (raw_data - k_min) / (k_max - k_min) * (10.0 - 1.0) + 1.0
    return k_scaled.reshape((REAL_NY, REAL_NX)) #wracamy do kształtu macierzy

kq_data_matrix = load_and_scale_kq(r'/mnt/c/PINN_mgr/perm_case1.dat')

#interpolator
x_coords = np.linspace(0, REAL_LX, REAL_NX)
y_coords = np.linspace(0, REAL_LY, REAL_NY)

#ciagła przestrzeń geometryczna do symulacji, wartości poza danymi przyjmują wartość 1.0 -> bounds_error=False, fill_value=1.0
interp_kq = RegularGridInterpolator((y_coords, x_coords), kq_data_matrix, 
                                    bounds_error=False, fill_value=1.0)


#tworzenie katalogów na wyniki
Path("results/fields").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

#MPI.COMM_WORLD to komunikator dla wszystkich procesów
#tworze prostokąt o początku w 0,0 i końcu w wymiarach podanych w SPE, calosć dziele na 100x20 elementów
domain = mesh.create_rectangle(MPI.COMM_WORLD, 
                               [np.array([0, 0]), np.array([REAL_LX, REAL_LY])], 
                               [REAL_NX, REAL_NY])


#przestrzeń funkcyjna Lagrangea gdzie stopień wielomianu to 1
V = fem.functionspace(domain, ("Lagrange", 1))

Kq = fem.Function(V)
def kq_map(x):
    # Mapowanie punktów siatki MES na wartości z interpolatora
    pts = np.stack((x[1], x[0]), axis=-1) #zamieniony x z y, odwrotnie do tego jak zdefiniowaliśmy interpolator
    return interp_kq(pts) #interpolator oblicza wartość przepuszczalności w tych punktach

Kq.interpolate(kq_map) #wypełniamy siatkę wartościami przepuszczalności

u_n = fem.Function(V)
#początkowe warunki u(x,0), jest w kółku daj ciśnienie 1 a jeśli nie to 0
u_n.interpolate(lambda x: np.where(np.sqrt((x[0]-REAL_LX/2)**2 + (x[1]-REAL_LY/2)**2) <= 2.0, 1.0, 0.0))

h = fem.Function(V)
#funkcja źródła h(x) w 2D
h.interpolate(lambda x: 1.0 + np.sin(2*np.pi*x[0]/REAL_LX) * np.sin(2*np.pi*x[1]/REAL_LY))

dt = 0.001
T = 2.0
num_steps = int(T/dt)

#nienane rozwiązanie
uh = fem.Function(V)

#funkcja testowa do słabego sformułowania
v = ufl.TestFunction(V)

#ze wzoru
def K(u):
    return Kq * ufl.exp(10 * u)

#słabe równanie
F = (
    (uh - u_n)/dt * v * ufl.dx #dyskretyzacja w czasie Eulera w przód
    + ufl.inner(K(uh) * ufl.grad(uh), ufl.grad(v)) * ufl.dx
    - h * v * ufl.dx
)

fdim = domain.topology.dim - 1 #wymiar krawędzi
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim) #związe między wnętrzem a krawędziami
boundary_facets = mesh.exterior_facet_indices(domain.topology) #indeksy zewnętrznych krawędzi
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V) #Dirichlet BC u=0 na całej granicy

petsc_options = {
    "snes_type": "newtonls", #metoda Netwona
    "snes_linesearch_type": "bt", #jeśli krok newtona zbyt duży to się wycofa
    "snes_rtol": 1e-6, #solver kończy pracę, gdy błąd względny spadnie o 6 rzędów wielkości.
    "snes_atol": 1e-6, #lub gdy błąd bezwzględny spadnie poniżej tej wartości.
    "ksp_type": "gmres", #Generalized Minimal Residual method
    "pc_type": "hypre", 
    "pc_hypre_type": "boomeramg", #Algebraic Multigrid (AMG)
}


problem = NonlinearProblem(F, uh, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="TimeStep")


xdmf_file = XDMFFile(MPI.COMM_WORLD, "results/fields/solution.xdmf", "w") #dane wizualne
xdmf_file.write_mesh(domain) 

metrics_file = open("results/metrics/metrics.csv", "w", newline="")
csv_writer = csv.writer(metrics_file)
csv_writer.writerow(["time", "delta_L2", "max_u", "mean_u"])

for n in range(num_steps):
    t = (n + 1) * dt #aktualny czas
    problem.solve() #rozwiązanie równania nieliniowego
    snes = problem.solver #dostep do solvera
    num_its = snes.getIterationNumber() #liczba iteracji newtona

    uh.name = "Pressure"
    xdmf_file.write_function(uh, t) #aktualne pole ciśnienia
    
    #metryki
    dx = ufl.dx(domain=domain) #miara całkowania

    #allReduce sumuje ze wszystkich procesorów
    #całka z 1.0 po całej domenie to objętość
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
    
    #aktualizacja rozwiązania na kolejny krok czasowy
    u_n.x.array[:] = uh.x.array

xdmf_file.close()
metrics_file.close()


# DANE DO PORÓWNANIA Z PINN 

x_test = np.linspace(0, REAL_LX, 100)
y_test = np.linspace(0, REAL_LY, 100)
X, Y = np.meshgrid(x_test, y_test)
pts_to_eval = np.vstack((X.flatten(), Y.flatten())).T
points_3d = np.zeros((pts_to_eval.shape[0], 3), dtype=np.float64)
points_3d[:, :2] = pts_to_eval

bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

local_values = []
local_pts = []

for i in range(points_3d.shape[0]):
    cells = colliding_cells.links(i)
    if len(cells) > 0:
        val = uh.eval(points_3d[i:i+1], np.array([cells[0]], dtype=np.int32))
        local_values.append(val[0])
        local_pts.append(pts_to_eval[i])

local_values = np.array(local_values)
local_pts = np.array(local_pts)

all_values = domain.comm.gather(local_values, root=0)
all_pts = domain.comm.gather(local_pts, root=0)

if domain.comm.rank == 0:
    flat_values = np.concatenate(all_values)
    flat_pts = np.concatenate(all_pts)
    
    U_final = griddata(flat_pts, flat_values, (X, Y), method='linear')

    comparison_data = {
        "x": X,
        "y": Y,
        "u_fenics": U_final
    }
    
    output_path = "results/metrics/fenics_for_pinn_comparison.npy"
    np.save(output_path, comparison_data)