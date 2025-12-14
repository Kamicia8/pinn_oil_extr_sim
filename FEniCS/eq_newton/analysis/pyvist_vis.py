import matplotlib.pyplot as plt
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx import fem, plot, mesh
from mpi4py import MPI
import pyvista
import csv
import h5py 
from pathlib import Path
import sys

metrics_file = "/mnt/c/PINN_mgr/FEniCS/eq_newton/results/metrics/metrics.csv"
xdmf_file = "/mnt/c/PINN_mgr/FEniCS/eq_newton/results/fields/solution.xdmf"
output_folder = Path("/mnt/c/PINN_mgr/FEniCS/eq_newton/analysis")
output_folder.mkdir(parents=True, exist_ok=True)

times, delta_L2, max_u, mean_u = [], [], [], []
try:
    with open(metrics_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            times.append(float(row[0]))
            delta_L2.append(float(row[1]))
            max_u.append(float(row[2]))
            mean_u.append(float(row[3]))
except FileNotFoundError:
    print(f"Błąd: Plik metryk nie znaleziony w lokalizacji: {metrics_file}")
    sys.exit(1)

comm = MPI.COMM_WORLD

with XDMFFile(comm, xdmf_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="mesh") 

    V = fem.functionspace(mesh, ("CG", 1))
    u = fem.Function(V)

    h5_file = xdmf_file.replace(".xdmf", ".h5")
    
    if times:
        last_step_index = len(times) 
        
        try:
            grid_name = f"u_{last_step_index}" 
            
            with h5py.File(h5_file, "r") as f:
                
                if f"/Function/{grid_name}" in f:
                    
                    function_group = f[f"/Function/{grid_name}"]
                    
                    dataset_name = list(function_group.keys())[0]
                    
                    full_path = f"/Function/{grid_name}/{dataset_name}"
                    
                    data_array = f[full_path][:]
                    
                    u.x.array[:] = data_array.flatten()
                
                else:
                    raise KeyError(f"grupa '{grid_name}' nie znaleziona w pliku HDF5 pod /Function/.")
            
        except Exception as e:
            print(f"błąd podczas wczytywania HDF5 {e}")
            sys.exit(1)
            
    else:
        raise ValueError("brak danych czasowych w pliku metryk")

if u.x.array.size == 0:
    print("nie udało się wczytać danych funkcji")
    sys.exit(1)
    

topology, cell_types, geometry = plot.vtk_mesh(u.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

grid.point_data["u"] = u.x.array
grid.set_active_scalars("u")

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(grid, show_edges=True, cmap="viridis", scalar_bar_args={'title': "u"})
plotter.view_xy()
plotter.screenshot(str(output_folder / "pyvista_plot.png"))
plotter.close()


#histogram wartości pola
plt.figure(figsize=(6,4))
plt.hist(u.x.array.real, bins=30, color='skyblue', edgecolor='k')
plt.xlabel("u")
plt.ylabel("Liczba punktów")
plt.title("Histogram wartości pola u w ostatnim kroku")
plt.grid(True)
plt.savefig(output_folder / "histogram.png")
plt.close()

print(f"Wykresy zapisane w folderze: {output_folder.resolve()}")