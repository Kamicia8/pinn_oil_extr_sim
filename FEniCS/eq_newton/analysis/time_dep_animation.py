import numpy as np
import csv
import h5py
from pathlib import Path
import sys

from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem, plot
import pyvista
import matplotlib as mpl

metrics_file = "/mnt/c/PINN_mgr/FEniCS/eq_newton/results/metrics/metrics.csv"
xdmf_file = "/mnt/c/PINN_mgr/FEniCS/eq_newton/results/fields/solution.xdmf"
h5_file = xdmf_file.replace(".xdmf", ".h5")

output_folder = Path("/mnt/c/PINN_mgr/FEniCS/eq_newton/analysis")
output_folder.mkdir(parents=True, exist_ok=True)
gif_path = output_folder / "u_time.gif"

times = []
try:
    with open(metrics_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            times.append(float(row[0]))
except FileNotFoundError:
    print(f"nie znaleziono {metrics_file}")
    sys.exit(1)

if not times:
    raise ValueError("Brak danych czasowych w metrics.csv")

comm = MPI.COMM_WORLD

with XDMFFile(comm, xdmf_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="mesh")

V = fem.functionspace(mesh, ("CG", 1))
u = fem.Function(V)

topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["u"] = u.x.array
grid.set_active_scalars("u")

plotter = pyvista.Plotter(off_screen=True)
plotter.open_gif(str(gif_path))

plotter.add_mesh(
    grid,
    show_edges=True,
    cmap=mpl.colormaps["viridis"],
    scalar_bar_args={"title": "u"},
    lighting=False,
)

plotter.view_xy()
plotter.camera.zoom(1.3)

with h5py.File(h5_file, "r") as f:

    for k, t in enumerate(times, start=1):
        grid_name = f"u_{k}"

        if f"/Function/{grid_name}" not in f:
            print(f"pomijam {grid_name} (brak w HDF5)")
            continue

        function_group = f[f"/Function/{grid_name}"]
        dataset_name = list(function_group.keys())[0]
        data = function_group[dataset_name][:]

        u.x.array[:] = data.flatten()
        grid.point_data["u"][:] = u.x.array

        text_actor = plotter.add_text(
            f"t = {t:.4f}",
            position="upper_left",
            font_size=12
        )
        plotter.write_frame()
        plotter.remove_actor(text_actor)


plotter.close()

if comm.rank == 0:
    print(f"gif zapisany do {gif_path}")
