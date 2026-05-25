"""FEniCSx 热传导仿真 — 代码骨架模板

适用于稳态/瞬态标量扩散问题（热传导、Poisson 方程等）。
标注「根据实际修改」的部分需要根据具体问题调整。
"""
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

result_dir = os.environ.get("RESULT_DIR", "result")
os.makedirs(result_dir, exist_ok=True)

# === 1. 网格创建 === （根据实际修改）
# 2D 矩形: mesh.create_rectangle(MPI.COMM_WORLD, [[x0,y0],[x1,y1]], [nx,ny])
# 3D 长方体: mesh.create_box(MPI.COMM_WORLD, [[x0,y0,z0],[x1,y1,z1]], [nx,ny,nz])
# 复杂几何: 使用 gmsh 生成（调用 gmsh.model.occ，禁止指定 tag 参数）
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [32, 32])
gdim = domain.geometry.dim

# === 2. 标量函数空间 ===
V = fem.functionspace(domain, ("Lagrange", 1))

# === 3. 边界条件 === （根据实际修改）
def boundary_left(x):
    return np.isclose(x[0], 0.0)

def boundary_right(x):
    return np.isclose(x[0], 1.0)

dofs_left = fem.locate_dofs_geometrical(V, boundary_left)
dofs_right = fem.locate_dofs_geometrical(V, boundary_right)
bc_left = fem.dirichletbc(PETSc.ScalarType(100.0), dofs_left, V)    # 根据实际修改
bc_right = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_right, V)    # 根据实际修改
bcs = [bc_left, bc_right]

# === 4. 弱形式 === （根据实际修改）
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
k = fem.Constant(domain, PETSc.ScalarType(1.0))    # 导热系数（根据实际修改）
f = fem.Constant(domain, PETSc.ScalarType(0.0))    # 源项（根据实际修改）

a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# === 5. 求解 ===
problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options_prefix="heat_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
uh = problem.solve()
uh.name = "Temperature"

# === 6. 结果保存 ===
from dolfinx.io import VTXWriter
with VTXWriter(domain.comm, os.path.join(result_dir, "temperature.bp"), [uh]) as vtx:
    vtx.write(0.0)

# === 7. 可视化截图 (pyvista 离屏渲染) ===
import pyvista as pv
pv.OFF_SCREEN = True
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["Temperature"] = uh.x.array
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(grid, scalars="Temperature", cmap="coolwarm", show_edges=False)
plotter.add_scalar_bar(title="Temperature")
plotter.view_xy()
plotter.screenshot(os.path.join(result_dir, "temperature.png"))
plotter.close()

# === 8. 物理量输出 ===
print(f"Temperature range: [{uh.x.array.min():.6f}, {uh.x.array.max():.6f}]")
