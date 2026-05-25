"""FEniCSx 线弹性仿真 — 代码骨架模板

适用于小变形线弹性力学问题（应力分析、位移计算等）。
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
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [1.0, 0.2, 0.2]],   # 根据实际修改
    [20, 6, 6],                              # 根据实际修改
)
gdim = domain.geometry.dim

# === 2. 向量函数空间 ===
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# === 3. 边界条件 === （根据实际修改）
def fixed_boundary(x):
    return np.isclose(x[0], 0.0)    # 根据实际修改

dofs_fixed = fem.locate_dofs_geometrical(V, fixed_boundary)
bc = fem.dirichletbc(np.zeros(gdim, dtype=PETSc.ScalarType), dofs_fixed, V)

# === 4. 材料参数 === （根据实际修改）
E = 210e9       # 杨氏模量 (Pa)
nu = 0.3        # 泊松比
mu = fem.Constant(domain, PETSc.ScalarType(E / (2.0 * (1.0 + nu))))
lmbda = fem.Constant(domain, PETSc.ScalarType(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))

# === 5. 本构关系 & 弱形式 ===
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(gdim) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# 体积力（根据实际修改）
f_body = fem.Constant(domain, np.array([0.0, 0.0, -1e4], dtype=PETSc.ScalarType))
# 面力（根据实际修改）
T_surface = fem.Constant(domain, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f_body, v) * ufl.dx + ufl.dot(T_surface, v) * ufl.ds

# === 6. 求解 ===
problem = LinearProblem(
    a, L, bcs=[bc],
    petsc_options_prefix="elasticity_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
uh = problem.solve()
uh.name = "Displacement"

# === 7. Von Mises 应力计算 ===
s_dev = sigma(uh) - (1.0 / 3.0) * ufl.tr(sigma(uh)) * ufl.Identity(gdim)
von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s_dev, s_dev))
V_vm = fem.functionspace(domain, ("DG", 0))
von_mises = fem.Function(V_vm, name="VonMises")
expr = fem.Expression(von_mises_expr, V_vm.element.interpolation_points())
von_mises.interpolate(expr)

# === 8. 结果保存 ===
from dolfinx.io import VTXWriter
with VTXWriter(domain.comm, os.path.join(result_dir, "displacement.bp"), [uh]) as vtx:
    vtx.write(0.0)

# === 9. 可视化截图 (pyvista 离屏渲染) ===
import pyvista as pv
pv.OFF_SCREEN = True

# 位移云图
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["Displacement"] = uh.x.array.reshape(-1, gdim)
warped = grid.warp_by_vector("Displacement", factor=1.0)
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(warped, scalars=np.linalg.norm(uh.x.array.reshape(-1, gdim), axis=1),
                 cmap="turbo", show_edges=False)
plotter.add_scalar_bar(title="Displacement Magnitude")
plotter.screenshot(os.path.join(result_dir, "displacement.png"))
plotter.close()

# === 10. 物理量输出 ===
disp = uh.x.array.reshape(-1, gdim)
print(f"Max displacement magnitude: {np.linalg.norm(disp, axis=1).max():.6e}")
print(f"Max Von Mises stress: {von_mises.x.array.max():.6e}")
