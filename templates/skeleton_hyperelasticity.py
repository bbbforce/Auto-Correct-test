"""FEniCSx 超弹性仿真 — 代码骨架模板

适用于大变形非线性弹性问题（Neo-Hookean, Mooney-Rivlin 等）。
使用 NonlinearProblem + NewtonSolver 求解。
标注「根据实际修改」的部分需要根据具体问题调整。
"""
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io, nls, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl

result_dir = os.environ.get("RESULT_DIR", "result")
os.makedirs(result_dir, exist_ok=True)

# === 1. 网格创建 === （根据实际修改）
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [1.0, 0.1, 0.1]],
    [20, 6, 6],
)
gdim = domain.geometry.dim

# === 2. 向量函数空间 ===
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# === 3. 边界条件 === （根据实际修改）
def fixed_end(x):
    return np.isclose(x[0], 0.0)

def loaded_end(x):
    return np.isclose(x[0], 1.0)

# 固定端
dofs_fixed = fem.locate_dofs_geometrical(V, fixed_end)
bc_fixed = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), dofs_fixed, V)

# 施加位移的端面（根据实际修改）
dofs_loaded = fem.locate_dofs_geometrical(V, loaded_end)
disp_val = np.array([0.1, 0.0, 0.0], dtype=default_scalar_type)   # 根据实际修改
bc_loaded = fem.dirichletbc(disp_val, dofs_loaded, V)

bcs = [bc_fixed, bc_loaded]

# === 4. 超弹性本构 (Neo-Hookean) === （根据实际修改）
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)

# 材料参数（根据实际修改）
E_val = 1e6
nu_val = 0.3
mu = fem.Constant(domain, default_scalar_type(E_val / (2.0 * (1.0 + nu_val))))
lmbda = fem.Constant(domain, default_scalar_type(E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))))

# 运动学
d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
J = ufl.variable(ufl.det(F))
Ic = ufl.variable(ufl.tr(C))

# 应变能密度 (Neo-Hookean)
psi = (mu / 2.0) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2.0) * ufl.ln(J)**2

# 第一 Piola-Kirchhoff 应力
P = ufl.diff(psi, F)

# 体力（根据实际修改）
B = fem.Constant(domain, np.zeros(gdim, dtype=default_scalar_type))

# 残差形式
F_form = ufl.inner(P, ufl.grad(v)) * ufl.dx - ufl.dot(B, v) * ufl.dx

# === 5. 非线性求解 ===
problem = NonlinearProblem(F_form, u, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# 求解器参数
solver.atol = 1e-8
solver.rtol = 1e-8
solver.max_it = 50
solver.convergence_criterion = "incremental"

# PETSc 线性求解器配置
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

num_its, converged = solver.solve(u)
assert converged, f"Newton solver did not converge after {num_its} iterations"
print(f"Newton solver converged in {num_its} iterations")

# === 6. 结果保存 ===
from dolfinx.io import VTXWriter
with VTXWriter(domain.comm, os.path.join(result_dir, "displacement.bp"), [u]) as vtx:
    vtx.write(0.0)

# === 7. 物理量输出 ===
disp = u.x.array.reshape(-1, gdim)
print(f"Max displacement magnitude: {np.linalg.norm(disp, axis=1).max():.6e}")
