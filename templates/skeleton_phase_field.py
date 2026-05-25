"""FEniCSx 相场断裂仿真 — 代码骨架模板

适用于准静态相场断裂问题（AT1/AT2 模型）。
使用交替最小化方法（staggered scheme）：位移场 u 和损伤场 d 交替求解。
标注「根据实际修改」的部分需要根据具体问题调整。
"""
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io, nls, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl

result_dir = os.environ.get("RESULT_DIR", "result")
os.makedirs(result_dir, exist_ok=True)

# === 1. 网格创建 === （根据实际修改）
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [1.0, 1.0]],
    [100, 100],
)
gdim = domain.geometry.dim

# === 2. 函数空间 ===
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))  # 位移场
W = fem.functionspace(domain, ("Lagrange", 1))             # 损伤场

# === 3. 边界条件 === （根据实际修改）
def bottom(x):
    return np.isclose(x[1], 0.0)

def top(x):
    return np.isclose(x[1], 1.0)

# 位移边界
dofs_bottom = fem.locate_dofs_geometrical(V, bottom)
bc_bottom = fem.dirichletbc(
    np.zeros(gdim, dtype=default_scalar_type), dofs_bottom, V
)

u_top = fem.Function(V)
dofs_top = fem.locate_dofs_geometrical(V, top)

# === 4. 材料参数 === （根据实际修改）
E_val = 210e3
nu_val = 0.3
Gc = 2.7e-3          # 临界能量释放率
l0 = 0.02            # 长度尺度参数
mu_val = E_val / (2.0 * (1.0 + nu_val))
lmbda_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

mu = fem.Constant(domain, default_scalar_type(mu_val))
lmbda = fem.Constant(domain, default_scalar_type(lmbda_val))
Gc_c = fem.Constant(domain, default_scalar_type(Gc))
l0_c = fem.Constant(domain, default_scalar_type(l0))

# === 5. 变量 ===
u = fem.Function(V, name="Displacement")
d = fem.Function(W, name="Damage")
d_old = fem.Function(W)

v = ufl.TestFunction(V)
w = ufl.TestFunction(W)

# === 6. 本构关系 ===
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma_undamaged(u):
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim) + 2 * mu * epsilon(u)

def psi_elastic(u):
    """弹性应变能密度"""
    eps = epsilon(u)
    return 0.5 * lmbda * ufl.tr(eps)**2 + mu * ufl.inner(eps, eps)

# 退化函数
def g(d):
    return (1.0 - d)**2 + 1e-6

# === 7. 位移子问题弱形式 ===
F_u = g(d) * ufl.inner(sigma_undamaged(u), epsilon(v)) * ufl.dx
J_u = ufl.derivative(F_u, u, ufl.TrialFunction(V))

# === 8. 损伤子问题弱形式 (AT2 模型) ===
d_trial = ufl.TrialFunction(W)
a_d = (
    (Gc_c / l0_c) * ufl.inner(d_trial, w) * ufl.dx
    + Gc_c * l0_c * ufl.inner(ufl.grad(d_trial), ufl.grad(w)) * ufl.dx
    + 2.0 * psi_elastic(u) * d_trial * w * ufl.dx
)
L_d = 2.0 * psi_elastic(u) * w * ufl.dx

# === 9. 载荷步循环 ===
load_steps = np.linspace(0, 0.01, 50)  # 根据实际修改

from dolfinx.io import VTXWriter
vtx_u = VTXWriter(domain.comm, os.path.join(result_dir, "displacement.bp"), [u])
vtx_d = VTXWriter(domain.comm, os.path.join(result_dir, "damage.bp"), [d])

for step_i, disp_val in enumerate(load_steps):
    # 更新边界条件
    u_top.interpolate(lambda x: (np.zeros_like(x[0]),
                                  np.full_like(x[0], disp_val)))
    bc_top = fem.dirichletbc(u_top, dofs_top)
    bcs_u = [bc_bottom, bc_top]

    # 交替最小化
    for alt_iter in range(20):
        # 求解位移
        problem_u = NonlinearProblem(F_u, u, bcs=bcs_u, J=J_u)
        solver_u = NewtonSolver(MPI.COMM_WORLD, problem_u)
        solver_u.atol = 1e-8
        solver_u.rtol = 1e-8
        solver_u.max_it = 50
        solver_u.solve(u)

        # 求解损伤
        problem_d = LinearProblem(
            a_d, L_d, bcs=[],
            petsc_options_prefix="damage_",
            petsc_options={"ksp_type": "cg", "pc_type": "hypre"},
        )
        d_new = problem_d.solve()

        # 不可逆性约束：d >= d_old
        d_new.x.array[:] = np.maximum(d_new.x.array, d_old.x.array)
        d.x.array[:] = d_new.x.array

        # 收敛检查
        diff = np.linalg.norm(d.x.array - d_old.x.array)
        if diff < 1e-5:
            break

    d_old.x.array[:] = d.x.array
    vtx_u.write(float(step_i))
    vtx_d.write(float(step_i))
    print(f"Load step {step_i}: disp={disp_val:.5f}, max_d={d.x.array.max():.4f}")

vtx_u.close()
vtx_d.close()

# === 10. 物理量输出 ===
disp = u.x.array.reshape(-1, gdim)
print(f"Max displacement: {np.linalg.norm(disp, axis=1).max():.6e}")
print(f"Max damage: {d.x.array.max():.4f}")
