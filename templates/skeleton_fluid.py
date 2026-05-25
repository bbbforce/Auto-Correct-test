"""FEniCSx Navier-Stokes 流体仿真 — 代码骨架模板

适用于不可压缩 Navier-Stokes 流体问题（腔流、管流等）。
使用增量压力修正方法 (IPCS) 进行时间步进。
标注「根据实际修改」的部分需要根据具体问题调整。
"""
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

result_dir = os.environ.get("RESULT_DIR", "result")
os.makedirs(result_dir, exist_ok=True)

# === 1. 网格创建 === （根据实际修改）
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [32, 32],
)
gdim = domain.geometry.dim

# === 2. 函数空间（Taylor-Hood P2/P1）===
V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
Q = fem.functionspace(domain, ("Lagrange", 1))

# === 3. 边界条件 === （根据实际修改）
def top_wall(x):
    return np.isclose(x[1], 1.0)

def no_slip(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

u_lid = fem.Function(V)
u_lid.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[0])))
dofs_top = fem.locate_dofs_geometrical(V, top_wall)
bc_top = fem.dirichletbc(u_lid, dofs_top)

dofs_noslip = fem.locate_dofs_geometrical(V, no_slip)
bc_noslip = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), dofs_noslip, V)
bcs_u = [bc_top, bc_noslip]

# === 4. 物理参数 === （根据实际修改）
dt = 0.001
T_end = 0.5
dt_c = fem.Constant(domain, default_scalar_type(dt))
mu_c = fem.Constant(domain, default_scalar_type(0.01))  # 运动粘度

# === 5. 变量定义 ===
u_n = fem.Function(V, name="u_n")
u_new = fem.Function(V, name="u_new")
p_n = fem.Function(Q, name="p_n")
p_new = fem.Function(Q, name="p_new")

u_trial = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)
p_trial = ufl.TrialFunction(Q)
q_test = ufl.TestFunction(Q)
f_body = fem.Constant(domain, np.zeros(gdim, dtype=default_scalar_type))

# === 6. IPCS 弱形式 ===
# Step 1: 暂估速度
F1 = (ufl.inner((u_trial - u_n) / dt_c, v_test) * ufl.dx
      + ufl.inner(ufl.grad(u_n) * u_n, v_test) * ufl.dx
      + mu_c * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
      + ufl.inner(ufl.grad(p_n), v_test) * ufl.dx
      - ufl.inner(f_body, v_test) * ufl.dx)
a1, L1 = ufl.lhs(F1), ufl.rhs(F1)

# Step 2: 压力修正
a2 = ufl.inner(ufl.grad(p_trial), ufl.grad(q_test)) * ufl.dx
L2 = -(1.0 / dt_c) * ufl.div(u_new) * q_test * ufl.dx

# Step 3: 速度修正
a3 = ufl.inner(u_trial, v_test) * ufl.dx
L3 = (ufl.inner(u_new, v_test) * ufl.dx
      - dt_c * ufl.inner(ufl.grad(p_new - p_n), v_test) * ufl.dx)

# === 7. 时间步进 ===
t, step = 0.0, 0
from dolfinx.io import VTXWriter
vtx_u = VTXWriter(domain.comm, os.path.join(result_dir, "velocity.bp"), [u_new])

while t < T_end:
    t += dt
    step += 1

    prob1 = LinearProblem(a1, L1, bcs=bcs_u,
                          petsc_options_prefix="step1_",
                          petsc_options={"ksp_type": "gmres", "pc_type": "ilu"})
    u_new.x.array[:] = prob1.solve().x.array

    prob2 = LinearProblem(a2, L2, bcs=[],
                          petsc_options_prefix="step2_",
                          petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
    p_new.x.array[:] = prob2.solve().x.array

    prob3 = LinearProblem(a3, L3, bcs=bcs_u,
                          petsc_options_prefix="step3_",
                          petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
    u_new.x.array[:] = prob3.solve().x.array

    u_n.x.array[:] = u_new.x.array
    p_n.x.array[:] = p_new.x.array

    if step % 50 == 0:
        vtx_u.write(t)
        vel = u_new.x.array.reshape(-1, gdim)
        print(f"Step {step}, t={t:.4f}, max|u|={np.linalg.norm(vel, axis=1).max():.6e}")

vtx_u.close()

# === 8. 物理量输出 ===
vel = u_new.x.array.reshape(-1, gdim)
print(f"Final max velocity: {np.linalg.norm(vel, axis=1).max():.6e}")
print(f"Final pressure range: [{p_new.x.array.min():.6e}, {p_new.x.array.max():.6e}]")
