# -*- coding: utf-8 -*-
"""
SDSC6008 Final Project — Proposal v2.2 对齐的 50 次仿真
设计: 2^3 全因子 (2 block x 4 组合 x 3 重复 = 24) + LHS 20 + CCD 轴向 6 = 50
仿真器: PyBaMM 26.10.0.0, DFN, Chen2020
响应: 比容量 mAh/g = Q_Ah / (3262 * t * AREA * (1-eps) * 1000) * 1000
输出: proposal_doe_dataset.csv
"""
import os
import numpy as np
import pandas as pd
import pybamm
from scipy.stats import qmc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "proposal_doe_dataset.csv")

model = pybamm.lithium_ion.DFN()
base = pybamm.ParameterValues("Chen2020")
AREA = float(base["Electrode height [m]"]) * float(base["Electrode width [m]"])
DENSITY = 3262.0  # kg/m^3
print("Electrode area (m2):", round(AREA, 6))

def solver_for(block):
    if block == 1:
        return pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
    elif block == 2:
        return pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    else:
        return pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)

def run(t_um, eps, r_um, block):
    pv = base.copy()
    pv.update({
        "Positive electrode thickness [m]": t_um * 1e-6,
        "Positive electrode porosity": eps,
        "Positive electrode active material volume fraction": 1 - eps,
        "Positive particle radius [m]": r_um * 1e-6,
    })
    sim = pybamm.Simulation(
        model, parameter_values=pv,
        experiment=pybamm.Experiment(["Discharge at 1 C until 3.0 V"]),
        solver=solver_for(block),
    )
    sol = sim.solve()
    Q = float(sol["Discharge capacity [A.h]"].entries[-1])
    mass_g = DENSITY * (t_um * 1e-6) * AREA * (1 - eps) * 1000.0
    spec = Q / mass_g * 1000.0  # Ah/kg == mAh/g
    return Q, mass_g, spec, float(sol["Time [s]"].entries[-1]), float(sol["Terminal voltage [V]"].entries[-1])

# ---------- 2^3 全因子 ----------
combos = {
    "(1)": (40, 0.20, 1),  "ab": (200, 0.60, 1),
    "ac": (200, 0.20, 12), "bc": (40, 0.60, 12),
    "a": (200, 0.20, 1),   "b": (40, 0.60, 1),
    "c": (40, 0.20, 12),   "abc": (200, 0.60, 12),
}
block_of = {"(1)": 1, "ab": 1, "ac": 1, "bc": 1, "a": 2, "b": 2, "c": 2, "abc": 2}

rng = np.random.default_rng(20261010)
rows = []
run_id = 0

for block in [1, 2]:
    combos_b = [c for c in combos if block_of[c] == block]
    plan = [(c, rep) for c in combos_b for rep in range(1, 4)]
    rng.shuffle(plan)
    for combo, rep in plan:
        t0, e0, r0 = combos[combo]
        # ±1% 均匀扰动
        t = t0 * rng.uniform(0.99, 1.01)
        e = e0 * rng.uniform(0.99, 1.01)
        r = r0 * rng.uniform(0.99, 1.01)
        Q, mass, spec, t_end, v_end = run(t, e, r, block)
        run_id += 1
        rows.append({
            "run": run_id, "design": "2^3", "block": block, "combo": combo, "replicate": rep,
            "A_thickness_um": round(t, 4), "B_porosity": round(e, 6), "C_radius_um": round(r, 4),
            "Q_Ah": round(Q, 6), "mass_g": round(mass, 5), "specific_capacity_mAh_g": round(spec, 4),
            "end_time_s": round(t_end, 1), "end_voltage_V": round(v_end, 4),
        })
    print(f"block {block} done")

# ---------- LHS 20 ----------
sampler = qmc.LatinHypercube(d=3, seed=20261010)
u = sampler.random(20)
lo = np.array([40.0, 0.20, 1.0]); hi = np.array([200.0, 0.60, 12.0])
pts = qmc.scale(u, lo, hi)
idx = np.arange(20); rng.shuffle(idx)
for k in idx:
    t, e, r = pts[k]
    Q, mass, spec, t_end, v_end = run(t, e, r, 0)
    run_id += 1
    rows.append({
        "run": run_id, "design": "LHS", "block": 0, "combo": "", "replicate": 1,
        "A_thickness_um": round(t, 4), "B_porosity": round(e, 6), "C_radius_um": round(r, 4),
        "Q_Ah": round(Q, 6), "mass_g": round(mass, 5), "specific_capacity_mAh_g": round(spec, 4),
        "end_time_s": round(t_end, 1), "end_voltage_V": round(v_end, 4),
    })
print("LHS done")

# ---------- CCD 轴向 6 ----------
center = (120.0, 0.40, 6.5)
axial = [
    (40, 0.40, 6.5), (200, 0.40, 6.5),
    (120, 0.20, 6.5), (120, 0.60, 6.5),
    (120, 0.40, 1), (120, 0.40, 12),
]
for k in rng.permutation(len(axial)):
    t, e, r = axial[k]
    Q, mass, spec, t_end, v_end = run(t, e, r, 0)
    run_id += 1
    rows.append({
        "run": run_id, "design": "CCD", "block": 0, "combo": "", "replicate": 1,
        "A_thickness_um": round(t, 4), "B_porosity": round(e, 6), "C_radius_um": round(r, 4),
        "Q_Ah": round(Q, 6), "mass_g": round(mass, 5), "specific_capacity_mAh_g": round(spec, 4),
        "end_time_s": round(t_end, 1), "end_voltage_V": round(v_end, 4),
    })
print("CCD done")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("\nrows:", len(df))
print(df.groupby("design")["specific_capacity_mAh_g"].agg(["count", "mean", "min", "max"]).round(3))
print("\nsaved:", OUT)
