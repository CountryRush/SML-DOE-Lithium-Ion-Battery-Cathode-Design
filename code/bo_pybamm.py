# -*- coding: utf-8 -*-
"""
BO on the ACTUAL PyBaMM simulator (not the surrogate).
Random vs BO-EI vs BO-UCB: 8 LHS initial + 20 iterations = 28 evals,
3 seeds each -> 252 PyBaMM runs.
Outputs in results_bo_sim/ (not used in the report yet).
"""
import os
import time
import numpy as np
import pandas as pd
import pybamm
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "bo_sim")
os.makedirs(OUT, exist_ok=True)

model = pybamm.lithium_ion.DFN()
params = pybamm.ParameterValues("Chen2020")
AREA = float(params["Electrode height [m]"]) * float(params["Electrode width [m]"])
DENSITY = 3262.0

# physical bounds
LO = np.array([40.0, 0.20, 1.0])
HI = np.array([200.0, 0.60, 12.0])

def to_phys(X):
    X = np.atleast_2d(X)
    return LO + X * (HI - LO)

_cache = {}

def sim_eval(x):
    """x: array of 3 normalized values in [0,1] -> specific capacity (mAh/g)"""
    key = tuple(np.round(x, 6))
    if key in _cache:
        return _cache[key]
    t_um, eps, r_um = to_phys(x)[0]
    pv = params.copy()
    pv.update({
        "Positive electrode thickness [m]": float(t_um) * 1e-6,
        "Positive electrode porosity": float(eps),
        "Positive electrode active material volume fraction": float(1 - eps),
        "Positive particle radius [m]": float(r_um) * 1e-6,
    })
    sim = pybamm.Simulation(
        model, parameter_values=pv,
        experiment=pybamm.Experiment(["Discharge at 1 C until 3.0 V"]),
        solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6),
    )
    sol = sim.solve()
    Q = float(sol["Discharge capacity [A.h]"].entries[-1])
    mass = DENSITY * float(t_um) * 1e-6 * AREA * (1 - eps) * 1000.0
    spec = Q / mass * 1000.0
    _cache[key] = spec
    return spec

def eval_batch(X):
    return np.array([sim_eval(x) for x in np.atleast_2d(X)])

def lhs(n, seed):
    return qmc.LatinHypercube(d=3, seed=seed).random(n)

def fit_gp(X, y):
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=[0.2]*3, nu=2.5,
             length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(1e-6, (1e-10, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    gp.fit(X, y)
    return gp

def run_random(seed):
    rng = np.random.default_rng(seed)
    X = rng.random((28, 3))
    y = eval_batch(X)
    return np.maximum.accumulate(y)

def run_bo(seed, acq):
    rng = np.random.default_rng(seed)
    X = lhs(8, seed); y = eval_batch(X)
    best = list(np.maximum.accumulate(y))
    for t in range(20):
        gp = fit_gp(X, y)
        cand = rng.random((2000, 3))
        mu, sd = gp.predict(cand, return_std=True)
        ybest = y.max()
        if acq == "EI":
            z = (mu - ybest - 0.01) / np.maximum(sd, 1e-9)
            a = (mu - ybest - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
        else:
            a = mu + 2.0 * sd
        xnew = cand[np.argmax(a)]
        ynew = sim_eval(xnew)
        X = np.vstack([X, xnew]); y = np.append(y, ynew)
        best.append(max(best[-1], ynew))
    return np.array(best), X, y

if __name__ == "__main__":
    t0 = time.time()
    rows = []
    curves = {"Random": [], "BO-EI": [], "BO-UCB": []}
    seeds = [1, 2, 3]

    for s in seeds:
        print(f"seed {s}: random ...", flush=True)
        r = run_random(s); curves["Random"].append(r)
        for method in ["BO-EI", "BO-UCB"]:
            print(f"seed {s}: {method} ...", flush=True)
            acq = "EI" if method == "BO-EI" else "UCB"
            c, X, y = run_bo(s, acq)
            curves[method].append(c)
            for i, (x, yy) in enumerate(zip(X, y), 1):
                ph = to_phys(x)[0]
                rows.append({"seed": s, "method": method, "eval": i,
                             "A_um": round(float(ph[0]), 3), "B": round(float(ph[1]), 6),
                             "C_um": round(float(ph[2]), 3), "capacity": round(float(yy), 4)})
        # log random evals
        rng = np.random.default_rng(s); Xr = rng.random((28, 3)); yr = eval_batch(Xr)
        for i, (x, yy) in enumerate(zip(Xr, yr), 1):
            ph = to_phys(x)[0]
            rows.append({"seed": s, "method": "Random", "eval": i,
                         "A_um": round(float(ph[0]), 3), "B": round(float(ph[1]), 6),
                         "C_um": round(float(ph[2]), 3), "capacity": round(float(yy), 4)})

    R = np.mean(curves["Random"], axis=0)
    E = np.mean(curves["BO-EI"], axis=0)
    U = np.mean(curves["BO-UCB"], axis=0)
    target = 300.0

    def ev_to(v):
        idx = np.where(v >= target)[0]
        return int(idx[0]) + 1 if len(idx) else None

    summary = pd.DataFrame({
        "method": ["Random", "BO-EI", "BO-UCB"],
        "final_best": [round(R[-1], 3), round(E[-1], 3), round(U[-1], 3)],
        "evals_to_300": [ev_to(R), ev_to(E), ev_to(U)],
    })
    summary.to_csv(os.path.join(OUT, "bo_sim_summary.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bo_sim_runs.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.5, 5))
    plt.plot(range(1, 29), R, "o-", label="Random")
    plt.plot(range(1, 29), E, "s-", label="BO-EI")
    plt.plot(range(1, 29), U, "^-", label="BO-UCB")
    plt.axhline(target, color="r", ls="--", lw=0.8, label=f"target={target}")
    plt.xlabel("Evaluations (PyBaMM simulations)")
    plt.ylabel("Best capacity found (mAh/g)")
    plt.title("BO on the actual PyBaMM simulator (3 seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_bo_sim.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUT, "bo_sim_report.txt"), "w", encoding="utf-8") as f:
        f.write("BO on actual PyBaMM simulator, 3 seeds, 28 evals each\n")
        f.write(f"Total wall time: {time.time()-t0:.1f} s\n\n")
        f.write(summary.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"\nDone in {time.time()-t0:.1f} s. Outputs: {OUT}")
