"""
Оценка показателя Ходжеса–Лемана rho(theta) по результатам power-эксперимента.

Модель экспоненциального хвоста:  1 - pi_n ~ C * exp(-rho * n)
                                  ln(1 - pi_n) ~ ln(C) - rho * n

Оценка наклона rho = -slope взвешенным МНК.

Веса (дельта-метод):  Var(ln(1-pi_hat)) ~ pi / (M (1-pi))  =>  w = M (1-pi) / pi.
Разброс остатков sigma^2 оценивается из взвешенных остатков — чтобы доверительный
интервал честно отражал и шум хвоста, и предасимптотическую кривизну.
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MC = 5000

n = np.array([1000, 2000, 4000, 6000, 8000, 10000, 15000])
power = {
    "KS": np.array([0.135, 0.248, 0.454, 0.671, 0.819, 0.885, 0.980]),
    "CVM": np.array([0.215, 0.386, 0.650, 0.828, 0.927, 0.967, 0.997]),
}


def estimate_rho(n, p, mc, n_min):
    """Weighted LS of ln(1-p) = a + b*n over n >= n_min. Returns dict with rho, se, CI, sigma2, fit data."""
    mask = (n >= n_min) & (p > 0) & (p < 1)
    nn = n[mask].astype(float)
    pp = p[mask]
    y = np.log(1 - pp)
    w = mc * (1 - pp) / pp

    X = np.column_stack([np.ones(len(nn)), nn])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)

    resid = y - X @ beta
    df = len(nn) - 2
    sigma2 = float((resid.T @ W @ resid) / df)  # >1 => шум/несоответствие модели больше дельта-метода

    cov = sigma2 * np.linalg.inv(XtWX)

    slope = beta[1]
    rho = -slope
    se = float(np.sqrt(cov[1, 1]))
    tcrit = stats.t.ppf(0.975, df)
    ci_lo = rho - tcrit * se
    ci_hi = rho + tcrit * se

    return {"rho": rho, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "sigma2": sigma2, "nn": nn, "y": y, "beta": beta}


def are_with_ci(rC, seC, rK, seK, df):
    are = rC / rK
    se_log = np.sqrt((seC / rC) ** 2 + (seK / rK) ** 2)
    tcrit = stats.t.ppf(0.975, df)
    lo = np.exp(np.log(are) - tcrit * se_log)
    hi = np.exp(np.log(are) + tcrit * se_log)
    return are, lo, hi


print("=" * 78)
print("Оценка показателя Ходжеса-Лемана rho (наклон ln(1-pi) по n)")
print("=" * 78)

res = {}
for name, p in power.items():
    for n_min in [1000, 4000]:
        r = estimate_rho(n, p, MC, n_min)
        res[(name, n_min)] = r
        print(f"{name:<4} n>={n_min:<5} точек={len(r['nn'])}: "
              f"rho = {r['rho']:.6f}  se = {r['se']:.6f}  "
              f"95% CI = [{r['ci_lo']:.6f}, {r['ci_hi']:.6f}]  sigma2 = {r['sigma2']:.1f}")

print()
print("=" * 78)
print("Относительная эффективность ARE = rho_CVM / rho_KS")
print("=" * 78)
for n_min in [1000, 4000]:
    rK, rC = res[("KS", n_min)], res[("CVM", n_min)]
    df = max(1, (len(rK["nn"]) - 2) + (len(rC["nn"]) - 2))
    are, lo, hi = are_with_ci(rC["rho"], rC["se"], rK["rho"], rK["se"], df)
    print(f"n>={n_min:<5}: ARE = {are:.3f}  95% CI = [{lo:.3f}, {hi:.3f}]")

fig, ax = plt.subplots(figsize=(9, 6))
colors = {"KS": "#1f77b4", "CVM": "#d62728"}
for name in ["KS", "CVM"]:
    p = power[name]
    ax.plot(n, np.log(1 - p), "o", color=colors[name], label=f"{name} (данные)",
            markersize=7, zorder=3)
    r = res[(name, 4000)]
    nn = r["nn"]
    yhat = r["beta"][0] + r["beta"][1] * nn
    ax.plot(nn, yhat, "-", color=colors[name], linewidth=2.5,
            label=f"{name} (подгонка, n≥4000, ρ = {r['rho']*1e4:.2f}·10⁻⁴)")

ax.set_xlabel("размер выборки n")
ax.set_ylabel("ln(1 − πₙ)")
ax.set_title("Экспоненциальный хвост: наклон = −ρ (Ходжес–Леман)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("phase0_fit.png", dpi=150, bbox_inches="tight")
print("\nsaved phase0_fit.png")
