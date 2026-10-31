#!/usr/bin/env python3
"""
interior.py - Interior experiment for the model sketch.

The domestic sector D uses F1 overwhelmingly but F2 a little (x1 >> x2 > 0).
The regime change raises the variance of F1's productivity (Sigma_11) only.

Outputs
  * formatted tables on screen
  * the same tables in interior_tables.md
  * simple charts in ./interior_figures/*.png

Requires fragmentation.py (the model engine) in the same folder.

Usage
  python interior.py          # tables + PNG charts
  python interior.py --show   # also open the charts on screen
"""
import os
import sys

import numpy as np

SHOW = "--show" in sys.argv
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import fragmentation as fr
except ImportError:
    sys.exit("interior.py needs fragmentation.py in the same folder.")

# =============================================================================
# Experiment settings
# =============================================================================
MU = np.array([0.0, fr.mu_1, -0.11])   # (mu_D, mu_F1, mu_F2): F2 used, but barely
SD_OLD, SD_NEW = 0.10, 0.30            # sd of F1 productivity before / after
SD_GRID = np.round(np.arange(0.10, 0.4001, 0.01), 2)   # sweep for the charts
SD_DEVIATION_CHECKS = [0.30, 0.34, 0.36, 0.40]
DELTA_L_GRID = [0.25, 0.60, 0.95]      # wage reset probabilities for robustness
IRF_QUARTERS = 12
TRANSITION_QUARTERS = 12               # quarters shown in the transition chart
FIG_DIR = "interior_figures"
MD_FILE = "interior_tables.md"

COL_INPUT = {"Labour": "tab:gray", "F1": "tab:red", "F2": "tab:blue"}
COL_SCEN = ["tab:gray", "tab:orange", "tab:purple", "tab:green"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})


# =============================================================================
# Formatting helpers
# =============================================================================
def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def num(x, d=4):
    return f"{x:.{d}f}"


def chg(new, base, d=1):
    return f"{100 * (new / base - 1):+.{d}f}%"


def sci(x):
    return f"{x:.1e}"


class Report:
    """Prints aligned tables and collects the same tables as Markdown."""

    def __init__(self, title):
        self.md = [f"# {title}", ""]
        print(f"\n{title}\n{'#' * len(title)}")

    def section(self, title):
        print(f"\n\n{title}\n{'=' * len(title)}")
        self.md += ["", f"## {title}", ""]

    def table(self, title, headers, rows, note=None, align=None):
        """align: one letter per column, 'l' or 'r' (default: first left, rest right)."""
        align = align or "l" + "r" * (len(headers) - 1)
        rows = [[str(c) for c in r] for r in rows]
        widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

        def line(cells):
            return "  ".join(c.ljust(widths[i]) if align[i] == "l" else c.rjust(widths[i])
                             for i, c in enumerate(cells))

        rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
        print(f"\n{title}\n{rule}\n{line(headers)}\n{rule}")
        for r in rows:
            print(line(r))
        print(rule)
        if note:
            print(f"Note: {note}")

        self.md += [f"**{title}**", "",
                    "| " + " | ".join(headers) + " |",
                    "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|"]
        self.md += ["| " + " | ".join(r) + " |" for r in rows]
        self.md += [""] + ([f"*Note: {note}*", ""] if note else [])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.md) + "\n")


# =============================================================================
# Computations
# =============================================================================
def frozen_f2_network(S, x2):
    """Best technique when F2's share is held at x2 (D can only reshore)."""
    cf = fr.closed_form(MU, S)
    x1 = (cf["b1"] - cf["M"][0, 1] * x2) / cf["M"][0, 0]
    return np.array([1 - x1 - x2, x1, x2])


def inflation_stats(x, S, shock_sd):
    """Phillips-curve objects, CPI-inflation moments and IRF for network x under Sigma S."""
    nk = fr.nk_objects(x)
    P, A, var, contrib = fr.cpi_moments(nk, S)
    shock = np.zeros(4)
    shock[2] = -shock_sd                                   # -1 s.d. shock to F1
    irf = fr.irf_cpi(P, A, shock, IRF_QUARTERS)
    return dict(nk=nk, P=P, var=var, contrib=contrib, irf=irf)


def compute():
    S_old, S_new = fr.Sigma(SD_OLD ** 2), fr.Sigma(SD_NEW ** 2)
    r = {"S_old": S_old, "S_new": S_new}

    # --- network block -------------------------------------------------------
    x_old = fr.solve_network(MU, S_old)[0]
    x_new = fr.solve_network(MU, S_new)[0]
    assert x_old[2] > 0, "calibration does not deliver an interior network"
    r.update(x_old=x_old, x_new=x_new,
             W_old=fr.W(x_old, MU, S_old), W_new=fr.W(x_new, MU, S_new))

    # --- analytical checks ---------------------------------------------------
    cf_new = fr.closed_form(MU, S_new)
    M = fr.closed_form(MU, S_old)["M"]
    Minv = np.linalg.inv(M)
    dS = SD_NEW ** 2 - SD_OLD ** 2
    sherman = x_old[1:] - fr.Gam * dS * x_old[1] * Minv[:, 0] / (1 + fr.Gam * dS * Minv[0, 0])
    eps = 1e-6
    x_eps = fr.solve_network(MU, fr.Sigma(SD_OLD ** 2 + eps))[0]
    one = np.ones(3)
    Hi = np.linalg.inv(-np.diag(fr.kap))
    Hhat_inv = Hi - np.outer(Hi @ one, one @ Hi) / (one @ Hi @ one)
    res_410 = [np.max(abs(x - (fr.x_ideal + Hhat_inv @ fr.R_vec(x, MU, S))))
               for x, S in [(x_old, S_old), (x_new, S_new)]]
    r["checks"] = [
        ("New F1 share: solver vs closed form", x_new[1], cf_new["interior"][0]),
        ("New F2 share: solver vs closed form", x_new[2], cf_new["interior"][1]),
        ("New F1 share: solver vs Sherman-Morrison", x_new[1], sherman[0]),
        ("New F2 share: solver vs Sherman-Morrison", x_new[2], sherman[1]),
        ("Share of F1's loss going to F2 vs M12/M22",
         (x_new[2] - x_old[2]) / (x_old[1] - x_new[1]), M[0, 1] / M[1, 1]),
        ("dx2/dSigma11: finite difference vs formula",
         (x_eps[2] - x_old[2]) / eps, fr.Gam * x_old[1] * M[0, 1] / np.linalg.det(M)),
        ("(4.10) fixed point, old regime (residual)", res_410[0], 0.0),
        ("(4.10) fixed point, new regime (residual)", res_410[1], 0.0),
    ]
    r["split"] = M[0, 1] / M[1, 1]

    # --- second-order condition (4.9) and unilateral deviations --------------
    r["soc_bound_sd"] = np.sqrt(fr.soc_bound_S11(S_old))
    r["deviations"] = []
    for sd in SD_DEVIATION_CHECKS:
        S = fr.Sigma(sd ** 2)
        x = fr.solve_network(MU, S)[0]
        arg, gain = fr.deviator_check(x, MU, S, n=201)
        r["deviations"].append((sd, np.linalg.eigvalsh(fr.soc_matrix(S)).min(), x, arg, gain))

    # --- Phillips-curve objects and inflation in four scenarios --------------
    x_frozen = frozen_f2_network(S_new, x_old[2])
    r["x_frozen"] = x_frozen
    r["scenarios"] = [
        ("Old regime", x_old, S_old, SD_OLD),
        ("New, network fixed", x_old, S_new, SD_NEW),
        ("New, F2 share frozen", x_frozen, S_new, SD_NEW),
        ("New, full adjustment", x_new, S_new, SD_NEW),
    ]
    r["stats"] = [inflation_stats(x, S, sd) for _, x, S, sd in r["scenarios"]]

    # --- one-off transition --------------------------------------------------
    def p_nat(Wv):                                           # natural prices, wage numeraire
        return np.array([0.0, -Wv[1], -MU[1], -MU[2]])
    chi0 = p_nat(r["W_old"]) - p_nat(r["W_new"])
    cpi, wage, p_end = fr.transition(r["stats"][3]["P"], chi0, H=400)
    r["transition"] = dict(cpi=cpi, wage=wage, dEz=r["W_new"][1] - r["W_old"][1],
                           closed=np.allclose(p_end - p_end[0], 0, atol=1e-9))

    # --- robustness to wage rigidity -----------------------------------------
    r["robust"] = []
    base_L = fr.calvo["L"]
    try:
        for dL in DELTA_L_GRID:
            fr.calvo["L"] = dL
            f1 = lambda x: 100 * np.sqrt(fr.cpi_moments(fr.nk_objects(x), S_new)[3]["F1"])
            tot = lambda x: 100 * np.sqrt(fr.cpi_moments(fr.nk_objects(x), S_new)[2])
            r["robust"].append(dict(dL=dL, vC=fr.nk_objects(x_old)["vC"][2],
                                    fixed=f1(x_old), frozen=f1(x_frozen), full=f1(x_new),
                                    tot_fixed=tot(x_old), tot_full=tot(x_new)))
    finally:
        fr.calvo["L"] = base_L

    # --- sweep over the size of the shock (for the charts) -------------------
    sw = {k: [] for k in ["Labour", "F1", "F2", "fixed", "frozen", "full"]}
    for sd in SD_GRID:
        S = fr.Sigma(sd ** 2)
        x = fr.solve_network(MU, S)[0]
        for k, v in zip(["Labour", "F1", "F2"], x):
            sw[k].append(100 * v)
        for k, xx in [("fixed", x_old), ("frozen", frozen_f2_network(S, x_old[2])), ("full", x)]:
            sw[k].append(100 * np.sqrt(fr.cpi_moments(fr.nk_objects(xx), S)[3]["F1"]))
    r["sweep"] = {k: np.array(v) for k, v in sw.items()}
    return r


# =============================================================================
# Tables
# =============================================================================
def write_tables(r, rep):
    x0, x1 = r["x_old"], r["x_new"]
    W0, W1 = r["W_old"], r["W_new"]

    rep.section("1. Calibration")
    rep.table("Parameters", ["Parameter", "Value", "Role"], [
        ["Risk aversion gamma", num(fr.gam, 2), "KMNT calibrated value"],
        ["Inverse Frisch phi", num(fr.phi, 2), ""],
        ["Risk weight Gamma, eq. (4.4)", num(fr.Gam, 3), "(gamma-1)(1+phi)/(gamma+phi)"],
        ["CES elasticity theta", num(fr.theta, 1), "enters only condition (4.9)"],
        ["Discount factor rho", num(fr.rho, 2), "quarterly"],
        ["TFP persistence eta", num(fr.eta, 2), "quarterly AR(1)"],
        ["Calvo reset: wages / prices", f"{fr.calvo['L']:.2f} / {fr.calvo['D']:.2f}",
         "M4: foreign = domestic prices"],
        ["Hat deltas: wages / prices",
         f"{fr.hat(fr.calvo['L']):.3f} / {fr.hat(fr.calvo['D']):.3f}", "eq. (5.13)"],
        ["Ideal shares (labour, F1, F2)", " / ".join(num(v, 2) for v in fr.x_ideal), "T2, interior"],
        ["Curvatures kappa (labour, F1, F2)", " / ".join(num(v, 2) for v in fr.kap),
         "separable TFP shifter"],
        ["Expected productivity mu (D, F1, F2)", " / ".join(num(v, 2) for v in MU), ""],
        ["sd of shocks: D / F2", "10% / 10%", "no cross-correlations"],
        ["sd of F1 shock: old -> new", f"{pct(SD_OLD, 0)} -> {pct(SD_NEW, 0)}", "variance x9"],
    ], align="lrl")

    rep.section("2. Network block")
    rep.table("Domestic sector's technique and aggregate TFP",
              ["Metric", "Old regime", "New regime", "Change"], [
        ["Labour share x0", pct(x0[0]), pct(x1[0]), f"{100*(x1[0]-x0[0]):+.1f} pp"],
        ["F1 share x1", pct(x0[1]), pct(x1[1]), f"{100*(x1[1]-x0[1]):+.1f} pp"],
        ["F2 share x2", pct(x0[2]), pct(x1[2]), f"{100*(x1[2]-x0[2]):+.1f} pp"],
        ["F1's share of foreign inputs", pct(x0[1]/(x0[1]+x0[2])), pct(x1[1]/(x1[1]+x1[2])), ""],
        ["Total foreign share s", pct(x0[1]+x0[2]), pct(x1[1]+x1[2]),
         f"{100*(x1[1]+x1[2]-x0[1]-x0[2]):+.1f} pp"],
        ["E[z] (log points x100)", num(100*W0[1], 3), num(100*W1[1], 3),
         f"{100*(W1[1]-W0[1]):+.3f}"],
        ["sd(z)", pct(np.sqrt(W0[2])), pct(np.sqrt(W1[2])), chg(np.sqrt(W1[2]), np.sqrt(W0[2]))],
        ["F1 contribution to V[z]", num(x0[1]**2*SD_OLD**2, 5), num(x1[1]**2*SD_NEW**2, 5), ""],
        ["F2 contribution to V[z]", num(x0[2]**2*0.01, 6), num(x1[2]**2*0.01, 6), ""],
        ["Planner objective W", num(W0[0], 5), num(W1[0], 5), ""],
    ], note=f"Each unit of share taken from F1 goes to F2 in the fixed fraction "
            f"M12/M22 = {pct(r['split'])}; the rest goes to labour.")

    rep.table("Analytical checks (solver vs formulas)",
              ["Check", "Solver", "Formula", "Abs. difference"],
              [[lab, num(a, 6), num(b, 6), sci(abs(a - b))] for lab, a, b in r["checks"]])

    rep.section("3. Second-order condition (4.9)")
    rep.table("Unilateral deviations by a single producer",
              ["sd of F1 shock", "Min eigenvalue (4.9)", "Equilibrium (x0, x1, x2)",
               "Best deviation", "Gain", "Verdict"],
              [[pct(sd, 0), f"{eig:+.4f}", " / ".join(num(v, 3) for v in x),
                " / ".join(num(v, 3) for v in arg), f"{gain:+.1e}",
                "equilibrium" if gain <= 0 else "deviation pays"]
               for sd, eig, x, arg, gain in r["deviations"]],
              note=f"(4.9) holds for sd of F1 shock below {pct(r['soc_bound_sd'])}. "
                   f"Gains are in log value; tiny negatives reflect the grid.",
              align="lrrrrl")

    rep.section("4. Phillips-curve objects")
    labels = [s[0] for s in r["scenarios"]]
    st = r["stats"]
    nk = [s["nk"] for s in st]
    xs = [s[1] for s in r["scenarios"]]
    slope = [n["kC"] * (fr.gam + fr.phi) for n in nk]
    rep.table("Slopes, cost-push exposures and index weights", ["Object"] + labels, [
        ["Foreign share s"] + [pct(x[1] + x[2]) for x in xs],
        ["Wage pass-through S"] + [num(n["S"], 4) for n in nk],
        ["CPI slope (gamma+phi) kappa^C"] + [num(v, 5) for v in slope],
        ["  change vs old"] + [chg(v, slope[0]) for v in slope],
        ["DC slope"] + [num(n["slopeDC"], 4) for n in nk],
        ["CPI exposure to D, v^C_D"] + [num(n["vC"][1], 5) for n in nk],
        ["CPI exposure to F1, v^C_F1"] + [num(n["vC"][2], 5) for n in nk],
        ["CPI exposure to F2, v^C_F2"] + [num(n["vC"][3], 5) for n in nk],
        ["DC index weight on F1"] + [pct(n["phiDC"][2]) for n in nk],
        ["DC index weight on F2"] + [pct(n["phiDC"][3]) for n in nk],
    ], note="Under M4 both slopes depend on the network only through the foreign share s.")

    rep.section("5. Inflation risk under G1")
    sd = lambda s, k: 100 * np.sqrt(s["contrib"][k])
    tot = [100 * np.sqrt(s["var"]) for s in st]
    rep.table("Standard deviation of quarterly CPI inflation (percentage points)",
              ["Component"] + labels, [
        ["Driven by F1 shocks"] + [num(sd(s, "F1"), 3) for s in st],
        ["Driven by F2 shocks"] + [num(sd(s, "F2"), 3) for s in st],
        ["Driven by D shocks"] + [num(sd(s, "D"), 3) for s in st],
        ["Total"] + [num(v, 3) for v in tot],
        ["F1 share of CPI variance"] + [pct(s["contrib"]["F1"] / s["var"]) for s in st],
        ["Impact of a -1 s.d. F1 shock (pp)"] + [num(100 * s["irf"][0], 3) for s in st],
    ])
    f1 = [sd(s, "F1") for s in st]
    rise = f1[1] - f1[0]
    rep.table("How much of the new F1-driven inflation risk the network absorbs",
              ["Scenario", "F1-driven sd", "vs network fixed", "Share of the rise absorbed"], [
        [labels[2], num(f1[2], 3), chg(f1[2], f1[1]), pct((f1[1] - f1[2]) / rise)],
        [labels[3], num(f1[3], 3), chg(f1[3], f1[1]), pct((f1[1] - f1[3]) / rise)],
        ["Contribution of diversification", "", f"{100*(f1[3]-f1[2])/f1[1]:+.1f} pp", ""],
    ], note="Benchmark: new shocks with the old network. Rise = fixed-network sd minus old-regime sd.")

    rep.section("6. One-off transition")
    t = r["transition"]
    rep.table("Adjustment to the new natural prices (network adopts immediately)",
              ["Metric", "Value"], [
        ["Change in E[z] = change in natural real wage", f"{100 * t['dEz']:+.3f}%"],
        ["Cumulative CPI inflation", f"{100 * t['cpi'].sum():+.3f}%"],
        ["Cumulative wage inflation", f"{100 * t['wage'].sum():+.3f}%"],
        ["Check: cumulative (CPI - wage) = -change in E[z]",
         f"{100 * (t['cpi'].sum() - t['wage'].sum()):+.3f}%"],
        ["CPI inflation in the first quarter", f"{100 * t['cpi'][0]:+.3f}%"],
        ["Relative-price gap closed", "yes" if t["closed"] else "NO"],
    ])

    rep.section("7. Robustness to wage rigidity")
    rep.table("F1-driven CPI inflation sd under the new shocks (pp)",
              ["Wage reset prob.", "v^C_F1 (old network)", "Network fixed",
               "F2 frozen", "Full adjustment", "Total CPI sd: full vs fixed"],
              [[num(d["dL"], 2), f"{d['vC']:+.5f}", num(d["fixed"], 3),
                f"{d['frozen']:.3f} ({chg(d['frozen'], d['fixed'])})",
                f"{d['full']:.3f} ({chg(d['full'], d['fixed'])})",
                chg(d["tot_full"], d["tot_fixed"])] for d in r["robust"]],
              note="v^C_F1 > 0 means a bad F1 draw is deflationary (relatively flexible wages).")


# =============================================================================
# Charts
# =============================================================================
def _shade_soc(ax, r):
    b = 100 * r["soc_bound_sd"]
    ax.axvspan(b, 100 * SD_GRID[-1], color="grey", alpha=0.12, label="(4.9) fails")
    for sd, ls, lab in [(SD_OLD, ":", "old regime"), (SD_NEW, "--", "new regime")]:
        ax.axvline(100 * sd, color="black", lw=0.8, ls=ls, label=f"{lab} ({pct(sd, 0)})")


def _save(fig, name):
    fig.tight_layout()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    return path


def make_charts(r):
    os.makedirs(FIG_DIR, exist_ok=True)
    x_axis = 100 * SD_GRID
    paths = []

    # 1. input shares vs size of the shock
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in ["Labour", "F1", "F2"]:
        ax.plot(x_axis, r["sweep"][k], color=COL_INPUT[k], lw=2, label=f"{k} share")
    _shade_soc(ax, r)
    ax.set(xlabel="sd of F1 productivity (%)", ylabel="Share of D's inputs (%)",
           title="Input shares as F1's uncertainty rises")
    ax.legend(frameon=False)
    paths.append(_save(fig, "1_input_shares.png"))

    # 2. F1-driven inflation risk vs size of the shock
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lab, c in [("fixed", "Network fixed", COL_SCEN[1]),
                      ("frozen", "F2 share frozen (reshoring only)", COL_SCEN[2]),
                      ("full", "Full adjustment", COL_SCEN[3])]:
        ax.plot(x_axis, r["sweep"][k], lw=2, color=c, label=lab)
    _shade_soc(ax, r)
    ax.set(xlabel="sd of F1 productivity (%)", ylabel="sd of quarterly CPI inflation (pp)",
           title="CPI-inflation risk from F1 shocks")
    ax.legend(frameon=False)
    paths.append(_save(fig, "2_inflation_risk_sweep.png"))

    # 3. four scenarios side by side
    labels = ["Old\nregime", "Network\nfixed", "F2 share\nfrozen", "Full\nadjustment"]
    st = r["stats"]
    slope = [s["nk"]["kC"] for s in st]
    panels = [
        ("CPI Phillips-curve slope, % change vs old", [100 * (v / slope[0] - 1) for v in slope], "{:+.1f}%"),
        ("F1-driven sd of CPI inflation (pp)", [100 * np.sqrt(s["contrib"]["F1"]) for s in st], "{:.2f}"),
        ("Total sd of CPI inflation (pp)", [100 * np.sqrt(s["var"]) for s in st], "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, vals, fmt) in zip(axes, panels):
        bars = ax.bar(range(4), vals, color=COL_SCEN)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=8, padding=2)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.grid(axis="x", visible=False)
    paths.append(_save(fig, "3_scenarios.png"))

    # 4. impulse responses to a -1 s.d. F1 shock
    fig, ax = plt.subplots(figsize=(7, 4))
    q = np.arange(IRF_QUARTERS)
    for (lab, *_), s, c in zip(r["scenarios"], st, COL_SCEN):
        if lab == "New, F2 share frozen":
            continue
        ax.plot(q, 100 * s["irf"], marker="o", ms=3, lw=1.8, color=c, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Quarters after the shock", ylabel="CPI inflation (pp)",
           title="CPI response to a -1 s.d. F1 productivity shock")
    ax.legend(frameon=False)
    paths.append(_save(fig, "4_irf_f1_shock.png"))

    # 5. one-off transition
    t = r["transition"]
    fig, ax = plt.subplots(figsize=(7, 4))
    q = np.arange(TRANSITION_QUARTERS)
    ax.bar(q - 0.2, 100 * t["cpi"][:TRANSITION_QUARTERS], width=0.4, color="tab:green", label="CPI inflation")
    ax.bar(q + 0.2, 100 * t["wage"][:TRANSITION_QUARTERS], width=0.4, color="tab:gray", label="Wage inflation")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Quarters after the regime change", ylabel="Quarterly inflation (%)",
           title="One-off transition to the new network")
    ax.set_xticks(q)
    ax.legend(frameon=False)
    paths.append(_save(fig, "5_transition.png"))

    # 6. robustness to wage rigidity
    fig, ax = plt.subplots(figsize=(7, 4))
    rb = r["robust"]
    xpos = np.arange(len(rb))
    frozen = [100 * (d["frozen"] / d["fixed"] - 1) for d in rb]
    full = [100 * (d["full"] / d["fixed"] - 1) for d in rb]
    b1 = ax.bar(xpos - 0.2, frozen, width=0.4, color=COL_SCEN[2], label="F2 frozen (reshoring only)")
    b2 = ax.bar(xpos + 0.2, full, width=0.4, color=COL_SCEN[3], label="Full adjustment")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%+.1f%%", fontsize=8, padding=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xpos, [f"wage reset prob. {d['dL']:.2f}" for d in rb])
    ax.set(ylabel="Change in F1-driven CPI sd vs fixed network (%)",
           title="Does the network dampen inflation risk?")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    paths.append(_save(fig, "6_robustness_wages.png"))

    return paths


# =============================================================================
def main():
    results = compute()
    rep = Report("Interior experiment: rise in F1's productivity variance with x1 >> x2 > 0")
    write_tables(results, rep)
    figs = make_charts(results)
    rep.section("Charts")
    rep.md += [f"![{os.path.basename(p)}]({p})" for p in figs]
    for p in figs:
        print(f"saved {p}")
    rep.save(MD_FILE)
    print(f"saved {MD_FILE}")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
