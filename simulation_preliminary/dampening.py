import numpy as np
import fragmentation as fr
from fragmentation import *

def sds(x, S):
    nk = nk_objects(x); P, A, var, c = cpi_moments(nk, S)
    return 100*np.sqrt(c['F1']), 100*np.sqrt(var), nk

muA = np.array([0, 0.05, -0.13])
S0, S1 = Sigma(0.01), Sigma(0.09)
x_old = solve_network(muA, S0)[0]
x_A = solve_network(muA, S1)[0]
x_noF2 = np.array([1 - closed_form(muA, S1)['corner_x1'], closed_form(muA, S1)['corner_x1'], 0.0])  # best F2-free technique
f_fix, t_fix, nk_fix = sds(x_old, S1)
f_nf2, t_nf2, nk_nf2 = sds(x_noF2, S1)
f_A,  t_A,  nk_A  = sds(x_A, S1)
f_old, t_old, nk_old = sds(x_old, S0)
print(f"F1-driven sd (pp/qtr): old {f_old:.3f} | new: fixed {f_fix:.3f}, reshoring only {f_nf2:.3f} ({100*(f_nf2/f_fix-1):.1f}%), with F2 {f_A:.3f} ({100*(f_A/f_fix-1):.1f}%)")
print(f"total CPI sd (pp/qtr): old {t_old:.3f} | new: fixed {t_fix:.3f}, reshoring only {t_nf2:.3f} ({100*(t_nf2/t_fix-1):.1f}%), with F2 {t_A:.3f} ({100*(t_A/t_fix-1):.1f}%)")
share_of_increase = (f_fix - f_A)/(f_fix - f_old)
print(f"share of the F1-driven sd increase absorbed: reshoring only {(f_fix-f_nf2)/(f_fix-f_old):.3f}, with F2 {share_of_increase:.3f}")

# decomposition of impact exposure v^C_F1 = -dD * x1 * f(S)
for lab, x, nk in [("reshoring only", x_noF2, nk_nf2), ("with F2", x_A, nk_A)]:
    r_share = x[1]/x_old[1]; r_tot = nk['vC'][2]/nk_fix['vC'][2]
    print(f"{lab}: v^C_F1 ratio {r_tot:.4f} = share ratio {r_share:.4f} x pass-through ratio {r_tot/r_share:.4f}")
print("F2 contribution to CPI var with F2:", round(cpi_moments(nk_A, S1)[3]['F2'], 8))

# maximal dampening inside (4.9), case A at sd1 = 34%
S34 = Sigma(0.34**2); x34 = solve_network(muA, S34)[0]
print(f"sd1=34%: F1-driven sd fixed {sds(x_old,S34)[0]:.3f} vs endogenous {sds(x34,S34)[0]:.3f} ({100*(sds(x34,S34)[0]/sds(x_old,S34)[0]-1):.1f}%)")

# robustness: wage rigidity (sign of v^C_F1 and direction of the pass-through channel)
for dL in [0.25, 1/3, 0.6, 0.95]:
    fr.calvo['L'] = dL
    a = sds(x_old, S1); b = sds(x_noF2, S1); c = sds(x_A, S1)
    print(f"delta_L={dL:.2f}: v^C_F1 old={nk_objects(x_old)['vC'][2]:+.5f}; F1-driven sd fixed {a[0]:.3f}, reshore {b[0]:.3f} ({100*(b[0]/a[0]-1):+.1f}%), with F2 {c[0]:.3f} ({100*(c[0]/a[0]-1):+.1f}%)")
fr.calvo['L'] = 0.25
