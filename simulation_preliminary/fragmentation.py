"""
Mono-sectoral domestic economy (D) + two foreign suppliers (F1, F2) in the
model_sketch.pdf framework. Sector order everywhere: 0 = union, 1 = D, 2 = F1, 3 = F2.
D chooses shares x = (x0, x1, x2) over (labour services, F1, F2) on the simplex, x >= 0.
Foreign sectors use only labour services (fixed technique). Household consumes only D.
"""
import itertools
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

np.set_printoptions(precision=5, suppress=True)

# ---------------------------------------------------------------- parameters
gam, phi, theta = 4.3, 1.0, 6.0                       # risk aversion (KMNT calib.), inverse Frisch, CES
Gam = (gam - 1) * (1 + phi) / (gam + phi)             # eq. (4.4)
rho, eta = 0.99, 0.9                                  # discount factor, AR(1) persistence (quarterly)
calvo = dict(L=0.25, D=1/3, F=1/3)                    # reset probs; M4: delta_F = delta_D
x_ideal = np.array([0.50, 0.25, 0.25])                # T2: interior ideal shares (labour, F1, F2)
kap = np.array([1.00, 0.35, 0.35])                    # separable quadratic TFP shifter curvatures
mu_D, mu_1 = 0.0, 0.05

def Sigma(s11, sDD=0.01, s22=0.01, s12=0.0, s1D=0.0, s2D=0.0):
    # order (D, F1, F2)
    return np.array([[sDD, s1D, s2D], [s1D, s11, s12], [s2D, s12, s22]])

def a_D(x):
    return -0.5 * np.sum(kap * (x - x_ideal) ** 2)

# ------------------------------------------------ network block (planner = decentralised FOC)
def W(x, mu, S):
    lam = np.array([1.0, x[1], x[2]])                 # Domar weights of (D, F1, F2)
    Ez = mu[0] + a_D(x) + x[1] * mu[1] + x[2] * mu[2]
    Vz = lam @ S @ lam
    return Ez - 0.5 * Gam * Vz, Ez, Vz

def R_vec(x, mu, S):
    # risk-adjusted prices of D's inputs (labour, F1, F2), eq. (4.8), wage numeraire
    lam = np.array([1.0, x[1], x[2]])
    cov = Gam * (S @ lam)                             # Cov[p_Fk, lambda_tilde] = Gam [Sigma lambda]_k
    return np.array([0.0, -mu[1] + cov[1], -mu[2] + cov[2]])

def grad_a(x):
    return -kap * (x - x_ideal)

def solve_network(mu, S):
    """Exact solution of the concave QP by active-set enumeration + KKT check."""
    best = None
    for zero in itertools.chain.from_iterable(itertools.combinations(range(3), r) for r in range(3)):
        free = [j for j in range(3) if j not in zero]
        # stationarity on free coords: grad a_j - R_j(x) = nu ; R linear in x -> linear system
        n = len(free)
        A = np.zeros((n + 1, n + 1)); b = np.zeros(n + 1)
        for r, j in enumerate(free):
            # grad a_j = -kap_j (x_j - xo_j);  R_j = -mu_j + Gam*(S lam)_j, lam=(1,x1,x2)
            A[r, r] += -kap[j]
            b[r] += kap[j] * x_ideal[j]
            if j > 0:
                b[r] += mu[j] - Gam * S[j, 0]
                for c, k in enumerate(free):
                    if k > 0:
                        A[r, c] -= Gam * S[j, k]
            A[r, n] = -1.0                              # - nu
        A[n, :n] = 1.0; b[n] = -1.0
        try:
            sol = np.linalg.solve(A, -b)
        except np.linalg.LinAlgError:
            continue
        x = np.zeros(3); x[free] = sol[:n]; nu = sol[n]
        if np.any(x < -1e-12):
            continue
        g = grad_a(x) - R_vec(x, mu, S)                 # net marginal value of each input
        psi = nu - g                                     # KKT multipliers (>=0 on zero shares)
        if np.all(psi[list(zero)] >= -1e-12) and np.allclose(g[free], nu, atol=1e-10):
            if best is None or W(x, mu, S)[0] > W(best[0], mu, S)[0] + 1e-14:
                best = (np.clip(x, 0, None), nu, psi)
    return best

def soc_matrix(S):
    # eq. (4.9) on the simplex tangent space, coordinates (v1, v2), v0 = -v1 - v2
    KM = kap[0] * np.ones((2, 2)) + np.diag(kap[1:])
    return KM - (theta - 1) * S[1:, 1:]               # must be positive definite

def deviator_logJ(xp, xs, mu, S):
    """log J(Omega') of eq. (4.5) up to a constant, others at the equilibrium xs."""
    # linear coefficients on eps = (eps_D, eps_1, eps_2)
    c = -Gam * np.array([1.0, xs[1], xs[2]]) + (1 - theta) * np.array([0.0, xs[1] - xp[1], xs[2] - xp[2]])
    mean = (1 - theta) * (-xp[1] * mu[1] - xp[2] * mu[2] - a_D(xp))
    return mean + 0.5 * c @ S @ c

def deviator_check(xs, mu, S, n=201):
    best, arg = -np.inf, None
    for i in range(n):
        for j in range(n - i):
            xp = np.array([1 - (i + j) / (n - 1), i / (n - 1), j / (n - 1)])
            v = deviator_logJ(xp, xs, mu, S)
            if v > best:
                best, arg = v, xp
    return arg, best - deviator_logJ(xs, xs, mu, S)

# ------------------------------------------------ closed forms (separable case, Sigma_1D=Sigma_2D=0 allowed)
def closed_form(mu, S):
    s0 = x_ideal[1] + x_ideal[2]
    b1 = kap[1] * x_ideal[1] + kap[0] * s0 + mu[1] - Gam * S[1, 0]
    b2 = kap[2] * x_ideal[2] + kap[0] * s0 + mu[2] - Gam * S[2, 0]
    M = kap[0] * np.ones((2, 2)) + np.diag(kap[1:]) + Gam * S[1:, 1:]
    corner_x1 = b1 / M[0, 0]
    G2 = b2 - M[0, 1] * corner_x1                      # >0  <=> F2 enters
    interior = np.linalg.solve(M, [b1, b2])
    return dict(b1=b1, b2=b2, M=M, corner_x1=corner_x1, G2=G2, interior=interior)

def threshold_S11(mu, S):
    cf = closed_form(mu, S)
    if cf["b2"] <= 0:
        return np.inf
    return ((kap[0] + Gam * S[1, 2]) * cf["b1"] / cf["b2"] - kap[0] - kap[1]) / Gam

def soc_bound_S11(S):
    k0, k1, k2 = kap
    t = theta - 1
    return (k0 + k1 - (k0 - t * S[1, 2]) ** 2 / (k0 + k2 - t * S[2, 2])) / t

# ------------------------------------------------ sticky-price block (Section 5)
def hat(d):
    return d * (1 - rho * (1 - d)) / (1 - rho * d * (1 - d))   # eq. (5.13)

def nk_objects(x):
    N = 4
    Om = np.zeros((N, N))
    Om[1, 0], Om[1, 2], Om[1, 3] = x                   # D buys labour, F1, F2
    Om[2, 0] = Om[3, 0] = 1.0                          # foreign sectors buy labour services (M4)
    I = np.eye(N); one = np.ones(N)
    alpha = np.eye(N)[0]; beta = np.eye(N)[1]
    L = np.linalg.inv(I - Om)
    lam = beta @ L
    d = np.array([calvo["L"], calvo["D"], calvo["F"], calvo["F"]])
    kvec = d * (1 - rho * (1 - d)) / (1 - d)           # eq. (5.4)
    K = np.diag(kvec); Dh = np.diag(kvec / (1 + kvec))
    assert np.allclose(np.diag(Dh), hat(d))
    # sketch route (5.11)
    Q = (I - Om) @ (I - np.outer(one, beta))
    Mi = np.linalg.inv(I + K @ Q)
    V = Mi @ K @ Q; B = Mi @ K @ alpha
    # R23 route (5.14)-(5.15)
    G = np.linalg.inv(I - Dh @ Om)
    dbeta = beta @ G @ Dh; dba = dbeta @ alpha
    B23 = G @ Dh @ alpha / (1 - dba)
    V23 = (G @ Dh - np.outer(B23, lam - dbeta)) @ (I - Om)
    assert np.allclose(V, V23) and np.allclose(B, B23), "sketch vs R23 mismatch"
    assert np.allclose(V @ one, 0), "V 1 != 0"
    vC = (dba * lam - dbeta) / (1 - dba)               # eq. (5.20)
    assert np.allclose(vC @ (I - Om), -beta @ V), "u^C formula mismatch"   # (5.18) with phi=beta
    wDC = lam * (1 - np.diag(Dh)) / np.diag(Dh); phiDC = wDC / wDC.sum()
    assert np.allclose(phiDC @ V, 0, atol=1e-12), "DC index not cost-push free"
    S_pass = x[0] + (x[1] + x[2]) * hat(calvo["F"])    # toy wage pass-through S
    assert np.isclose(dba, hat(calvo["D"]) * hat(calvo["L"]) * S_pass)
    return dict(Om=Om, L=L, lam=lam, V=V, B=B, dba=dba, kC=dba / (1 - dba), vC=vC,
                phiDC=phiDC, slopeDC=(gam + phi) / wDC.sum(), S=S_pass, Dh=Dh)

def solve_dynamics(V, L):
    """pi_t = Phi_p p_{t-1} + Phi_A logA_t solving (5.16) with (5.10)."""
    N = V.shape[0]; I = np.eye(N)
    P = -V.copy()
    for _ in range(20000):
        Pn = np.linalg.solve(I - rho * (I - V) @ P, rho * (I - V) @ P - V)
        if np.max(abs(Pn - P)) < 1e-14:
            break
        P = Pn
    res = P - (rho * (I - V) @ P @ (I + P) - V)
    assert np.max(abs(res)) < 1e-10, "Phi_p not solved"
    assert np.allclose(P @ np.ones(N), 0, atol=1e-10), "Phi_p 1 != 0"
    A = np.linalg.solve(I - rho * (I - V) @ (P + eta * I), -V @ L)
    return P, A

def cpi_moments(nk, S):
    P, A = solve_dynamics(nk["V"], nk["L"])
    N = 4; I = np.eye(N); one = np.ones(N); e0 = I[0]; beta = I[1]
    Proj = I - np.outer(one, e0)
    T = Proj @ (I + P); U = Proj @ A
    assert np.max(abs(np.linalg.eigvals(T))) < 1, "relative prices not stationary"
    F = np.block([[T, U], [np.zeros((N, N)), eta * I]])
    Spad = np.zeros((N, N)); Spad[1:, 1:] = S
    Qm = np.zeros((2 * N, 2 * N)); Qm[N:, N:] = (1 - eta ** 2) * Spad
    Pcov = solve_discrete_lyapunov(F, Qm)
    c = np.concatenate([P.T @ beta, A.T @ beta])
    var_pi = c @ Pcov @ c
    # contribution by shock source (shocks are independent in baseline)
    contrib = {}
    for k, name in zip([1, 2, 3], ["D", "F1", "F2"]):
        Sk = np.zeros((N, N)); Sk[k, k] = Spad[k, k]
        Qk = np.zeros((2 * N, 2 * N)); Qk[N:, N:] = (1 - eta ** 2) * Sk
        contrib[name] = c @ solve_discrete_lyapunov(F, Qk) @ c
    return P, A, var_pi, contrib

def irf_cpi(P, A, shock, H=12):
    N = 4; p = np.zeros(N); a = shock.copy(); out = []
    for _ in range(H):
        pi = P @ p + A @ a
        out.append(pi[1]); p = p + pi; a = eta * a
    return np.array(out)

def transition(P, chi0, H=400):
    p = chi0.copy(); cpi, wage = [], []
    for _ in range(H):
        pi = P @ p; p = p + pi
        cpi.append(pi[1]); wage.append(pi[0])
    return np.array(cpi), np.array(wage), p

# ------------------------------------------------ run
def report(label, mu, S, verbose=True):
    x, nu, psi = solve_network(mu, S)
    Wv, Ez, Vz = W(x, mu, S)
    cf = closed_form(mu, S)
    socm = soc_matrix(S); soc_ok = np.all(np.linalg.eigvalsh(socm) > 0)
    dev_arg, dev_gain = deviator_check(x, mu, S)
    nk = nk_objects(x)
    P, A, var_pi, contrib = cpi_moments(nk, S)
    out = dict(x=x, nu=nu, psi=psi, Ez=Ez, Vz=Vz, W=Wv, cf=cf, soc_ok=soc_ok,
               soc_eig=np.linalg.eigvalsh(socm), dev_arg=dev_arg, dev_gain=dev_gain, nk=nk,
               P=P, A=A, var_pi=var_pi, contrib=contrib)
    if verbose:
        print(f"\n=== {label} ===")
        print(" shares (labour, F1, F2):", x, "  foreign share s =", round(x[1] + x[2], 5))
        print(" KKT multipliers psi:", psi, "  nu:", round(nu, 5))
        print(" closed form: corner x1 =", round(cf['corner_x1'], 5), " G2 =", round(cf['G2'], 5),
              " interior =", cf['interior'], " b2 =", round(cf['b2'], 5))
        print(" E z =", round(Ez, 5), " V z =", round(Vz, 6), " W =", round(Wv, 6),
              " F1 part of Vz =", round(x[1] ** 2 * S[1, 1], 6))
        print(" SOC (4.9) eigenvalues:", nk and np.linalg.eigvalsh(socm), " ok:", soc_ok)
        print(" best unilateral deviation:", dev_arg, " gain:", f"{dev_gain:.2e}")
        print(" Domar lambda:", nk['lam'], " S =", round(nk['S'], 5), " delta^b alpha =", round(nk['dba'], 6))
        print(" CPI slope kappa^C (gamma+phi) =", round(nk['kC'] * (gam + phi), 6),
              "  DC slope =", round(nk['slopeDC'], 6))
        print(" v^C (union, D, F1, F2):", nk['vC'])
        print(" DC weights:", nk['phiDC'])
        print(" sd(pi^C) under G1 (quarterly, pp):", round(100 * np.sqrt(var_pi), 4),
              " variance shares:", {k: round(v / var_pi, 4) for k, v in contrib.items()})
    return out

if __name__ == "__main__":
    print(f"Gamma = {Gam:.4f}, theta-1 = {theta-1}, Gamma/(theta-1) = {Gam/(theta-1):.4f}")
    print("hat deltas (L, D=F):", hat(calvo['L']), hat(calvo['D']))
    for tag, mu2 in [("A", -0.13), ("B", -0.20)]:
        mu = np.array([mu_D, mu_1, mu2])
        S0, S1 = Sigma(0.01), Sigma(0.09)
        print("\n##########  CASE", tag, " mu_2 =", mu2,
              "  threshold Sigma11* =", round(threshold_S11(mu, S0), 5),
              " (sd", round(np.sqrt(threshold_S11(mu, S0)), 4), ")",
              "  SOC bound Sigma11 =", round(soc_bound_S11(S0), 5),
              " (sd", round(np.sqrt(soc_bound_S11(S0)), 4), ")")
        r0 = report(f"{tag} old regime (sd_1 = 10%)", mu, S0)
        r1 = report(f"{tag} new regime (sd_1 = 30%)", mu, S1)
        # transition: inherited prices at old natural levels, new network
        pnat = lambda r: np.array([0.0, -r['Ez'], -mu[1], -mu[2]])
        chi0 = pnat(r0) - pnat(r1)
        cpi, wage, pend = transition(r1['P'], chi0)
        print(" transition: Delta Ez =", round(r1['Ez'] - r0['Ez'], 6),
              " cum CPI infl =", round(cpi.sum(), 6), " cum wage infl =", round(wage.sum(), 6),
              " first-qtr CPI =", round(cpi[0], 6),
              " rel. gap closed:", np.allclose(pend - pend[0], 0, atol=1e-9))
        sh = np.zeros(4); sh[2] = -np.sqrt(0.01); print(" IRF CPI to -1sd F1 (old):", irf_cpi(r0['P'], r0['A'], sh, 6))
        sh[2] = -np.sqrt(0.09); print(" IRF CPI to -1sd F1 (new):", irf_cpi(r1['P'], r1['A'], sh, 6))
