# -*- coding: utf-8 -*-
"""
Last revised 30/08/2026

@author: Gabriele Bianchini @ UniPD

Shooting method for massless scalar QNMs on the quasi-de Sitter metric (case i).
See accompanying README.md for the physics and methodology.
"""

import mpmath as mp

# %% CONFIGURATION -- change k0 and ell here for a different case

mp.mp.dps = 15

Lam = mp.mpf(1)
k0 = mp.mpf(-5) / 3
ell = mp.mpf(2)


# %% METRIC AND MASTER-EQUATION COEFFICIENTS

def f(r): return 1 - Lam * r**2 / 3
def Nf(r): return 1 - 2 * Lam * r**2 / 3
def D(r): return 1 + k0 * r**2
def h(r): return f(r) * D(r) / Nf(r)

rN = mp.sqrt(mp.mpf(3) / (2 * Lam))
rc = mp.sqrt(3 / Lam)

def fp(r): return -2 * Lam * r / 3
def Dp(r): return 2 * k0 * r
def Np(r): return -4 * Lam * r / 3

def hp(r):
    return ((fp(r) * D(r) + f(r) * Dp(r)) * Nf(r) - f(r) * D(r) * Np(r)) / Nf(r)**2

def pp_prime(r):
    return (fp(r) * h(r) + f(r) * hp(r)) / 2

def P2(r): return h(r)
def P1(r): return pp_prime(r) / f(r)
def P0(r, omega): return omega**2 / f(r) - ell * (ell + 1) / r**2 - pp_prime(r) / (f(r) * r)

def V(r):
    return f(r) * ell * (ell + 1) / r**2 + (fp(r) * h(r) + f(r) * hp(r)) / (2 * r)

def p_func(r):
    return mp.sqrt(f(r) * h(r))


# %% RK4 INTEGRATOR

def F(r, y, omega):
    y1, y2 = y
    return (y2, -(P1(r) * y2 + P0(r, omega) * y1) / P2(r))

def rk4_step(r, y, sigma, omega):
    k1 = F(r, y, omega)
    y2_ = (y[0] + sigma / 2 * k1[0], y[1] + sigma / 2 * k1[1])
    k2 = F(r + sigma / 2, y2_, omega)
    y3_ = (y[0] + sigma / 2 * k2[0], y[1] + sigma / 2 * k2[1])
    k3 = F(r + sigma / 2, y3_, omega)
    y4_ = (y[0] + sigma * k3[0], y[1] + sigma * k3[1])
    k4 = F(r + sigma, y4_, omega)
    return (
        y[0] + sigma / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        y[1] + sigma / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    )

def integrate(r_start, y_start, r_end, N, omega):
    sigma = (r_end - r_start) / N
    r_cur, y_cur = r_start, y_start
    for _ in range(N):
        y_cur = rk4_step(r_cur, y_cur, sigma, omega)
        r_cur = r_cur + sigma
    return y_cur


# %% LOCAL FROBENIUS SERIES -- general in k0 and ell (see README for derivation)

# --- at r_N (Dirichlet branch, s = 3/2) ---
c1N = mp.sqrt(6) * (33 * k0 + 46) / (60 * (3 * k0 + 2))
c2N = (5283 * k0**2 + 6228 * k0 + 2764) / (560 * (9 * k0**2 + 12 * k0 + 4))

def R_rN(x):
    return x**mp.mpf('1.5') * (1 + c1N * x + c2N * x**2)

def R_rN_prime(x):
    P = 1 + c1N * x + c2N * x**2
    Pp = c1N + 2 * c2N * x
    return x**mp.mpf('0.5') * (mp.mpf('1.5') * P + x * Pp)

# --- at r_C (outgoing waves) ---
kappa = mp.sqrt((-Lam - 3 * k0) / 3)

def c1_rC(omega):
    num = (3 * omega * (Lam + 3 * k0) * (Lam + k0)
           - 1j * kappa * (4 * (Lam + 3 * k0)**2 + 9 * omega**2 * (Lam + k0)
                            + 2 * ell * (ell + 1) * Lam * (Lam + 3 * k0)))
    den = 12 * (Lam + 3 * k0)**2 * (1j * kappa - omega)
    return mp.sqrt(3) * mp.sqrt(Lam) / den * num

def c2_rC(omega):
    num = (4j * ell * (ell + 1) * (3 * k0 + 1) * (
               ell * (ell + 1) * kappa * (3 * k0 + 1)
               + kappa * (36 * k0**2 + 9 * k0 * omega**2 + 42 * k0 + 9 * omega**2 + 10)
               + 6j * omega * (3 * k0 + 1))
           + omega * (4212 * k0**4 - 3564j * k0**3 * kappa * omega + 1215 * k0**3 * omega**2 + 6048 * k0**3
                      + 81j * k0**2 * kappa * omega**3 - 3564j * k0**2 * kappa * omega + 1107 * k0**2 * omega**2 + 3024 * k0**2
                      + 162j * k0 * kappa * omega**3 - 792j * k0 * kappa * omega + 117 * k0 * omega**2 + 624 * k0
                      + 81j * kappa * omega**3 - 39 * omega**2 + 44))
    den = 96j * (3 * k0 + 1)**3 * (6 * k0 * kappa + 9j * k0 * omega + 3 * kappa * omega**2 + 2 * kappa + 3j * omega)
    return num / den

def R_rC(u, omega):
    s = 1j * omega / (2 * kappa)
    c1, c2 = c1_rC(omega), c2_rC(omega)
    return u**s * (1 + c1 * u + c2 * u**2)

def R_rC_prime(u, omega):
    s = 1j * omega / (2 * kappa)
    c1, c2 = c1_rC(omega), c2_rC(omega)
    return u**(s - 1) * (s + c1 * (s + 1) * u + c2 * (s + 2) * u**2)


# %% SCALE-INVARIANT WRONSKIAN AND MATCHING POINT

def make_DeltaL(eps, epsp, r_match, N):
    def DeltaL(omega):
        x0 = eps
        r0 = rN + x0
        y0 = (R_rN(x0), R_rN_prime(x0))
        y_in = integrate(r0, y0, r_match, N, omega)
        L_in = y_in[1] / y_in[0]

        u0 = epsp
        r0c = rc - u0
        y0c = (R_rC(u0, omega), R_rC_prime(u0, omega))
        y_out = integrate(r0c, y0c, r_match, N, omega)
        L_out = y_out[1] / y_out[0]

        return L_in - L_out
    return DeltaL

def r_match_balanced(omega_ref, eps, epsp):
    r_lo, r_hi = rN + eps, rc - epsp
    def integrand(r):
        a = mp.sqrt(V(r) - omega_ref**2).real
        return a / p_func(r)
    total = mp.quad(integrand, [r_lo, r_hi])
    target = total / 2
    def cum_minus_target(r_try):
        return mp.quad(integrand, [r_lo, r_try]) - target
    return mp.findroot(cum_minus_target, (r_lo + r_hi) / 2)


# %% SEARCH AND REFINEMENT TOOLS

def scan_grid(DL, re_vals, im_vals):
    """Print |DeltaL(omega)| over a grid. re_vals, im_vals: lists of floats."""
    header = 'Im\\Re  ' + '  '.join(f'{r:7.3f}' for r in re_vals)
    print(header)
    for im in im_vals:
        row = [abs(DL(mp.mpc(re, im))) for re in re_vals]
        print(f'{im:6.3f} ' + '  '.join(f'{float(v):7.3f}' for v in row))

def find_mode(DL, guess, tol='1e-8', maxsteps=20):
    try:
        root = mp.findroot(DL, guess, tol=mp.mpf(tol), maxsteps=maxsteps)
        return root, abs(DL(root))
    except Exception as e:
        print(f'  findroot failed from guess={complex(guess)}: {str(e)[:80]}')
        return None, None

def _aitken(x1, x2, x3):
    d1, d2 = x2 - x1, x3 - x2
    d2a = d2 - d1
    if abs(d2a) < 1e-30:
        return None
    return x1 - d1**2 / d2a

def refine_mode(omega_guess, eps=mp.mpf('0.01'),
                 epsp_steps=('0.0008', '0.0004', '0.0002', '0.0001'),
                 N_steps=(1500, 3000, 6000, 8000)):
    """
    Refine a rough mode candidate: recompute a balanced matching point, then
    run an epsilon'-convergence sequence and extrapolate Re and Im separately
    with Aitken's Delta^2 process. Returns (re, re_unc, im, im_unc, points).
    """
    r_match = r_match_balanced(omega_guess, eps, mp.mpf(epsp_steps[0]))
    print(f'r_match = {r_match}')

    points = []
    omega_ref = omega_guess
    for epsp_val, N in zip(epsp_steps, N_steps):
        DL = make_DeltaL(eps, mp.mpf(epsp_val), r_match, N)
        root, res = find_mode(DL, omega_ref)
        if root is None:
            print(f'  epsp={epsp_val}: failed, stopping with {len(points)} points')
            break
        print(f'  epsp={epsp_val}  N={N}  ->  omega = {complex(root)}   residual={float(res):.2e}')
        points.append(root)
        omega_ref = root

    if len(points) < 3:
        print('Not enough converged points to extrapolate.')
        return None

    re_seq = [p.real for p in points]
    im_seq = [p.imag for p in points]
    re_ests = [_aitken(*re_seq[i:i+3]) for i in range(len(points) - 2)]
    im_ests = [_aitken(*im_seq[i:i+3]) for i in range(len(points) - 2)]
    re_ests = [e for e in re_ests if e is not None]
    im_ests = [e for e in im_ests if e is not None]

    re_val, re_unc = (sum(re_ests[-2:]) / 2, abs(re_ests[-1] - re_ests[-2])) if len(re_ests) >= 2 \
        else (re_seq[-1], abs(re_seq[-1] - re_seq[-2]))
    im_val, im_unc = (sum(im_ests[-2:]) / 2, abs(im_ests[-1] - im_ests[-2])) if len(im_ests) >= 2 \
        else (im_seq[-1], abs(im_seq[-1] - im_seq[-2]))

    print(f'RESULT: ({re_val:.4f} +/- {re_unc:.4f}) + ({im_val:.4f} +/- {im_unc:.4f})i')
    return re_val, re_unc, im_val, im_unc, points


# %% SCAN -- run this to locate a mode candidate

omega_rif = mp.mpc(0, -1.24)  # rough reference, adjust based on what you're looking for
r_match_scan = r_match_balanced(omega_rif, mp.mpf('0.01'), mp.mpf('0.0005'))
DL_scan = make_DeltaL(mp.mpf('0.01'), mp.mpf('0.0005'), r_match_scan, 1200)

scan_grid(DL_scan, re_vals=[-0.3, -0.1, 0.0, 0.1, 0.3],
                    im_vals=[-0.9, -1.1, -1.3, -1.5])


# %% REFINE -- once you have a candidate from the scan above

result = refine_mode(mp.mpc(0, -1.24))