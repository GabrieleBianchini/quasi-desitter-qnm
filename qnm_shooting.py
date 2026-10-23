# -*- coding: utf-8 -*-
"""
Last revised 23/10/2026

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

# --- at r_C (outgoing waves), series truncated at order M ---
kappa = mp.sqrt((-Lam - 3 * k0) / 3)
M = 60

# power-series helpers: a series is a list of M coefficients in u = rc - r
def s_poly(c):
    return [mp.mpf(c[i]) if i < len(c) else mp.mpf(0) for i in range(M)]

def s_mul(a, b):
    return [mp.fsum(a[i] * b[n - i] for i in range(n + 1)) for n in range(M)]

def s_div(a, b):
    q = []
    for n in range(M):
        q.append((a[n] - mp.fsum(q[i] * b[n - i] for i in range(n))) / b[0])
    return q

def s_der(a):
    return [(n + 1) * a[n + 1] for n in range(M - 1)] + [mp.mpf(0)]

# expansion in u of the coefficients of u^2 alpha R_uu + u beta R_u + gamma R = 0,
# with alpha = P2/u, beta = -P1, gamma = u P0 = omega^2 gamma_w + gamma_0
r_u = s_poly([rc, -1])
r2_u = s_mul(r_u, r_u)
f_u = [x / 3 for x in s_mul(s_poly([0, 1]), s_poly([2 * rc, -1]))]
D_u = [(1 if n == 0 else 0) + k0 * r2_u[n] for n in range(M)]
N_u = [(1 if n == 0 else 0) - 2 * Lam / 3 * r2_u[n] for n in range(M)]
Q_u = s_div(D_u, N_u)                                   # D/N
alpha_u = [x / 3 for x in s_mul(s_poly([2 * rc, -1]), Q_u)]
fr_u = s_poly([-2 * Lam * rc / 3, 2 * Lam / 3])        # df/dr
P1_u = [s_mul(fr_u, Q_u)[n] - s_mul(f_u, s_der(Q_u))[n] / 2 for n in range(M)]
beta_u = [-x for x in P1_u]
gamma_w = s_div(s_poly([3 / Lam]), s_poly([2 * rc, -1]))
gamma_0 = [-ell * (ell + 1) * s_div(s_poly([0, 1]), r2_u)[n]
           - s_div(s_mul(s_poly([0, 1]), P1_u), r_u)[n] for n in range(M)]

def a_rC(omega):
    """Coefficients a_n of R = u^s sum a_n u^n from the Frobenius recursion."""
    s = -1j * omega / (2 * kappa)
    gam = [omega**2 * gamma_w[n] + gamma_0[n] for n in range(M)]
    a = [mp.mpc(1)]
    for n in range(1, M):
        rhs = mp.fsum((alpha_u[k] * (n - k + s) * (n - k + s - 1) + beta_u[k] * (n - k + s) + gam[k]) * a[n - k]
                      for k in range(1, n + 1))
        a.append(-rhs / (alpha_u[0] * n * (n + 2 * s)))
    return s, a

def R_rC(u, omega):
    s, a = a_rC(omega)
    return u**s * mp.fsum(a[n] * u**n for n in range(M))

def R_rC_prime(u, omega):
    # derivative with respect to u; note dR/dr = -dR/du
    s, a = a_rC(omega)
    S = mp.fsum(a[n] * u**n for n in range(M))
    dS = mp.fsum(n * a[n] * u**(n - 1) for n in range(1, M))
    return s * u**(s - 1) * S + u**s * dS


# %% SCALE-INVARIANT WRONSKIAN

def make_DeltaL(eps, u0, r_match, N):
    def DeltaL(omega):
        x0 = eps
        r0 = rN + x0
        y0 = (R_rN(x0), R_rN_prime(x0))
        y_in = integrate(r0, y0, r_match, N, omega)
        L_in = y_in[1] / y_in[0]

        r0c = rc - u0
        y0c = (R_rC(u0, omega), -R_rC_prime(u0, omega))
        y_out = integrate(r0c, y0c, r_match, N, omega)
        L_out = y_out[1] / y_out[0]

        return L_in - L_out
    return DeltaL

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

def refine_mode(omega_guess, eps='1e-3', u0='0.1', N=1000):
    """
    Refine a mode candidate, then repeat the calculation changing the
    numerical parameters one at a time (eps, u0, N, M, r_match).
    The uncertainty is the largest change of Re and Im with respect to
    the reference value. Returns (re, re_unc, im, im_unc, points).
    """
    global M
    r_mid = (rN + rc) / 2
    DL = make_DeltaL(mp.mpf(eps), mp.mpf(u0), r_mid, N)
    ref, res = find_mode(DL, omega_guess)
    if ref is None:
        return None
    print(f'reference:  omega = {complex(ref)}   residual={float(res):.2e}')

    variations = [
        ('eps=1e-2',    dict(eps=mp.mpf('1e-2'))),
        ('eps=1e-4',    dict(eps=mp.mpf('1e-4'))),
        ('u0=0.15',     dict(u0=mp.mpf('0.15'))),
        ('N doubled',   dict(N=2 * N)),
        ('r_match=1.40', dict(r_match=mp.mpf('1.40'))),
        ('M=40',        dict(M=40)),
    ]
    points = [ref]
    for name, var in variations:
        M_old = M
        M = var.get('M', M)
        DLv = make_DeltaL(var.get('eps', mp.mpf(eps)), var.get('u0', mp.mpf(u0)),
                          var.get('r_match', r_mid), var.get('N', N))
        root, res = find_mode(DLv, ref)
        M = M_old
        if root is not None:
            print(f'  {name:13s} ->  omega = {complex(root)}')
            points.append(root)

    re_unc = max(abs(p.real - ref.real) for p in points)
    im_unc = max(abs(p.imag - ref.imag) for p in points)
    print(f'RESULT: ({ref.real:.6f} +/- {re_unc:.1e}) + ({ref.imag:.6f} +/- {im_unc:.1e})i')
    return ref.real, re_unc, ref.imag, im_unc, points


# %% SCAN -- run this to locate a mode candidate

DL_scan = make_DeltaL(mp.mpf('1e-3'), mp.mpf('0.1'), (rN + rc) / 2, 1000)

scan_grid(DL_scan, re_vals=[0.0, 0.1, 0.3],
                    im_vals=[-0.9, -1.0, -1.1, -1.2])


# %% REFINE -- once you have a candidate from the scan above

result = refine_mode(mp.mpc(0, -1.09))
