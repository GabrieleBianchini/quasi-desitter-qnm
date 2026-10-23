# quasi-desitter-qnm

### Massless Scalar Quasinormal Modes on a Quasi-de Sitter Metric

This project computes the quasinormal mode (QNM) spectrum of a massless scalar field on a quasi-de Sitter metric, using a shooting method with arbitrary-precision arithmetic (`mpmath`).

The calculation is part of:
- G. Bianchini, O. Luongo, *"Quasinormal modes of massless scalar perturbations on a quasi-de Sitter spacetime"* (in preparation).

The metric was introduced in:

- R. Giambò, O. Luongo, *"De Sitter-like Configurations with Asymptotic Quintessence Environment"*, Class. Quantum Grav. **41**, 125005 (2024). [doi:10.1088/1361-6382/ad43a9](https://doi.org/10.1088/1361-6382/ad43a9)

further explored in:

- D. Corona, R. Giambò, O. Luongo, *"Motion of test particles in quasi anti-de Sitter regular black holes"*, Int. J. Geom. Meth. Mod. Phys. **21**, 2440019 (2024). [arXiv:2402.18997](https://arxiv.org/abs/2402.18997)

with its thermodynamics studied in:

- O. Luongo, M. Azizinia, K. Boshkayev, *"Entropy of a Quasi-de Sitter Spacetime and the Role of Specific Heat"*, Entropy **28**, 43 (2026). [doi:10.3390/e28010043](https://www.mdpi.com/1099-4300/28/1/43)

The metric is

```
ds^2 = -f(r) dt^2 + dr^2/h(r) + r^2 dOmega_2^2
```

with `f = 1 - (Lambda/3) r^2`, `h = f*D/N`, `N = 1 - (2*Lambda/3) r^2`, `D = 1 + k0*r^2`. In the configuration studied here (`k0 <= -(2/3)*Lambda`), the domain is `r in (r_N, r_C)`, bounded by a genuine curvature singularity at `r_N` and a standard cosmological horizon at `r_C`.

---

## Code Description

### Master equation and potential

Starting from the massless Klein-Gordon equation and separating variables as `psi = e^{-i*omega*t} * Y_lm * R(r)/r`, the radial equation reduces to a Schrodinger-like form:

```
- d^2R/dr_*^2 + V(r) R = omega^2 R
```

with tortoise coordinate `dr_*/dr = 1/sqrt(f*h)`. In the code this is implemented directly in terms of `f`, `h` and their derivatives rather than the explicit closed form of `V(r)`:

```python
def P2(r): return h(r)
def P1(r): return pp_prime(r) / f(r)
def P0(r, omega): return omega**2 / f(r) - ell * (ell + 1) / r**2 - pp_prime(r) / (f(r) * r)
```

so that the master equation is written as `P2(r) R'' + P1(r) R' + P0(r,omega) R = 0`, equivalent to the Schrodinger form above after dividing by `P2`.

### Boundary conditions

At `r_N`, a Frobenius analysis gives two indicial roots, `s = 0` and `s = 3/2`. Neither diverges, so the choice of branch is fixed by a self-adjointness argument (Horowitz-Marolf/Wald): only `s = 3/2` gives a vanishing boundary term, so the physical condition is Dirichlet, `R -> 0`. At `r_C`, the physical condition is purely outgoing waves. Near the horizon `r_* ~ -ln(u)/(2*kappa)`, with `u = r_C - r`, so `e^{i*omega*r_*} = u^{-i*omega/(2*kappa)}` and the indicial exponent is `s = -i*omega/(2*kappa)`, `kappa = sqrt((-Lambda-3*k0)/3)` (the surface gravity of the cosmological horizon).

### Local series

Both boundary conditions are implemented as local Frobenius series, not just the leading power:

```python
def R_rN(x):
    return x**mp.mpf('1.5') * (1 + c1N * x + c2N * x**2)

def R_rC(u, omega):
    s, a = a_rC(omega)
    return u**s * mp.fsum(a[n] * u**n for n in range(M))
```

`c1N`, `c2N` are general in `k0` and `ell` (they turn out to be independent of `ell` up to this order). At `r_C` the coefficients `a_n` are obtained from the Frobenius recursion, after expanding the coefficients of the equation in powers of `u = r_C - r`, and the series is truncated at `M = 60`. The series converges for `u < r_C - r_N`. The residual near `r_N` scales as `O(x^3)`, as the truncation order predicts, and near `r_C` the results do not change between `M = 40` and `M = 80`.

### Integrator

A standard 4th-order Runge-Kutta integrator, run in `mpmath`'s arbitrary-precision arithmetic (not machine floats), carries each series solution from near its boundary to a common matching point. The leg from `r_C` starts at a finite distance `u0 = 0.1` from the horizon, where the series at order `M` is accurate to many more digits than needed. Starting much closer to the horizon with a low-order series would let the truncation error, which acts as a small ingoing wave, grow by a factor of order `(u_match/u0)^(|Im(omega)|/kappa)` before reaching the matching point.

```python
def rk4_step(r, y, sigma, omega):
    k1 = F(r, y, omega)
    y2_ = (y[0] + sigma / 2 * k1[0], y[1] + sigma / 2 * k1[1])
    k2 = F(r + sigma / 2, y2_, omega)
    ...
```

With the series at order `M` and a finite `u0`, the default precision (`mp.mp.dps = 15`) and `N = 1000` steps per leg are enough to reproduce all the modes of the paper to six decimal places.

### Matching and root-finding

The two integrated solutions are compared at the matching point through their logarithmic derivative, `L_in - L_out`, rather than the raw Wronskian:

```python
def make_DeltaL(eps, u0, r_match, N):
    def DeltaL(omega):
        ...
        return L_in - L_out
    return DeltaL
```

This is because the raw Wronskian's magnitude depends on the series normalization and decreases monotonically with `Im(omega)` regardless of the underlying physics, making it useless as a root-finding target. The logarithmic derivative is scale-invariant and only vanishes at genuine QNM frequencies.

The matching point is the midpoint `(r_N + r_C)/2`. The roots do not depend on this choice, which is checked during refinement.

### Search and refinement

`scan_grid` prints `|DeltaL(omega)|` over a grid of trial frequencies, used to locate candidate modes by eye before refining them:

```python
def scan_grid(DL, re_vals, im_vals):
    for im in im_vals:
        row = [abs(DL(mp.mpc(re, im))) for re in re_vals]
        print(...)
```

`refine_mode` takes a rough candidate, converges it, and then repeats the calculation changing one numerical parameter at a time: `eps`, `u0`, the number of steps `N`, the matching point and the truncation order `M`. The largest change of Re and Im with respect to the reference value is reported as the uncertainty. A full refinement takes a few minutes.

---

## Usage

```bash
pip install -r requirements.txt
```

Edit the `CONFIGURATION` cell at the top of `qnm_shooting.py` to set `k0` and `ell`, then run the `SCAN` cell to locate a candidate and the `REFINE` cell to converge it with an uncertainty estimate. The file is organized in `# %%` cells (Spyder / VS Code style) so each step can be re-run independently without recomputing the setup.

---

## Known limitations

- The series at `r_C` converges for `u < r_C - r_N`, so `u0` and the matching point must stay inside this radius.
- The Frobenius normalization `a_0 = 1` is singular at `omega = -i*n*kappa`, where the recursion denominator vanishes. Root-finding should not start exactly at these points.
- Modes have been computed up to `|Im(omega)| = 6`. Much more damped modes would need a higher `M` or a different method, such as Leaver's continued fraction.

---

## Changelog

- **v1.2** (October 2026). Series at `r_C` truncated at `M = 60` through the Frobenius recursion, integration started at finite `u0 = 0.1`, matching at the midpoint. The Aitken extrapolation is replaced by a direct check on the numerical parameters.
- **v1.1** (October 2026). Corrected outgoing boundary condition at `r_C`: `s = -i*omega/(2*kappa)`.
- **v1.0** (August 2026). First release.
