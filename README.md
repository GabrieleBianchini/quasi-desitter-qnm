# quasi-desitter-qnm

### Massless Scalar Quasinormal Modes on a Quasi-de Sitter Metric

This project computes the quasinormal mode (QNM) spectrum of a massless scalar field on a quasi-de Sitter metric, using a shooting method with arbitrary-precision arithmetic (`mpmath`).

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

At `r_N`, a Frobenius analysis gives two indicial roots, `s = 0` and `s = 3/2`. Neither diverges, so the choice of branch is fixed by a self-adjointness argument (Horowitz-Marolf/Wald): only `s = 3/2` gives a vanishing boundary term, so the physical condition is Dirichlet, `R -> 0`. At `r_C`, the physical condition is purely outgoing waves, with indicial exponent `s = +i*omega/(2*kappa)`, `kappa = sqrt((-Lambda-3*k0)/3)` (the surface gravity of the cosmological horizon).

### Local series

Both boundary conditions are implemented as local Frobenius series, not just the leading power:

```python
def R_rN(x):
    return x**mp.mpf('1.5') * (1 + c1N * x + c2N * x**2)

def R_rC(u, omega):
    s = 1j * omega / (2 * kappa)
    c1, c2 = c1_rC(omega), c2_rC(omega)
    return u**s * (1 + c1 * u + c2 * u**2)
```

`c1N`, `c2N` are general in `k0` and `ell` (they turn out to be independent of `ell` up to this order); `c1_rC`, `c2_rC` are general in `k0`, `ell` and `omega`. All four were derived symbolically and validated against the full equation, with residuals scaling as `O(x^3)` / `O(u^3)` as the series truncation order predicts.

### Integrator

A standard 4th-order Runge-Kutta integrator, run in `mpmath`'s arbitrary-precision arithmetic (not machine floats), carries each series solution from near its boundary to a common matching point:

```python
def rk4_step(r, y, sigma, omega):
    k1 = F(r, y, omega)
    y2_ = (y[0] + sigma / 2 * k1[0], y[1] + sigma / 2 * k1[1])
    k2 = F(r + sigma / 2, y2_, omega)
    ...
```

Arbitrary precision matters here: as the mode number `n` grows, the two independent solutions separate exponentially fast, and the sub-dominant branch is lost to floating-point rounding long before it would be lost analytically. Increasing `mp.mp.dps` (decimal digits of precision) and the number of integration steps `N` postpones this, at a real computational cost.

### Matching and root-finding

The two integrated solutions are compared at the matching point through their logarithmic derivative, `L_in - L_out`, rather than the raw Wronskian:

```python
def make_DeltaL(eps, epsp, r_match, N):
    def DeltaL(omega):
        ...
        return L_in - L_out
    return DeltaL
```

This is because the raw Wronskian's magnitude depends on the series normalization and decreases monotonically with `Im(omega)` regardless of the underlying physics, making it useless as a root-finding target. The logarithmic derivative is scale-invariant and only vanishes at genuine QNM frequencies.

The matching point itself is chosen to balance the two integration legs (`r_match_balanced`), based on the classical WKB turning-point criterion, so that neither leg has to cross an exponentially larger stretch of the domain than the other.

### Search and refinement

`scan_grid` prints `|DeltaL(omega)|` over a grid of trial frequencies, used to locate candidate modes by eye before refining them:

```python
def scan_grid(DL, re_vals, im_vals):
    for im in im_vals:
        row = [abs(DL(mp.mpc(re, im))) for re in re_vals]
        print(...)
```

`refine_mode` takes a rough candidate and pushes the boundary-layer parameter `epsilon'` (how close to `r_C` the local series is evaluated) toward zero in a fixed sequence, re-solving for the root at each step. Since this sequence converges geometrically but not exactly linearly, Re and Im are each extrapolated separately with Aitken's Delta-squared process, which infers the limit from the measured convergence ratio instead of assuming one:

```python
def _aitken(x1, x2, x3):
    d1, d2 = x2 - x1, x3 - x2
    d2a = d2 - d1
    return x1 - d1**2 / d2a
```

The spread between two independent Aitken estimates (using different but overlapping triples of points) is reported as the uncertainty on the final value.

---

## Usage

```bash
pip install -r requirements.txt
```

Edit the `CONFIGURATION` cell at the top of `qnm_shooting.py` to set `k0` and `ell`, then run the `SCAN` cell to locate a candidate and the `REFINE` cell to converge it with an uncertainty estimate. The file is organized in `# %%` cells (Spyder / VS Code style) so each step can be re-run independently without recomputing the setup.

---

## Known limitations

- Higher overtones (`n >= 2`) are numerically harder to converge with direct shooting; this is a known limitation of the method (as opposed to e.g. Leaver's continued-fraction approach), not specific to this metric.
- The matching-point and precision settings that work well for one mode are not guaranteed to work for another; some manual tuning of `epsp`, `N` and `mp.mp.dps` is generally needed when searching for new modes.
