# Robust SNR Inference under Volatility Clustering and Non-Gaussian Heavy Tails
## Extensions via Subsampling and Self-Normalization

**Author:** Nihar Mahesh Jani — niharmaheshjani@gmail.com
**Status:** Independent research notebook (`code.ipynb`), single-file, fully reproducible
**Inspiration:** *Signal-to-Noise Ratio Inference under Volatility Clustering & Heavy Tails*,
López de Prado, Porcu, Zoonekynd & Engle (2026), SSRN: https://ssrn.com/abstract=6568702

---

## Abstract

The plug-in Sharpe ratio is the single most widely reported statistic in quantitative
finance, and it is also one of the most casually mis-inferred. López de Prado, Porcu,
Zoonekynd & Engle (2026) showed that under GARCH(1,1) volatility clustering with
heavy-tailed innovations, the asymptotic variance of the plug-in Sharpe ratio estimator
inflates in a way that depends on the persistence parameter φ = α + β and the fourth
standardised moment κ_z of the innovation distribution, and they derived a closed-form
correction (their Proposition 4.7) that restores correct coverage — provided the fourth
moment of returns is finite.

This repository takes that closed-form result as a starting point and asks four
questions a practitioner is bound to ask next. What happens if I do not trust my GARCH
specification (E1, HAC)? What happens if I want a fully nonparametric alternative (E2,
stationary bootstrap)? What happens if the fourth moment does not exist at all — the
regime of Theorem 5.3, which describes a great deal of real financial data, especially
in crypto and emerging markets (E3, subsampling)? And is there a method so simple that
it requires no bandwidth, no block-length rule, and no tail-index estimate, yet still
performs respectably across both regimes (E4, Ibragimov–Müller group inference)?

I answer these questions with a from-scratch implementation of all five inference
procedures, a 400-replication × 5-method × 3-DGP Monte Carlo study at T = 800, a
dedicated coverage-versus-sample-size study in the stable-law regime spanning
T ∈ {100, …, 2000}, and an empirical application to six liquid assets (SPY, QQQ, GLD,
TLT, BTC-USD, EEM) over 2019–2023. The central finding — documented honestly, including
where my own extensions fall short — is that the finite-fourth-moment / stable-law
distinction is not a technicality. It is the single decision that determines which of
the five confidence intervals a practitioner should trust.

---

## A Note on Where These Ideas Come From

Everything in this repository sits on the shoulders of the closed-form result in
Proposition 4.7 of López de Prado, Porcu, Zoonekynd & Engle (2026), and of the regime
characterisation in their Theorem 5.3. I did not derive either of those results — I
implemented them, verified them numerically, and then asked what a practitioner would
naturally want to do next. The four extensions (E1–E4) are my own work, conceived,
coded, and stress-tested by me, but the questions they answer were placed in front of me
by the paper itself. Where my simulations exposed a limitation in one of my own
extensions — most notably the stationary bootstrap's stalling coverage in the
stable-law regime, discussed in Section 9 — I have reported it exactly as I found it,
including the subsequent discussion with Enrique Porcu that helped me understand *why*
it happens. If any part of this repository helps a reader understand the original
paper better, I hope they will go on to read it directly; the questions it raises are
better than any of the answers I have given here.

This is **not** the official implementation of López de Prado, Porcu, Zoonekynd &
Engle (2026), and it is not endorsed by or affiliated with any of its authors.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Notation, Calibration, and Data-Generating Processes](#2-notation-calibration-and-data-generating-processes)
3. [Reproduction: Proposition 4.7 and the Paper Confidence Interval](#3-reproduction-proposition-47-and-the-paper-confidence-interval)
4. [Extension E1 — HAC-Based Delta-Method Inference](#4-extension-e1--hac-based-delta-method-inference)
5. [Extension E2 — Stationary Bootstrap Inference](#5-extension-e2--stationary-bootstrap-inference)
6. [Extension E3 — Subsampling Inference (Primary Recommendation)](#6-extension-e3--subsampling-inference-primary-recommendation)
7. [Extension E4 — Ibragimov–Müller Group Inference](#7-extension-e4--ibragimovmüller-group-inference)
8. [Monte Carlo Design and Results](#8-monte-carlo-design-and-results)
9. [The Stable-Law Stress Test: Coverage versus Sample Size](#9-the-stable-law-stress-test-coverage-versus-sample-size)
10. [Empirical Application: Six Real Assets, 2019–2023](#10-empirical-application-six-real-assets-20192023)
11. [The Regime-Separation Decision Rule](#11-the-regime-separation-decision-rule)
12. [Repository Structure](#12-repository-structure)
13. [Computational Implementation Notes](#13-computational-implementation-notes)
14. [Reproducing the Results](#14-reproducing-the-results)
15. [Honest Verdicts and Self-Assessment](#15-honest-verdicts-and-self-assessment)
16. [How to Cite](#16-how-to-cite)
17. [Acknowledgments](#17-acknowledgments)
18. [Questions, Corrections, and Feedback](#18-questions-corrections-and-feedback)

---

## 1. Motivation

Every practitioner who has ever reported a Sharpe ratio has, at some point, faced the
following uncomfortable question: how much of that number is signal, and how much is
noise accumulated over a finite sample? The classical answer — invoke a CLT, treat
returns as i.i.d., and report SR̂ ± 1.96/√T — is wrong in two ways simultaneously for
almost every real return series. First, daily returns cluster in volatility: a large
move today raises the probability of a large move tomorrow, which means the effective
number of independent observations is smaller than T. Second, daily returns are
heavy-tailed: extreme observations occur far more often than a Gaussian model would
predict, which inflates the variance of any sample moment that depends on R² or higher
powers.

López de Prado, Porcu, Zoonekynd & Engle (2026) addressed both problems at once for the
GARCH(1,1) case. Their Proposition 4.7 gives a closed-form expression for the asymptotic
variance V_sym of the plug-in Sharpe ratio estimator under conditional symmetry, as a
function of the GARCH parameters (α, β) and the fourth standardised moment κ_z of the
innovation distribution. Their Theorem 5.3 identifies the boundary of this result's
validity: when a diagnostic quantity I call DIAG (defined below) becomes non-positive,
the fourth moment of returns no longer exists, V_sym is no longer finite, and the
closed-form correction cannot be applied.

This repository is organised around a simple observation: DIAG > 0 versus DIAG ≤ 0 is
not a corner case. It is the fault line that separates "use the closed-form correction,
or something equally model-based like HAC" from "use a method that makes no assumption
about which moments exist at all." Sections 4–7 build exactly one method for each side
of that fault line, plus one model-free method (HAC) that is preferred when the
fault line has not yet been crossed but the practitioner does not trust the specific
GARCH specification. Sections 8–10 then measure, with Monte Carlo precision and on real
data, exactly how much this distinction matters in practice.

---

## 2. Notation, Calibration, and Data-Generating Processes

### 2.1 Calibration

All simulation experiments share a single baseline calibration, chosen to match the
parameter values reported in Table 1 of López de Prado, Porcu, Zoonekynd & Engle (2026)
for the MSCI World daily return series. Keeping these values unchanged is what makes
the reproduction in Section 3 directly comparable to the original paper.

| Symbol | Value | Interpretation |
|---|---|---|
| μ (`MU`) | 5 × 10⁻⁴ | Unconditional daily mean return |
| α (`ALPHA`) | 0.13 | GARCH(1,1) shock (ARCH) coefficient |
| β (`BETA`) | 0.81 | GARCH(1,1) persistence (GARCH) coefficient |
| φ = α + β (`PHI`) | 0.94 | Total volatility persistence — strongly clustered |
| σ² (`S2`) | 9 × 10⁻⁴ | Unconditional daily variance (σ ≈ 3% per day) |
| κ_z (`KZ`) | 6.0 | Fourth standardised moment of the innovation, ≡ Student-t with 6 d.f. |
| ω (`OMEGA`) | σ²(1 − φ) = 5.4 × 10⁻⁵ | GARCH intercept implied by stationarity |
| SR (`TRUE_SR`) | μ/σ ≈ 0.016667 | Population (daily) signal-to-noise ratio |

A single global random seed (`np.random.seed(42)`) is set once at the top of the
notebook so that every table and figure can be regenerated bit-for-bit from this script.

### 2.2 The Fourth-Moment Diagnostic, DIAG

The validity of the closed-form correction in Proposition 4.7 hinges on a single scalar
diagnostic:

```
DIAG = 1 − α²κ_z − 2αβ − β²
```

At the baseline calibration, DIAG = 1 − 0.13²·6 − 2·0.13·0.81 − 0.81² = **0.0319 > 0**,
so the fourth moment of returns is finite and Proposition 4.7 applies directly. The
notebook asserts this condition (`assert DIAG > 0`) immediately after computing it, so
that any future change to the calibration that crosses the fourth-moment boundary is
caught at the top of the script rather than silently producing NaNs downstream.

DIAG ≤ 0 is precisely the regime addressed by Theorem 5.3 of the paper: the fourth
moment of returns is infinite, V_sym is undefined, and any inferential procedure that
implicitly relies on a finite fourth moment — including, importantly, the HAC estimator
of E1 and the stationary bootstrap of E2 — loses its theoretical justification. This is
the regime that Extensions E3 and E4 are built for.

### 2.3 Three Data-Generating Processes

Three DGPs are implemented, each addressing a distinct question about how robust the
five inference procedures are to departures from the paper's exact setting.

**(a) GARCH(1,1) with Student-t(6) innovations — the correctly specified DGP.**
This is the world the paper's closed-form CI was built for. Returns follow

```
R_t = μ + y_t,     y_t = σ_t z_t,     σ_t² = ω + α y_{t-1}² + β σ_{t-1}²
```

with `z_t` drawn from a standardised Student-t distribution with 6 degrees of freedom
(matching κ_z = 6), clipped to ±7 standard deviations to prevent numerically pathological
draws from destabilising the recursion. The inner recursion is JIT-compiled with Numba
(`@njit(fastmath=True)`) and run with a 300-step burn-in, with the conditional variance
floored at 10⁻¹² and capped at 0.5 for numerical stability. Matching the paper's exact
calibration (α = 0.13, β = 0.81, df = 6) is what allows the coverage numbers reproduced
in Section 8 to be compared directly against the paper's own Table.

**(b) EGARCH(1,1) — Nelson (1991), intentionally misspecified.**
This DGP asks: what happens when the practitioner uses the right *idea* (Proposition
4.7's closed-form correction, or the model-free HAC alternative) but the wrong *model*?
The conditional log-variance follows

```
log h_t = 0.96 · log h_{t-1} + 0.21 · (|z_{t-1}| − √(2/π)) − 0.04 · z_{t-1}
```

with `z_t ~ N(0,1)`, the 0.21 coefficient governing the magnitude response, and the
−0.04 coefficient introducing a leverage (asymmetry) effect entirely absent from the
symmetric GARCH(1,1) the paper's closed form was derived for. Calibration is chosen so
that the unconditional variance remains close to σ² = 9 × 10⁻⁴, keeping the overall
return spread comparable across DGPs. The HAC extension (E1) is model-free within the
finite-fourth-moment class, so it is expected to partially recover here even though the
Paper CI's underlying GARCH(1,1) assumption no longer literally describes the data.

**(c) Stable-law DGP — Theorem 5.3, infinite fourth moment.**
This is the stress test, and the only DGP for which DIAG would be negative if the GARCH
formula were naively applied. Innovations are drawn from a symmetric α-stable
distribution with tail index α_stable = 1.7 via the Chambers–Mallows–Stuck (CMS, 1976)
parametrisation:

```
For α_stable ≠ 1:
  z = sin(α_s(U + Bπ/2α_s)) / cos(U)^{1/α_s}
      × [cos(U − α_s(U + Bπ/2α_s)) / E]^{(1−α_s)/α_s}
```

with `U ~ Uniform(−π/2, π/2)`, `E ~ Exponential(1)`, and `B = 0` for symmetry. Because
the strict variance of an α-stable variate with α_stable < 2 does not exist, the spread
of `z` is matched to the GARCH DGP via its **interquartile range** rather than its
variance — the target IQR is set to `2√σ² × 1.35`, using the fact that the IQR of a
standard Gaussian is approximately 1.35σ. This guarantees that all three DGPs have a
comparable "typical" scale of daily fluctuation, even though only the first two have a
finite variance in the usual sense. Only Extensions E3 (subsampling) and E4
(Ibragimov–Müller) are designed to remain valid in this regime; their performance here
is the central subject of Section 9.

---

## 3. Reproduction: Proposition 4.7 and the Paper Confidence Interval

### 3.1 The Plug-In Sharpe Ratio Estimator

The plug-in estimator (Equation (2) of the paper) is the familiar sample mean divided
by the sample standard deviation, with Bessel's correction:

```
SR̂ = R̄ / σ̂,     σ̂ = sqrt( (1/(T-1)) Σ (R_t − R̄)² )
```

This single function (`plug_in_sharpe`) is the common building block beneath every one
of the five inference procedures in this repository — all five methods are different
ways of attaching a confidence interval to this same point estimate.

### 3.2 The Closed-Form Asymptotic Variance V_sym

Under conditional symmetry — which causes the cross-moment ("skewness") channel Ω_yu of
the more general result to vanish — Proposition 4.7 gives:

```
V_sym = 1 + (μ²/4σ²) · [(1−β)²(κ_z−1)(1+φ)] / [(1−φ)·DIAG]
```

Evaluated at the baseline calibration (μ = 5×10⁻⁴, σ² = 9×10⁻⁴, α = 0.13, β = 0.81,
φ = 0.94, κ_z = 6, DIAG = 0.0319), this gives:

```
V_paper  = 1.0127
V_iid    = 1 + SR²/2 = 1.00014     (i.i.d. Gaussian benchmark, for comparison)
```

I read this contrast as the headline number of the entire reproduction exercise:
volatility clustering at φ = 0.94, with Student-t(6) innovations, inflates the
asymptotic variance of the plug-in Sharpe ratio by **(1.0127 / 1.00014) − 1 ≈ 1.26%**
relative to the i.i.d. Gaussian benchmark. That 1.26% may sound small, but it compounds:
every confidence interval built on the (incorrect) i.i.d. assumption is too narrow by a
factor that scales with √V, and — as the φ → 1 curve in Figure 1 (panel 3) shows —
this inflation accelerates sharply, and eventually diverges, as persistence approaches
the fourth-moment boundary.

The resulting Paper CI is then the textbook normal-approximation interval built around
this corrected variance:

```
Paper CI:  SR̂ ± z_{0.975} · √(V_paper / T)
```

`paper_closed_form_V` returns `NaN` whenever DIAG ≤ 0, which gives every downstream
function a clean, automatic way to detect that the fourth-moment regime has been left.

### 3.3 The V_sym Curve as φ → 1

Panel 3 of `result_image.png` plots V_sym (Proposition 4.7) against φ = α + β over the
grid φ ∈ [0.50, 0.97], holding the ratio α/β fixed at the baseline calibration's
0.13 : 0.81 split (i.e. α = 0.15φ, β = 0.85φ) and κ_z, μ, σ² fixed at their baseline
values. The curve is essentially flat and close to the i.i.d. benchmark for φ ≲ 0.85,
and then bends sharply upward as φ approaches the fourth-moment boundary implied by
DIAG = 0, eventually diverging. The calibrated value φ = 0.94 sits visibly on the
steep part of this curve (V_paper = 1.0127, marked with a dotted horizontal line),
which is the figure's way of making a single point vividly: **the same persistence
level that is entirely normal for daily equity index returns is already deep in the
regime where ignoring volatility clustering meaningfully understates Sharpe ratio
uncertainty** — and a modest further increase in persistence would push V_sym, and
hence the width of any honestly-computed confidence interval, sharply higher still.

---

## 4. Extension E1 — HAC-Based Delta-Method Inference

### 4.1 Motivation

Proposition 4.7's closed form is exact, but it requires the practitioner to plug in
α, β, and κ_z — quantities that in practice are themselves estimated, with error, from
a GARCH fit. E1 asks: can we get a valid confidence interval for SR that depends only on
weak stationarity and mixing conditions, without committing to a specific GARCH
specification at all? The answer is the classical heteroskedasticity-and-autocorrelation
-consistent (HAC) sandwich variance, combined with a delta-method linearisation of
SR = μ/σ.

### 4.2 Newey–West Long-Run Covariance

The long-run covariance between two mean-zero series `a` and `b` is estimated with the
Newey–West (1987) Bartlett-kernel estimator:

```
Ω̂(a,b) = γ̂_{ab}(0) + Σ_{h=1}^{M} (1 − h/(M+1)) · [γ̂_{ab}(h) + γ̂_{ab}(−h)]
```

The bandwidth M is chosen via the Andrews (1991) AR(1) plug-in rule,
`M = max(⌊4(n/100)^{2/9}⌋, 3)`. The implementation vectorises the lag sum with NumPy dot
products rather than a nested Python loop, which I found cuts runtime by roughly an
order of magnitude — a meaningful saving given that this routine is called inside every
one of the 400 × 3 = 1,200 Monte Carlo replications in Section 8.

### 4.3 The Delta-Method Variance of SR̂

Writing SR = μ/σ as a function of the first two moments (μ, m₂ = E[R²]), with
σ = √(m₂ − μ²), the gradient with respect to (μ, m₂) is

```
g₁ = ∂SR/∂μ  =  m₂ / σ³
g₂ = ∂SR/∂m₂ = −μ / (2σ³)
```

and the HAC-based asymptotic variance of √n(SR̂ − SR) is the corresponding quadratic
form in the long-run covariance matrix of (R − μ, R² − m₂):

```
V̂_HAC = g₁² Ω̂₁₁ + 2 g₁ g₂ Ω̂₁₂ + g₂² Ω̂₂₂
```

giving the confidence interval `SR̂ ± z_{0.975}·√(V̂_HAC / n)`.

### 4.4 Where E1 Is Expected to Break, and Where It Quietly Does Not

The theoretical justification for E1 requires Ω̂₁₁, Ω̂₁₂, Ω̂₂₂ — long-run covariances of
R and R² — to be estimating finite quantities. In the stable-law DGP of Section 2.2(c),
with α_stable = 1.7, the *variance* of R² does not exist in population, so Ω̂₂₂ does not
converge to anything in the usual sense, and the HAC sandwich loses its formal
justification entirely. What I find empirically (Section 8) is more nuanced than an
outright failure: at T = 800, HAC coverage in the stable-law DGP is 92.0%, only
modestly below its 92.2% and 92.5% coverage in the GARCH and EGARCH DGPs respectively.
The interval does not visibly *break*, in the sense of becoming wildly miscalibrated —
but its theoretical foundation has, and the coverage-versus-T study in Section 9 shows
that, unlike E3 and E4, HAC coverage in the stable-law regime does not reliably approach
95% as T grows; it remains in the low-90s across the entire range T ∈ {100, …, 2000}.
E1 is therefore my preferred method whenever DIAG > 0, and a usable-but-theoretically
ungrounded fallback when DIAG ≤ 0.

---

## 5. Extension E2 — Stationary Bootstrap Inference

### 5.1 Motivation and the Politis–White Block-Length Rule

After HAC, the natural next step is to drop the delta-method linearisation entirely and
resample the data itself, respecting its dependence structure. The stationary bootstrap
of Politis & Romano (JASA, 1994) resamples *blocks* of geometrically distributed random
length from a circularly-wrapped version of the series, which preserves short-range
dependence on average without requiring the practitioner to choose a fixed block length.

The mean block length b̄ is chosen automatically via the Politis–White (2004) /
Patton–Politis–White (2009) rule, which balances the bias and variance of the resulting
bootstrap distribution estimate:

```
b̄* = [2G² / D]^{1/3} · n^{1/3}

where  G = Σ_{k=1}^{M} k · γ̂(k)
       D = [2 Σ_{k=1}^{M} γ̂(k)]²
```

`γ̂(k)` is the sample autocovariance of the (demeaned) series at lag k, and M is a
flat-top bandwidth set to ⌊√n⌋ (capped at n/4). The resulting geometric parameter is
p* = 1/b̄*, clipped to [2, n/3] for numerical stability. If D is degenerate or G is
negative — both of which can occur for near-white-noise series — the rule falls back to
b̄ = n^{1/4}.

### 5.2 The Algorithm

For each of B = 999 bootstrap replications: (i) build a pseudo-series of length n by
concatenating blocks of geometric-random length p, with starting points drawn uniformly
from {0, …, n−1} on the circularly-wrapped series `R_circ = [R, R]`; (ii) compute the
plug-in SR̂* on the pseudo-series; (iii) form the pivot statistic
`√n(SR̂* − SR̂_n)`. The 95% CI inverts the empirical α/2 and 1−α/2 quantiles of these
B pivots:

```
lo = SR̂_n − q̂_{1−α/2} / √n
hi = SR̂_n − q̂_{α/2}   / √n
```

All B = 999 pseudo-series are assembled inside a single Numba-JIT kernel
(`_stationary_bootstrap_stats`) from pre-generated arrays of block start positions and
geometric block lengths — random-number generation happens once in vectorised NumPy,
and the kernel does nothing but assemble series and compute statistics, which is the
part that genuinely benefits from compilation.

### 5.3 The Stalling Phenomenon, Reported as Found

Here is the finding I am most careful to report exactly as I observed it, because it is
the one place in this project where an extension I built does not do what I initially
hoped. The stationary bootstrap, however carefully its block length is tuned, **implicitly
relies on the resampled series having a finite variance** — the pivot statistic
`√n(SR̂* − SR̂_n)` is itself a ratio of sample moments, and its bootstrap distribution
inherits whatever moment problems the original series has. In the stable-law DGP, this
manifests as a coverage that does not converge to 95% as T grows, but instead
*stalls* in the low-90s: 89.1% at T = 100, rising only to 93.0% by T = 2000 (full table
in Section 9). This is not finite-sample noise that more data would fix — the gap
persists across an order-of-magnitude range in T, which is exactly the signature of a
*conceptually* mis-specified variance estimator rather than a slowly-converging correct
one. A subsequent discussion with Enrique Porcu helped me see why: the bootstrap
distribution of a ratio statistic under infinite-variance resampling does not have the
same tail behaviour as the true sampling distribution of SR̂, no matter how the block
length is chosen. This single finding is the reason Extensions E3 and E4 exist, and it
is documented in detail — and visually, via the annotated arrow in panel 7 of
`result_image.png` — precisely so that a reader does not have to rediscover it the hard
way.

---

## 6. Extension E3 — Subsampling Inference (Primary Recommendation)

### 6.1 The Politis–Romano–Wolf Subsampling Principle

Subsampling (Politis, Romano & Wolf, 1999, Chapter 11) takes a different and, for this
problem, decisive approach: rather than resampling *with replacement* (which inherits
the moment structure of the data, as Section 5.3 shows), it computes the statistic of
interest on every contiguous subsample of length b < n, and uses the empirical
distribution of these `n − b + 1` overlapping subsample statistics — rescaled by √b — to
approximate the sampling distribution of `√n(SR̂_n − SR)` at rate `(b/n)^{1/2}`. Crucially,
this approximation **requires no assumption about which moments of R exist**; it relies
only on the existence of a non-degenerate limiting distribution for the rescaled
statistic, which holds under both the finite-fourth-moment Gaussian/Student-t world and
the stable-law world of Theorem 5.3.

The resulting CI has exactly the same pivotal form as the bootstrap CI:

```
lo = SR̂_n − q̂_{1−α/2} / √n
hi = SR̂_n − q̂_{α/2}   / √n
```

but the quantiles q̂ now come from the subsampling distribution of `√b(SR̂_{n,b,t} −
SR̂_n)` rather than from a resampling distribution.

### 6.2 Block Size and Implementation

Block size is set to `b = ⌊n^{1/3}⌋`, the standard cube-root rate for heavy-tailed means
(PRW 1999, Ch. 11): larger b introduces bias from the (b/n)^{1/2} approximation, smaller
b inflates the variance of the empirical quantile estimates, and n^{1/3} is the
rate-optimal compromise. The implementation builds the full `(n−b+1) × b` matrix of
overlapping subsamples as a **zero-copy NumPy stride-trick view** (`as_strided`), and
then computes all `n − b + 1` subsample means, variances, and Sharpe ratios in a single
vectorised pass — avoiding both an explicit Python loop over subsamples and any memory
duplication of the underlying array.

### 6.3 Why Subsampling Survives Where the Bootstrap Stalls

The contrast with Section 5.3 is the central empirical result of this repository.
Across T ∈ {100, …, 2000} in the stable-law DGP, subsampling coverage ranges from 94.8%
to 97.8% — never falling meaningfully below the 95% nominal level, and if anything
slightly *over*-covering at smaller T (97.8% at T = 200). The price for this robustness
is width: at T = 800 in the GARCH DGP, the subsampling CI is 14.2% wider than the Paper
CI (W_ratio = 1.1421), and in the stable-law DGP it is 11.7% wider (W_ratio = 1.1169).
I regard this as an entirely fair trade: a 12–14% wider interval that is *honestly*
calibrated is preferable to a narrower interval whose nominal coverage is a polite
fiction outside the regime it was designed for. This is the method I recommend as the
default choice for any practitioner who is not confident that DIAG > 0 for their data —
which, as Section 10 shows, includes some of the most heavily traded assets in the
world.

---

## 7. Extension E4 — Ibragimov–Müller Group Inference

### 7.1 Algorithm

The fourth method takes the opposite philosophy from subsampling: instead of using
*all* the data through overlapping windows, it partitions the series into q
**non-overlapping** contiguous blocks, computes the plug-in SR̂_j on each block
separately, and then runs an ordinary one-sample t-test on the q block-level estimates
(Ibragimov & Müller, JBES 2010):

```
1. Partition R into q contiguous blocks of size ⌊n/q⌋.
2. Compute SR̂_j for j = 1, …, q.
3. μ̂_q = mean(SR̂_j),   s_q = std(SR̂_j, ddof=1)
4. CI:  μ̂_q ± t_{q−1, 0.975} · (s_q / √q)
```

### 7.2 Why This Works, and Why q = 8

The justification (Theorem 1 of Ibragimov & Müller, 2010) rests on two conditions: each
block-level SR̂_j must be asymptotically normal *or* stable as the block length grows
(true under both the GARCH/Student-t and the α-stable DGPs of this repository), and
non-overlapping blocks from an α-mixing process must be asymptotically independent
(approximately true here, since each block spans many GARCH "memory" lengths at
φ = 0.94). The elegance of the method is that the **t-critical-value absorbs the unknown
long-run variance entirely** — no bandwidth, no block-length rule, and no estimate of
the stable index α_stable is required anywhere in the procedure. This is what
"self-normalised" means in practice, and it is why I use E4 as a robustness benchmark
that requires essentially no tuning.

The default q = 8 is a deliberate compromise: q = 4 gives `t_{3,0.975} = 3.18`,
producing uncomfortably wide intervals, while q = 16 shortens each block to the point
where the asymptotic-independence assumption becomes strained. Results are reassuringly
stable for q ∈ {4, 8, 16}; q = 8 is reported throughout.

### 7.3 Performance

E4 is the most conservative (widest) method in every single cell of the Monte Carlo
table in Section 8 — its width ratio relative to the Paper CI ranges from 1.1432
(EGARCH) to 1.1737 (GARCH) to 1.1703 (stable-law). What it buys for that width is
*consistency*: its coverage is 94.0%, 92.8%, and 94.0% across the three DGPs at T = 800
— the smallest spread of any method — and in the stable-law coverage-versus-T study
(Section 9) it is the only method besides subsampling that stays within roughly one
percentage point of 95% across the *entire* range T ∈ {100, …, 2000}, including at
T = 100 where every other method (HAC: 90.0%, SB: 89.1%) is visibly under-covering. I
use it as the robustness cross-check whenever subsampling's result looks surprising.

---

## 8. Monte Carlo Design and Results

### 8.1 Experimental Protocol

`run_full_experiment(T=800, n_reps=400)` is the central Monte Carlo study of this
repository. For each of the three DGPs (GARCH-correct, EGARCH-misspecified,
stable-law), and for each of 400 independent replications of length T = 800, all five
confidence intervals — Paper, HAC (E1), Stationary Bootstrap (E2), Subsampling (E3), and
Ibragimov–Müller (E4) — are computed on the *same* simulated series, and checked for
whether they cover the true population SR (`TRUE_SR ≈ 0.016667`). 400 replications give
coverage estimates accurate to roughly ±1 percentage point at 95% confidence — enough
to distinguish a genuinely miscalibrated method from sampling noise, without making the
experiment prohibitively slow.

One implementation detail matters for interpreting the table below: the Paper CI's
half-width is computed once via `_paper_se(T) = √(V_paper / T)` using the **true**
baseline calibration parameters (Section 2.1), and this same half-width — centred on
each replication's own SR̂ — is used in all three DGPs. This means the "Paper" row in
the EGARCH and stable-law columns answers the question *"if a practitioner happened to
know the true GARCH(1,1)+Student-t(6) parameters from Table 1, and naively applied them
regardless of which DGP actually generated the data, how would the resulting interval
perform?"* — which is a meaningful question, but a different one from "is Proposition
4.7 valid for this DGP." I return to this distinction in Section 15.

### 8.2 Full Results Table (T = 800, 400 replications)

| DGP | Method | Coverage (%) | Mean Width | Width / Paper |
|---|---|---:|---:|---:|
| **GARCH (correct)** | Paper (Prop 4.7) | 93.2 | 0.13947 | 1.0000 |
| | HAC (E1) | 92.2 | 0.13770 | 0.9873 |
| | Stationary Bootstrap (E2) | 92.5 | 0.13766 | 0.9871 |
| | Subsampling (E3) | 97.0 | 0.15929 | 1.1421 |
| | Ibragimov–Müller (E4) | 94.0 | 0.16370 | 1.1737 |
| **EGARCH (misspecified)** | Paper (Prop 4.7) | 93.0 | 0.13947 | 1.0000 |
| | HAC (E1) | 92.5 | 0.13712 | 0.9832 |
| | Stationary Bootstrap (E2) | 91.0 | 0.13679 | 0.9808 |
| | Subsampling (E3) | 95.8 | 0.16167 | 1.1592 |
| | Ibragimov–Müller (E4) | 92.8 | 0.15944 | 1.1432 |
| **Stable-law (Theorem 5.3)** | Paper (Prop 4.7) | 93.2 | 0.13947 | 1.0000 |
| | HAC (E1) | 92.0 | 0.13764 | 0.9869 |
| | Stationary Bootstrap (E2) | 91.2 | 0.13745 | 0.9855 |
| | Subsampling (E3) | 95.5 | 0.15577 | 1.1169 |
| | Ibragimov–Müller (E4) | 94.0 | 0.16322 | 1.1703 |

### 8.3 Interpretation

Three patterns are worth drawing out explicitly.

**HAC is narrower than Proposition 4.7 in every DGP, including the one the closed form
was built for.** In the GARCH-correct DGP, HAC's width ratio of 0.9873 means it is
1.27% narrower than the Paper CI while losing only one percentage point of coverage
(92.2% vs. 93.2%) — both comfortably close to nominal. I read this as encouraging
evidence that Proposition 4.7's closed form, while exact, is *conservative* in practice:
the model-free HAC alternative recovers nearly the same coverage with a tighter
interval, without requiring the practitioner to estimate α, β, or κ_z at all. This is
verdict **E1_HAC_narrower_than_paper_GARCH = TRUE** in `results_extended.json`.

**Subsampling and Ibragimov–Müller never fall below nominal coverage by more than a
fraction of a point, in any of the three DGPs — at the cost of 12–17% extra width.**
This is exactly the behaviour I designed E3 and E4 to have: a small, consistent width
penalty in exchange for coverage that does not degrade when the DGP changes underneath
the method.

**HAC and the stationary bootstrap both *also* look fine at T = 800 in the stable-law
DGP** — 92.0% and 91.2% respectively, only about a point below their GARCH-DGP values.
At a single sample size, this might look like reassuring evidence that the
finite-fourth-moment / stable-law distinction does not matter much in practice. Section
9 shows that this single-T snapshot is misleading: it is precisely at T = 800 that the
stationary bootstrap's stalling trajectory happens to be passing close to its eventual
asymptote. Looking at a single T, however large, is not sufficient to distinguish "valid
but currently a little under-covering" from "asymptotically biased by a fixed amount" —
which is exactly why Section 9 exists.

---

## 9. The Stable-Law Stress Test: Coverage versus Sample Size

### 9.1 Design

`stable_regime_coverage_vs_T` repeats the coverage check of Section 8, restricted to the
stable-law DGP (α_stable = 1.7) and to the four methods that do not require the
fourth-moment diagnostic to hold (HAC, Stationary Bootstrap, Subsampling, IM), across
T ∈ {100, 200, 350, 500, 750, 1000, 1500, 2000}, with **1,000 replications per T** —
giving Monte Carlo standard errors of roughly ±0.7 percentage points, tight enough to
distinguish a genuine plateau from a slowly converging sequence.

### 9.2 Results Table

| T | HAC (E1) | Stationary Bootstrap (E2) | Subsampling (E3) | Ibragimov–Müller (E4) |
|---:|---:|---:|---:|---:|
| 100 | 90.0% | 89.1% | 96.2% | 94.1% |
| 200 | 91.6% | 90.9% | 97.8% | 94.8% |
| 350 | 93.2% | 92.7% | 96.6% | 94.7% |
| 500 | 93.2% | 91.8% | 97.4% | 95.0% |
| 750 | 93.0% | 92.9% | 94.8% | 94.0% |
| 1000 | 93.4% | 91.8% | 96.0% | 94.7% |
| 1500 | 94.1% | 92.3% | 95.9% | 93.9% |
| 2000 | 93.1% | 93.0% | 94.9% | 94.1% |

### 9.3 Interpretation: The Stalling Pattern, Made Visible

This table is, in my view, the most important single result in this repository, and it
is reproduced visually as panel 7 of `result_image.png` with an explicit annotation at
the T = 500 point.

The **Stationary Bootstrap (E2)** column tells a clear story: 89.1% at T = 100, rising
through the low 90s, and reaching only 93.0% even at T = 2000 — a twenty-fold increase
in sample size buys roughly four percentage points of coverage, and the trajectory shows
no sign of approaching 95%. This is verdict **E2_SB_stalls_in_stable_regime = TRUE**
(formally checked as "coverage at T = 500 below 93%": 91.8% < 93%).

The **HAC (E1)** column shows a similar, slightly less severe pattern: it climbs from
90.0% at T = 100 to roughly 93–94% by T = 1000–1500, but the T = 2000 value (93.1%) is
*lower* than the T = 1500 value (94.1%) — consistent with a method oscillating around an
asymptote below 95% rather than converging to it. HAC's theoretical justification in
this regime is, as discussed in Section 4.4, formally absent; this table is the
empirical manifestation of that absence.

**Subsampling (E3)** and **Ibragimov–Müller (E4)**, by contrast, hover at or above 95%
across the *entire* range — Subsampling between 94.8% and 97.8%, IM between 93.9% and
95.0% — with no visible trend toward a sub-95% plateau. This is verdicts
**E3_Sub_better_than_SB_in_stable = TRUE** (Subsampling's 97.4% at T = 500 versus SB's
91.8%) and **E4_IM_robust_in_stable_regime = TRUE** (IM's 95.0% at T = 500, comfortably
above the 88% threshold used in the formal check).

The practical takeaway is blunt: **in the stable-law regime, more data does not fix the
stationary bootstrap, and is only a partial fix for HAC.** A practitioner working with
T = 2000 daily observations — roughly eight years of data — using the stationary
bootstrap on a heavy-tailed asset is still building intervals that miss the true
Sharpe ratio about 7% of the time at a nominal 5% rate, almost 40% more often than
advertised. Subsampling and IM group inference do not have this problem at any sample
size tested.

---

## 10. Empirical Application: Six Real Assets, 2019–2023

### 10.1 Data, Regime Detection, and Method

Section 12 of the notebook applies all five CI methods to daily log-returns of six
assets — chosen to span the asset-class spectrum from "textbook GARCH" to "extreme
tails" — downloaded via `yfinance` over **2019-01-01 to 2023-12-31**:

| Ticker | Description | n (obs.) |
|---|---|---:|
| SPY | S&P 500 ETF — large-cap US equity | 1,257 |
| QQQ | Nasdaq-100 ETF — tech-heavy, heavier tails | 1,257 |
| GLD | SPDR Gold Shares — commodity | 1,257 |
| TLT | 20+ Year Treasury ETF — rate-sensitive | 1,257 |
| BTC-USD | Bitcoin — crypto, extreme tails | 1,824 |
| EEM | Emerging Markets ETF — EM risk, skew | 1,257 |

For each asset, a GARCH(1,1) model with Student-t innovations is fit via the `arch`
package (`vol="GARCH", p=1, q=1, dist="t"`), yielding estimates of α, β, and the
Student-t degrees of freedom ν, from which κ_z = 3 + 6/(ν−4) for ν > 4 (and κ_z = ∞
for ν ≤ 4). The fourth-moment diagnostic DIAG = 1 − α²κ_z − 2αβ − β² is then computed
from these *fitted* parameters, and the regime flag — `finite-4th-moment` if DIAG > 0,
`stable-law` otherwise — determines whether the Paper CI (Proposition 4.7) can be
constructed at all. Annualisation uses the standard √252 scaling for both the Sharpe
ratio point estimate and all confidence interval bounds.

### 10.2 Per-Asset Results

| Asset | SR_ann | Regime | α | β | φ | κ_z | ν (df) | Paper CI (ann.) | HAC CI (ann.) | SB CI (ann.) | Sub CI (ann.) | IM CI (ann.) |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| **SPY** | 0.688 | stable-law | 0.179 | 0.808 | 0.987 | 4.86 | 7.22 | N/A | [−0.159, 1.534] | [−0.232, 1.465] | [−0.393, 1.513] | [−0.094, 2.254] |
| **QQQ** | 0.790 | stable-law | 0.144 | 0.847 | 0.992 | 4.42 | 8.24 | N/A | [−0.013, 1.594] | [−0.094, 1.564] | [−0.310, 1.584] | [−0.077, 2.112] |
| **GLD** | 0.610 | finite-4th-moment | 0.055 | 0.916 | 0.972 | 7.90 | 5.22 | [−0.283, 1.503] | [−0.244, 1.464] | [−0.306, 1.468] | [−0.382, 1.575] | [−0.379, 1.633] |
| **TLT** | −0.114 | finite-4th-moment | 0.102 | 0.875 | 0.977 | 3.29 | 24.80 | [−0.993, 0.764] | [−0.896, 0.668] | [−0.859, 0.811] | [−1.017, 0.744] | [−1.274, 1.243] |
| **BTC-USD** | 0.588 | stable-law | 0.069 | 0.931 | 1.000 | ∞ | 2.95 | N/A | [−0.162, 1.338] | [−0.276, 1.426] | [−0.233, 1.243] | [−0.330, 1.780] |
| **EEM** | 0.125 | finite-4th-moment | 0.101 | 0.847 | 0.949 | 4.11 | 9.39 | [−0.753, 1.003] | [−0.691, 0.941] | [−0.728, 0.873] | [−0.889, 1.034] | [−1.009, 1.446] |

### 10.3 Reading the Empirical Figure and Its Implications

`empirical_figure.png` displays these six results as horizontal interval plots, one per
asset, annotated with the point estimate (white diamond), the GARCH-fit parameters, and
a regime badge (teal "Finite-4th-moment" or red "⚠ Stable-law"). Three observations
stand out, and I think all three are genuinely useful for a practitioner, beyond their
role in validating the simulation results of Sections 8–9.

**The regime split is not academic — it falls exactly where Theorem 5.3 says it should.**
SPY, QQQ, and BTC-USD are all flagged `stable-law` by the GARCH-fit diagnostic, which
means **Proposition 4.7's Paper CI is simply not constructible for three of the six
assets in this study** — including the S&P 500 itself. For SPY, the fitted κ_z = 4.86
combined with α = 0.179, β = 0.808 gives DIAG = −0.0985 < 0. For BTC-USD, the fitted
Student-t degrees of freedom (ν = 2.95) is below 4, which makes κ_z formally infinite
and forces the stable-law flag regardless of α and β. This is exactly the situation
Section 1 motivates in the abstract: the closed-form correction, however elegant, is
unavailable for a meaningful share of liquid, widely-held assets, which is precisely why
E3 and E4 exist.

**For the three assets where the Paper CI *is* available (GLD, TLT, EEM), HAC is
consistently the narrowest interval, and IM is consistently the widest** — exactly
reproducing the ordering from the Monte Carlo study in Section 8.2. For GLD, HAC's
width (1.708) is the tightest of the five, while IM's width (2.012) is the widest — a
ratio of about 1.18, very close to the 1.17 ratio observed in the GARCH Monte Carlo DGP.

**Perhaps the most striking finding in the entire empirical section: every single
confidence interval, for every asset, for every method, contains zero.** Point estimates
of annualised Sharpe ratios range from a respectable 0.79 (QQQ) down to a mildly
negative −0.11 (TLT) — numbers that, reported alone, would suggest meaningfully
different risk-adjusted performance across these six assets over 2019–2023. Yet at a
95% confidence level, **none of these point estimates are statistically distinguishable
from zero**, by any of the five methods, including the narrowest (HAC). This is not a
flaw in any of the methods — it is an honest statement about how much information five
years of daily returns actually contains about the long-run Sharpe ratio, once
volatility clustering and heavy tails are properly accounted for. I consider this the
single most important practical message of the entire repository: *the headline number
on a Sharpe-ratio tear sheet is usually far less informative than its precision implies,
and the gap between "looks impressive" and "is statistically supported" is exactly what
the methods in this repository are built to quantify.*

---

## 11. The Regime-Separation Decision Rule

Bringing Sections 3–10 together, the practical decision rule this repository arrives at
— and which is reproduced as the text panel (panel 9) of `result_image.png` — is the
following.

**If DIAG > 0 (finite fourth moment — most equity, commodity, and rates data over short
horizons):**
- **Paper CI (Proposition 4.7):** valid; use directly if α, β, κ_z are trusted.
- **HAC (E1):** valid, model-free, and empirically *narrower* than the Paper CI —
  my preferred default in this regime.
- **Subsampling (E3) and IM (E4):** valid, but wider; use as robustness checks.
- **Stationary Bootstrap (E2):** valid and of moderate width; a reasonable nonparametric
  alternative when HAC's delta-method linearisation feels too restrictive.

**If DIAG ≤ 0 (infinite fourth moment — Theorem 5.3's regime, which the empirical
section shows includes SPY, QQQ, and BTC-USD over 2019–2023):**
- **Paper CI:** not constructible — `paper_closed_form_V` returns NaN by design.
- **HAC (E1):** loses its theoretical justification; numerically usable but its
  coverage-versus-T trajectory (Section 9) does not reliably approach 95%.
- **Stationary Bootstrap (E2):** stalls near 92% coverage regardless of sample size —
  not recommended.
- **Subsampling (E3):** my **primary recommendation**. Valid by construction, with a
  width penalty of roughly 12–17% relative to the (unavailable) Paper CI scale.
- **Ibragimov–Müller (E4):** robustness benchmark, requiring no tuning beyond q, and
  empirically the most stable across sample sizes of any method tested.

---

## 12. Repository Structure

```
.
├── code.ipynb               # The entire project — run top to bottom
├── README.md                # This file
├── result_image.png          # Figure 1: 9-panel reproduction + extension results
├── empirical_figure.png      # Figure 2: 3×2 panel, per-asset empirical CIs
├── results_extended.json     # Output: Monte Carlo tables (Sections 8–9) + verdicts
└── empirical_results.json    # Output: per-asset empirical results (Section 10)
```

`result_image.png` and `empirical_figure.png` are the two figures generated by
`make_extended_figure` and `make_empirical_figure` respectively; `results_extended.json`
and `empirical_results.json` are the corresponding machine-readable outputs, including
the four formal verdicts (`E1`–`E4`) discussed in Section 15.

---

## 13. Computational Implementation Notes

Three implementation choices are worth flagging for anyone extending this code.

**Numba JIT compilation** is used for the three loops that are called thousands of
times across the Monte Carlo study: the GARCH(1,1) recursion (`_garch11_kernel`), the
EGARCH(1,1) recursion (`_egarch_kernel`), and the stationary bootstrap's pseudo-series
assembly (`_stationary_bootstrap_stats`). All three are warmed up with dummy inputs at
the start of the `__main__` block specifically so that the one-time JIT compilation cost
is paid before timing begins, and does not contaminate the reported runtimes.

**Stride-trick vectorisation** is used in the subsampling CI (Section 6.2) to construct
the `(n−b+1) × b` matrix of overlapping subsamples as a zero-copy view via
`numpy.lib.stride_tricks.as_strided`, avoiding both an explicit loop over `n − b + 1 ≈
n` subsamples and any memory duplication of the underlying array — essential given that
this routine runs inside every one of the 1,200 + 8,000 = 9,200 Monte Carlo replications
across Sections 8 and 9.

**Numerical safeguards** are applied throughout: GARCH conditional variances are floored
at 10⁻¹² and capped at 0.5; Student-t innovations are clipped to ±7 standard deviations;
EGARCH log-variances are clipped to [−12, 4] and standardised shocks to [−5, 5]; and
every Sharpe-ratio denominator uses `max(σ̂, 1e-12)` to prevent division-by-zero on
degenerate samples. None of these safeguards bind under the baseline calibration — they
exist purely to prevent rare numerical excursions during the stable-law DGP's
heavy-tailed draws from propagating into NaNs.

---

## 14. Reproducing the Results

The entire project runs as a single Jupyter notebook, top to bottom, with no external
configuration.

**Requirements:**
```
numpy
scipy
pandas
matplotlib
numba
yfinance
arch
```
installed via:
```bash
pip install numpy scipy pandas matplotlib numba yfinance arch
```

**To run:**
1. Clone or download this repository.
2. Open `code.ipynb` in Jupyter (or VS Code with the Jupyter extension).
3. Run all cells top to bottom.

Running the notebook end to end will: (i) print the reproduced paper constants
(V_paper, V_iid, DIAG) for verification against Table 1 of the original paper; (ii) run
the full 400-replication × 5-method × 3-DGP Monte Carlo study (Section 8); (iii) run the
1,000-replication coverage-versus-T study in the stable-law regime (Section 9); (iv)
save `results_extended.json` and regenerate `result_image.png`; (v) download five years
of daily data for the six assets in Section 10 via `yfinance` (requires an internet
connection); and (vi) save `empirical_results.json` and regenerate `empirical_figure.png`.

A single global seed (`np.random.seed(42)`, set once at the top of the notebook) makes
every number in `results_extended.json` and every panel of `result_image.png`
reproducible exactly. The empirical figures in Section 10 will vary slightly with the
data vintage returned by `yfinance` at the time of execution, since the 2019–2023 window
is fixed but the underlying price history is subject to routine provider-side revisions
(e.g. dividend adjustments).

---

## 15. Honest Verdicts and Self-Assessment

`results_extended.json` records four formal verdicts, each corresponding to a claim made
in this README, and each currently evaluating to **TRUE**:

| Verdict | Claim | Check | Result |
|---|---|---|---|
| `E1_HAC_narrower_than_paper_GARCH` | HAC (E1) is narrower than the Paper CI in the GARCH-correct DGP | W_ratio(HAC, GARCH) < 1.0 | **TRUE** (0.9873) |
| `E2_SB_stalls_in_stable_regime` | The stationary bootstrap (E2) stalls below ~93% coverage in the stable-law regime | Coverage(SB, T=500) < 93% | **TRUE** (91.8%) |
| `E3_Sub_better_than_SB_in_stable` | Subsampling (E3) achieves higher coverage than the stationary bootstrap (E2) in the stable-law regime | Coverage(Sub, T=500) > Coverage(SB, T=500) | **TRUE** (97.4% > 91.8%) |
| `E4_IM_robust_in_stable_regime` | Ibragimov–Müller (E4) remains robust (coverage ≥ 88%) in the stable-law regime | Coverage(IM, T=500) ≥ 88% | **TRUE** (95.0%) |

Beyond these four formal checks, I want to flag two caveats that I believe a careful
reader of this README should hold onto, in the spirit of reporting limitations as
clearly as results.

**First**, as noted in Section 8.1, the "Paper CI" reported in the EGARCH and
stable-law columns of the Monte Carlo table uses the *true baseline GARCH parameters*,
not parameters estimated from the (misspecified or stable-law) data itself. Its near-95%
coverage in those columns therefore should not be read as "Proposition 4.7 is valid
outside the GARCH-correct regime" — it is closer to "an interval of this particular
width, centred on the sample mean, happens to cover the true SR at roughly this rate
under these alternative DGPs," which is a different and weaker statement. The empirical
application in Section 10, where the Paper CI is constructed from *fitted* parameters
and is consequently simply unavailable for SPY, QQQ, and BTC-USD, is the more faithful
test of Proposition 4.7's domain of applicability.

**Second**, E1 (HAC) is reported throughout as "valid in the finite-fourth-moment
regime, breaks in the stable-law regime" — but Section 9's table shows its stable-law
coverage (90.0%–94.1% across T) is not dramatically different from its finite-moment
coverage (92.0%–92.5% at T=800 in Sections 8.2). The honest statement is that HAC's
*theoretical* justification requires a finite fourth moment, and its *empirical*
coverage in the stable-law DGP at α_stable = 1.7 — which is not extremely far from the
α_stable = 2 (Gaussian) boundary — happens to remain only modestly degraded. A heavier
tail (smaller α_stable) would likely widen this gap; this repository does not sweep
α_stable, and I would treat HAC's stable-law performance as "approximately tolerable at
α_stable ≈ 1.7," not as "robust to arbitrary tail thickness."

---

## 16. How to Cite

If you use or build on this work, please cite both the original paper and this
implementation.

**Original paper:**
```
López de Prado, M., Porcu, E., Zoonekynd, V., & Engle, R. (2026).
Signal-to-Noise Ratio Inference under Volatility Clustering & Heavy Tails.
SSRN Working Paper 6568702. https://ssrn.com/abstract=6568702
```

**This implementation:**
```
Jani, N. M. (2026). Robust SNR Inference under Volatility Clustering and
Non-Gaussian Heavy Tails: Extensions via Subsampling and Self-Normalization.
GitHub repository.
Contact: niharmaheshjani@gmail.com
```

---

## 17. Acknowledgments

This repository would not exist without Proposition 4.7 and Theorem 5.3 of López de
Prado, Porcu, Zoonekynd & Engle (2026) — every closed-form expression reproduced in
Section 3, and every regime boundary referenced throughout Sections 4–11, originates
with their work, and I am grateful for the clarity with which they posed it. I am also
grateful to Enrique Porcu for the conversation that helped me understand *why* the
stationary bootstrap stalls in the stable-law regime (Section 5.3) — a finding that, on
its own, I would have been tempted to dismiss as a tuning problem rather than the
structural feature it turned out to be. Any errors, oversimplifications, or
overstatements in the four extensions (E1–E4), the Monte Carlo design, or the empirical
application are mine alone.

---

## 18. Questions, Corrections, and Feedback

If something in this notebook does not run, produces a result you cannot reproduce, or
you believe there is an error in the mathematics — in the reproduction of Proposition
4.7, or in any of the four extensions — please open an issue, or email me directly at
**niharmaheshjani@gmail.com**. I would genuinely like to know, and I would rather
correct a mistake than let it stand.

---

*Written by Nihar Mahesh Jani — independently, with deep appreciation for the paper that
made it possible, and with no affiliation to its authors.*
