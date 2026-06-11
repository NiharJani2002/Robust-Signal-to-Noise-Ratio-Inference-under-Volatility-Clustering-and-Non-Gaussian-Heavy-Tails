# Robust SNR Inference under Volatility Clustering and Non-Gaussian Heavy Tails
### Extensions via Subsampling and Self-Normalization

**Author:** Nihar Mahesh Jani — niharmaheshjani@gmail.com

> This is an independent reproduction and extension of ideas from *"Signal-to-Noise Ratio Inference under Volatility Clustering & Heavy Tails"* by López de Prado, Porcu, Zoonekynd & Engle (2026), SSRN: [6568702](https://ssrn.com/abstract=6568702). This is **not** the official paper code and is not affiliated with or endorsed by its authors.

---

## Why this project exists

The plug-in Sharpe ratio is one of the most quoted numbers in finance, and one of the least understood when it comes to its confidence interval. Most practitioners — myself included, until I sat down with this paper — treat the textbook standard error as if it were carved in stone. It isn't. The moment your returns show volatility clustering (which is every return series that has ever existed) or heavy tails (which is most of them), that textbook interval becomes too narrow, and your "95% confidence" quietly becomes something closer to 91% or 92%.

López de Prado, Porcu, Zoonekynd & Engle laid out exactly how wrong this can get, and gave a closed-form correction (their Proposition 4.7) for the case where the fourth moment of returns is finite. Reading that paper raised an obvious question for me: **what happens at the edge of their assumptions, and what happens just past it?**

That question is what this repository tries to answer. I built a from-scratch Python reproduction of their core result, then added four extensions of my own — each one designed to relax an assumption the original closed form depends on. None of this changes or critiques their work; it tries to map the territory just beyond it, in the spirit of "here is where the map you drew ends, and here is what I found when I kept walking."

## What's inside

- **`code.ipynb`** — a single, self-contained, numbered notebook (12 sections) that reproduces the paper's constants and figures, then runs the four extensions and a full Monte Carlo study.
- **`results_extended.json`** — the raw numerical output of the Monte Carlo experiments (coverage rates, CI widths, width ratios, and the coverage-vs-sample-size sweep).
- **`result_image.png`** — a 9-panel summary figure combining the original paper's plots with the new results.

## The five methods, side by side

| Method | Origin | Assumption needed | Role |
|---|---|---|---|
| **Paper CI** | López de Prado et al. (2026), Prop. 4.7 | Finite 4th moment, correctly specified GARCH | The baseline — closed-form, fastest, best when the model is right |
| **HAC (E1)** | Newey–West delta-method | Finite 4th moment, any weakly-dependent DGP | Model-free check on the paper CI |
| **Stationary Bootstrap (E2)** | Politis–Romano | Finite variance | General-purpose, but has a hidden floor |
| **Subsampling (E3)** | Politis–Romano–Wolf | None — works without a finite 4th moment | The primary recommendation for heavy-tailed data |
| **Ibragimov–Müller (E4)** | Ibragimov–Müller (2010) | None — self-normalising group t-test | A nearly tuning-free robustness check |

## What the experiments found

**On the paper's home turf (GARCH, correctly specified):** the reproduction lines up with the paper — coverage of 93.2% against a 95% nominal target, and the HAC estimator (E1) comes in slightly narrower (width ratio 0.987) while still covering well. That's a small, encouraging confirmation that the closed-form interval is a touch conservative, not that anything is broken.

**When the model is wrong but moments still exist (EGARCH):** both the Paper CI and HAC hold up reasonably (93.0% and 92.5% coverage), which is exactly what you'd hope — the world doesn't end just because you picked the wrong volatility model, as long as the fourth moment is finite.

**When the fourth moment disappears (the stable-law regime, Theorem 5.3):** this is where things get interesting. The Paper CI and HAC both edge under 95% coverage (93.2% and 92.0%), but the real story is the **Stationary Bootstrap**, which stalls at roughly 91–93% coverage *regardless of sample size* — from T=100 all the way to T=2000. It isn't noisy; it's structurally capped, because it implicitly assumes a variance that, in this regime, doesn't exist.

**Subsampling (E3) and Ibragimov–Müller (E4)** are the two methods that don't share that ceiling. Subsampling tracks toward 95% as T grows (and even overshoots slightly at small T, trading a bit of width for safety), and IM group inference stays close to nominal across the entire range — 94.1% at T=100 and 94.1% at T=2000, remarkably stable for a method with essentially one tuning knob (the number of blocks, q=8).

The headline numbers from `results_extended.json`:

- True SR ≈ 0.01667, V_paper ≈ 1.0127, V_iid ≈ 1.00014 — volatility clustering alone inflates the asymptotic variance by about 1.25% over the i.i.d. benchmark.
- DIAG ≈ 0.0319 — small and positive, meaning the GARCH calibration (α=0.13, β=0.81, φ=0.94) sits close to, but safely inside, the fourth-moment boundary where Prop. 4.7's closed form is valid.

## How to use this

Practically, the regime separation reduces to one question: **does your return series have a finite fourth moment?**

- **Yes** (most equity, FX, rates data over reasonable horizons): the Paper CI is valid and the HAC estimator (E1) is a good model-free cross-check. Both are trustworthy.
- **Uncertain or no** (crypto, distressed credit, anything with episodic extreme moves): reach for Subsampling (E3) as the primary tool, with Ibragimov–Müller (E4) as a second opinion. Avoid leaning on the stationary bootstrap here — it will look fine in a single sample and quietly under-cover on average.

## Running it

```bash
pip install numpy scipy pandas matplotlib numba
jupyter notebook code.ipynb
```

The notebook is deterministic (seeded), self-timed, and produces `snr_extended.png` plus the JSON results table on a full run. The Monte Carlo study (400 reps × 5 methods × 3 DGPs, plus a 1000-rep sweep over sample sizes) takes a few minutes thanks to Numba-compiled inner loops.

## A note on attribution

Every idea that *isn't* explicitly labeled E1–E4 in the notebook belongs to López de Prado, Porcu, Zoonekynd & Engle (2026) — I've simply tried to implement it faithfully and check it against my own simulations. The four extensions (HAC variance, stationary bootstrap, subsampling, and Ibragimov–Müller group inference) are my own additions, built to probe exactly where the original closed form starts to need help, and offered back in the hope they're a useful complement to the original paper rather than a substitute for reading it. If anything here is useful to you, the credit for the underlying question belongs to them — I just got curious about the edges.

If you spot an error, disagree with a modeling choice, or want to push this further, I'd genuinely like to hear about it.

## License

Code is provided for research and educational purposes. Please cite both this repository and the original paper (SSRN 6568702) if you build on this work.
