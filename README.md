# Statistical Pharmacology via Kakutani Dichotomy IX

**Notch-Sparing γ-Secretase Modulators for Alzheimer's Disease: a Second-Order Kakutani Index Screen with a Cone-Geodesic Model of Processive Trimming**

<!-- TODO: replace with the reserved Zenodo DOI before the batch release -->
[![DOI](https://img.shields.io/badge/DOI-reserved%2C%20pending-lightgrey.svg)](https://doi.org/)

Zhengyi Chen<sup>1</sup>, Ruqing Chen<sup>2</sup>

<sup>1</sup> Guangxi Key Laboratory of Drug Discovery and Optimization, School of Pharmacy, Guilin Medical University, Guilin 541199, P. R. China — chenzhengyi@glmc.edu.cn
<sup>2</sup> GUT Geoservice Inc., Montreal, Québec, Canada — ruqing@hotmail.com

This part of the series *Statistical Pharmacology via Kakutani Dichotomy* is the first application paper. It takes the second-order correlated Kakutani index CKI of Paper II, the four-quadrant ligand classifier of Paper IV (DOI 10.5281/zenodo.23018236), the discrete ∞-Laplacian cone geodesic of Paper VI (10.5281/zenodo.23029360) and the estimators of Papers VII–VIII (10.5281/zenodo.23039869, 10.5281/zenodo.23045772), and applies them to one target: Presenilin-1 / γ-secretase.

The pharmacological problem is the one that killed Semagacestat. γ-Secretase cleaves both APP and Notch, and an orthosteric inhibitor that blocks Aβ production blocks Notch S3 proteolysis with it. What is wanted instead is a **modulator**: a ligand that leaves the catalytic dyad alone and instead changes *which* Aβ species the enzyme releases, by rearranging the covariance of the transmembrane machinery that carries out processive trimming.

In the language of the series that is a **quadrant-II ligand**: invisible to an assay that watches the active site, loud in the second-order index.

## Status

Experiment 09 is complete and passes all 8 acceptance tests (`results/exp09_console_log.txt`). The companion script passes its 15 checks (`results/paper09_tables_summary.txt`). The unit tests pass: 14 new model tests plus the 12 inherited package tests.

The paper is complete: `paper/paper09_ad_gamma_secretase_gsm.tex` and `.pdf`, 17 pages, compiling with zero errors, zero warnings and zero overfull or underfull boxes. Its numbers come from `code/paper09_compute_tables.py`, which reads the CSV files and the console log and writes `results/paper09_tables_summary.txt`, the table rows `results/paper09_table{1,2,3,4a,4b,5}_rows.tex` and `results/paper09_tables.json`.

The DOI badge above is a placeholder until the Zenodo reservation is made.

## Main results of the paper

- **Proposition 2.2, exact spectral floor.** The stiffness is a connected graph Laplacian plus a tether, so λ_min = t = 0.9000 exactly, on the uniform direction. Measured: 0.9000, condition number 27.62.
- **Theorem 2.3, second-order Schur screening.** A stiffness perturbation supported on the machinery changes the catalytic precision by a quantity *exactly bilinear* in the coupling block K_CM. Hence leakage and the mean term are O(s²) and the covariance term is O(s⁴), while CKI_TM is untouched. Measured exponents over a 64-fold scan: 1.9906, 1.9709, 1.9755, 3.9633 against 2, 2, 2, 4, with successive local exponents converging monotonically. This is the mechanism that makes quadrant II reachable.
- **Theorem 3.2, closed form and bounds.** For equal weights d_cone = w·Σ_{m≤k} m^{−1/2}, proved by downward induction and reproduced to 4.44×10⁻¹⁶ for k ≤ 6; and max_j w_j ≤ d_cone < Σ_j w_j for k ≥ 2, verified on all 243 ligands.
- **Proposition 3.3 and Numerical observation 3.4, isotropic optimality.** The anisotropy family is *even* in a (proved, since w(−a) is a permutation of w(a)), so a = 0 is critical; on 2001 grid points it is the strict minimiser of d_cone, minimum increment 6.71×10⁻⁸. Schur-convexity of d_cone is stated as an open problem.
- **Proposition 4.1, saturation.** S_Notch ≤ CKI_TM/ε₀ always. Across all 243 ligands the largest ratio S_Notch·ε₀/CKI_TM is 0.9957: the index is essentially always at its ceiling, which is why Flurbiprofen reaches 93 % of its ceiling while failing the potency constraint.
- **Proposition 4.2, what is definitional and what is not.** The activity assay's 0 % recall on quadrant II is definitional given the shared threshold. The empirical content is that quadrant II is non-empty and well populated (60/240), which follows from Theorem 2.3, and that none of the 120 assay hits meets both design constraints although 60 of them are potent.

## Design brief

A ligand *L* is sought in quadrant II of the (K_catalytic, CKI_TM) plane:

| constraint | quantity | threshold | meaning |
|---|---|---|---|
| (1) Notch safety | K_catalytic(L), CKI on the 30 dof of the Asp257–Asp385 dyad and the Notch S3 register | < 0.025 | Notch S3 retention exp(−3 K_catalytic) > 92 % |
| (2) channel rearrangement | CKI_TM(L), CKI on the 270 dof of TM3, TM6a and the PAL motif | > 3.2 | the processive path is actually reorganised |
| (3) synchronous stepping | d<sub>L</sub><sup>cone</sup> on {0,1}<sup>3</sup> | < 3w | the three-residue step becomes one cooperative move |

with the selectivity index and the ratio law

$$S_{\mathrm{Notch}}(L)=\frac{\mathrm{CKI}_{\mathrm{TM}}(L)}{K_{\mathrm{catalytic}}(L)+\varepsilon_0},\qquad R_{42/40}(L)=R_0\,e^{-\eta\left[\mathrm{CKI}_{\mathrm{cov}}(L)-d_L^{\mathrm{cone,eff}}(L)\right]}.$$

## Model

A harmonic (Gaussian) elastic model of the Presenilin-1 catalytic core, 300 degrees of freedom in three blocks:

| block | dof | segments |
|---|---|---|
| catalytic | 30 | D257 loop (8), D385 loop (8), S3 register (14) |
| TM6a/PAL gate | 70 | TM6a (40), PAL (30) |
| TM1–TM9 channel | 200 | TM1 24, TM2 24, TM3 26, TM4 22, TM5 22, TM6 24, TM7 22, TM8 20, TM9 16 |

The stiffness K is the weighted Laplacian of a contact graph plus a bilayer tether, so K is positive definite with λ<sub>min</sub> equal to the tether (0.9) and condition number 27.6, and Σ = K<sup>−1</sup> in units of k<sub>B</sub>T. The catalytic aspartates sit on TM6 and TM7 as they do in PS1, and their couplings to the rest of the structure are deliberately weak (0.35 and 0.25 against path stiffnesses of 4–6). **That weak coupling is the whole mechanism of quadrant II**, and the experiment measures it rather than assuming it: a gate-and-channel ligand changes the catalytic covariance by ‖ΔΣ_C‖/‖Σ_C‖ = 0.045 while reaching CKI_TM = 4.25, against 0.444 for the orthosteric inhibitor.

A ligand enters through three occupancy strengths — active site, gate, channel — which add springs, soften backbones and apply a binding force; μ = Σf.

## Results

### Reference compounds

| compound | s_cat | s_gate | s_chan | K_catalytic | Notch retention | CKI_TM | D_rot | S_Notch | quadrant |
|---|---|---|---|---|---|---|---|---|---|
| Semagacestat | 1.000 | 0.05 | 0.05 | 6.72×10⁻¹ | 13.31 % | 0.034 | 0.029 | 0.051 | **IV** |
| Flurbiprofen | 0.004 | 0.18 | 0.12 | 7.34×10⁻⁵ | 99.98 % | 0.219 | 0.192 | 204 | **III** |
| KMS-AD-309 | 0.020 | 1.00 | 1.00 | 1.69×10⁻³ | 99.49 % | 4.254 | 4.043 | 1583 | **II** |

| compound | w<sub>TM3</sub> | w<sub>TM6a</sub> | w<sub>PAL</sub> | Σw | d<sup>cone</sup> | σ | ΔΔG‡ (kcal/mol) | k(42→38) | R<sub>42/40</sub> | fold |
|---|---|---|---|---|---|---|---|---|---|---|
| Semagacestat | 1.103 | 1.079 | 1.086 | 3.268 | 2.489 | 0.011 | 0.005 | 1.01× | 0.1814 | 1.00× |
| Flurbiprofen | 1.157 | 1.126 | 1.151 | 3.433 | 2.614 | 0.066 | 0.033 | 1.06× | 0.1783 | 1.02× |
| KMS-AD-309 | 1.302 | 1.210 | 1.275 | 3.787 | 2.884 | 0.735 | **0.409** | 1.94× | **0.0698** | **2.61×** |

The three compounds separate exactly as the clinical record does, and for the reason the index says they should:

- **Semagacestat** lands in quadrant IV. It scores an enormous K_catalytic and almost no channel rearrangement, so Notch S3 retention collapses to 13 % while R<sub>42/40</sub> moves by 0.4 %. An inhibitor turns the enzyme down; it does not change what the enzyme makes. That is the Phase III failure mode, in one number.
- **Flurbiprofen** is safe and silent: quadrant III, retention 99.98 %, but CKI_TM = 0.22, a fifteenth of the threshold. R<sub>42/40</sub> falls 2 %.
- **KMS-AD-309** is the target class: K_catalytic 1.7×10⁻³ (fifteen times below the safety threshold), CKI_TM 4.25, and 95 % of its covariance term is the **rotation** part D_rot — it rearranges the channel's eigenvectors rather than merely stiffening the spectrum.

**S_Notch is a ranking index, not a filter.** It saturates at CKI_TM/ε₀ as K_catalytic → 0, which is why Flurbiprofen scores 204 while failing constraint (2) outright. The two absolute constraints select; S_Notch orders what survives.

### The cone geodesic

Each advance of the substrate by three residues requires (x_TM3, x_TM6a, x_PAL) : (0,0,0) → (1,1,1). The exact top-down recursion for the discrete ∞-Laplacian distance on {0,1}³ reproduces the closed form to 4.4×10⁻¹⁶:

$$d_L^{\mathrm{cone}} = w\left(1+\tfrac{1}{\sqrt2}+\tfrac{1}{\sqrt3}\right) = 2.28425\,w = 0.761486 \times 3w.$$

A scan at fixed Σw shows the discount is **largest for an isotropic ligand**: d<sup>cone</sup>/Σw rises from 0.7615 at equal weights to 0.8224 when the weights are spread (1.8, 1.0, 0.2)×w̄. Cooperativity has to be shared across all three helices to be worth anything — a modulator that couples only one helix pays nearly the sequential price.

Two independent routes to the same effect agree within 1.34×: the ratio law predicts a 2.61× fall in R<sub>42/40</sub>, and the cone barrier discount σ·k<sub>B</sub>T·(Σw − d<sup>cone</sup>) = 0.409 kcal/mol predicts a 1.94× acceleration of the Aβ42→Aβ38 trimming step. This consistency is an acceptance criterion, not an observation after the fact.

### Virtual screen, 240 ligands

| quadrant | n | K_catalytic | CKI_TM | Notch retention | R<sub>42/40</sub> |
|---|---|---|---|---|---|
| I | 60 | 2.0×10⁻¹ – 6.7×10⁻¹ | 4.07 – 7.00 | 13.4 – 54.3 % | 0.024 – 0.074 |
| II | 60 | 4.3×10⁻⁶ – 1.2×10⁻² | 4.31 – 6.64 | 96.3 – 100 % | 0.028 – 0.069 |
| III | 60 | 2.0×10⁻⁵ – 1.1×10⁻² | 0.12 – 1.55 | 96.6 – 100 % | 0.147 – 0.180 |
| IV | 60 | 2.1×10⁻¹ – 6.7×10⁻¹ | 0.10 – 1.71 | 13.4 – 53.5 % | 0.143 – 0.180 |

Exactly the 60 quadrant-II ligands satisfy both design constraints. A conventional cell-based activity read-out — a hit iff the active site is engaged, K_catalytic > 0.025 — returns 120 hits, **recall 0 % on quadrant II, and every single hit sits in quadrant I or IV**. The standard assay does not merely miss the useful class; it enriches for the Notch-liable one.

## Limits

- **(a)** The dynamics are a harmonic Gaussian surrogate on a *schematic* contact graph with the correct block topology (Asp257 on TM6, Asp385 on TM7, TM6a/PAL as the gate), not molecular dynamics and not a structure. Nothing here is a docking or free-energy calculation.
- **(b)** The three compounds enter only through their **ligand class** — active-site, gate and channel occupancy strengths — not through their chemistry. The model reproduces the clinical separation of a GSI, a weak GSM and a designed modulator; it does not predict it from structure.
- **(c)** `KMS-AD-309` is a designed point in the model's parameter space, not a synthesised compound.
- **(d)** η = 0.45 and R₀ = 0.182 in the ratio law are phenomenological and calibrated, not derived. The internal-consistency check between the ratio law and the cone barrier constrains them, but does not fix them.
- **(e)** The quadrant classification accuracy of 100 % on the screen is a statement that the sampled strength ranges realise their intended quadrants. It is not a validation of the classifier against experiment; the informative number is the 0 % assay recall.

## Repository layout

```
.
├── kakutani_pharma/        # package inherited from Papers VII-VIII (estimators, featurizer, scaling, calibration)
├── code/
│   ├── exp09_lib.py                      # PS1 elastic model, ligand perturbation, cone geodesic, read-outs
│   ├── exp09_ad_gamma_secretase_gsm.py   # the experiment: compounds, screen, figure, acceptance
│   └── paper09_compute_tables.py         # companion: table rows, Schur exponents, cone checks
├── tests/
│   ├── test_exp09.py                     # 14 model tests
│   ├── test_kakutani_pharma.py           # 8 inherited
│   └── test_calibration.py               # 4 inherited
├── data/
│   ├── exp09_compound_comparison.csv
│   ├── exp09_quadrant_screen.csv
│   └── exp09_cone_scan.csv
├── figures/fig09_ad_gamma_secretase_gsm.{pdf,png}
├── results/                              # exp09_console_log.txt, paper09_tables_summary.txt,
│                                         # paper09_table*_rows.tex, paper09_tables.json
├── paper/                                # paper09_ad_gamma_secretase_gsm.tex / .pdf
├── CITATION.cff, LICENSE, requirements.txt, pyproject.toml, .gitignore, .gitattributes
└── README.md
```

## Reproducing

```bash
pip install -r requirements.txt
python tests/test_exp09.py && python tests/test_kakutani_pharma.py && python tests/test_calibration.py
python code/exp09_ad_gamma_secretase_gsm.py      # about 10 seconds; prints OVERALL ACCEPTANCE: PASS
python code/paper09_compute_tables.py            # table rows and 15 checks; prints ALL CHECKS PASS
cd paper && pdflatex paper09_ad_gamma_secretase_gsm.tex   # run three times
```

## Citation

```bibtex
@misc{ChenChen2026IX,
  author    = {Chen, Zhengyi and Chen, Ruqing},
  title     = {Statistical Pharmacology via Kakutani Dichotomy IX: Notch-Sparing Gamma-Secretase Modulators for Alzheimer's Disease},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.RESERVED}
}
```

## Funding

National Natural Science Foundation of China (No. 22464010), Guangxi Natural Science Foundation of China (No. 2025GXNSFAA069294), and the 2025 Bagui Youth Top Talent Project.

## License

Code under the MIT License; text, figures and data under CC BY 4.0 (see `LICENSE`).
