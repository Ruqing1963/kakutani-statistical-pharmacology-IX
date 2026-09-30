# Statistical Pharmacology via Kakutani Dichotomy IX

**Notch-Sparing γ-Secretase Modulators for Alzheimer's Disease, Schur-Complement Confinement of the Catalytic Block, and a Cone-Geodesic Model of Processive Trimming**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23051518.svg)](https://doi.org/10.5281/zenodo.23051518)

Zhengyi Chen<sup>1</sup>, Ruqing Chen<sup>2</sup>

<sup>1</sup> Guangxi Key Laboratory of Drug Discovery and Optimization, School of Pharmacy, Guilin Medical University, Guilin 541199, P. R. China — chenzhengyi@glmc.edu.cn
<sup>2</sup> GUT Geoservice Inc., Montreal, Québec, Canada — ruqing@hotmail.com

This part of the series *Statistical Pharmacology via Kakutani Dichotomy* is the first application paper. It takes the second-order correlated Kakutani index CKI of Paper II, the four-quadrant ligand classifier of Paper IV (DOI 10.5281/zenodo.23018236), the discrete ∞-Laplacian cone geodesic of Paper VI (10.5281/zenodo.23029360) and the estimators of Papers VII–VIII (10.5281/zenodo.23039869, 10.5281/zenodo.23045772), and applies them to one target: Presenilin-1 / γ-secretase.

The pharmacological problem is the one that killed Semagacestat. γ-Secretase cleaves both APP and Notch, and an orthosteric inhibitor that blocks Aβ production blocks Notch S3 proteolysis with it. What is wanted instead is a **modulator**: a ligand that leaves the catalytic dyad alone and instead changes *which* Aβ species the enzyme releases, by rearranging the covariance of the transmembrane machinery that carries out processive trimming.

In the language of the series that is a **quadrant-II ligand**: invisible to an assay that watches the active site, loud in the second-order index.

## Status

Experiment 09 is complete and passes all 12 acceptance tests (`results/exp09_console_log.txt`). The companion script passes its 23 checks (`results/paper09_tables_summary.txt`). The unit tests pass: 20 model tests plus the 12 inherited package tests.

The paper is complete: `paper/paper09_ad_gamma_secretase_gsm.tex` and `.pdf`, 22 pages, compiling with zero errors, zero warnings and zero overfull or underfull boxes. Its numbers come from `code/paper09_compute_tables.py`, which reads the CSV files and the console log and writes `results/paper09_tables_summary.txt`, the table rows `results/paper09_table{1,2,3,4a,4b,4c,5,6}_rows.tex` and `results/paper09_tables.json`.

DOI: [10.5281/zenodo.23051518](https://doi.org/10.5281/zenodo.23051518). Repository: [Ruqing1963/kakutani-statistical-pharmacology-IX](https://github.com/Ruqing1963/kakutani-statistical-pharmacology-IX).

## Structural basis

The block topology — and only the topology — is taken from three cryo-EM structures of human γ-secretase:

| PDB | content | what the model takes from it |
|---|---|---|
| [6IYC](https://www.rcsb.org/structure/6IYC) | γ-secretase–APP-C83, 2.6 Å ([Zhou et al., *Science* **363** (2019) eaaw0930](https://doi.org/10.1126/science.aaw0930)) | APP TM surrounded by five PS1 helices; hybrid β-sheet sets the cleavage register; FAD mutations cluster at the PS1–substrate interface |
| [6IDF](https://www.rcsb.org/structure/6IDF) | γ-secretase–Notch-100, 2.7 Å ([Yang et al., *Nature* **565** (2019) 192–197](https://doi.org/10.1038/s41586-018-0813-8)) | Notch uses the same register machinery, so an orthosteric inhibitor cannot separate the two substrates |
| [7D8X](https://www.rcsb.org/structure/7D8X) | γ-secretase + E2012 + L685,458, 2.60 Å ([Yang et al., *Cell* **184** (2021) 521–533.e14](https://doi.org/10.1016/j.cell.2020.11.049)) | an inhibitor and a modulator bind the *same enzyme at once*, at different sites — the structural form of the two-coordinate picture |

**The model is not built on these coordinates.** Nothing is docked, no geometry is used, and in particular the model does *not* place E2012 at the extracellular allosteric site that 7D8X resolves: it represents a modulator by the gate–channel coupling it induces, which is a consequence rather than a pose. See Limit (a).

## Main results of the paper

- **Proposition 2.2, exact spectral floor.** The stiffness is a connected graph Laplacian plus a tether, so λ_min = t = 0.9000 exactly, on the uniform direction. Measured: 0.9000, condition number 27.62.
- **Theorem 2.3, second-order Schur screening.** A stiffness perturbation supported on the machinery changes the catalytic precision by a quantity *exactly bilinear* in the coupling block K_CM. Hence leakage and the mean term are O(s²) and the covariance term is O(s⁴), while CKI_TM is untouched. Measured exponents over a 64-fold scan: 1.9906, 1.9709, 1.9755, 3.9633 against 2, 2, 2, 4, with successive local exponents converging monotonically. This is the mechanism that makes quadrant II reachable.
- **Theorem 3.2, closed form and bounds.** For equal weights d_cone = w·Σ_{m≤k} m^{−1/2}, proved by downward induction and reproduced to 4.44×10⁻¹⁶ for k ≤ 6; and max_j w_j ≤ d_cone < Σ_j w_j for k ≥ 2, verified on all 243 ligands.
- **Proposition 3.3 and Numerical observation 3.4, isotropic optimality.** The anisotropy family is *even* in a (proved, since w(−a) is a permutation of w(a)), so a = 0 is critical; on 2001 grid points it is the strict minimiser of d_cone, minimum increment 6.71×10⁻⁸. Schur-convexity of d_cone is stated as an open problem.
- **Proposition 3.5, both biomarkers from one partition.** Exact for the model: Aβ38/Aβ42 = ρ and R₄₂/₄₀ = R₀(1+ρ₀)/(1+ρ), with the design-brief exponential law recovered as the ρ ≫ 1 limit. One fitted constant reproduces the one-constant law across five compounds to within 8.91 %.
- **Proposition 4.1, saturation.** S_Notch ≤ CKI_TM/ε₀ always. Across all 245 ligands the largest ratio S_Notch·ε₀/CKI_TM is 0.9957: the index is essentially always at its ceiling, which is why R-flurbiprofen reaches 93 % of its ceiling while failing the potency constraint, and why it ranks E2012 above the more potent lead.
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
| Avagacestat | 0.280 | 0.32 | 0.28 | 1.44×10⁻¹ | 64.96 % | 0.713 | 0.648 | 4.92 | **IV** |
| R-Flurbiprofen | 0.004 | 0.18 | 0.12 | 7.34×10⁻⁵ | 99.98 % | 0.219 | 0.192 | 204 | **III** |
| E2012 | 0.008 | 0.95 | 0.92 | 2.93×10⁻⁴ | 99.91 % | 3.818 | 3.640 | 2952 | **II** |
| KMS-AD-309 | 0.020 | 1.00 | 1.00 | 1.69×10⁻³ | 99.49 % | 4.254 | 4.043 | 1583 | **II** |

| compound | Γ | ΔΔG‡ (kcal/mol) | Aβ38/Aβ42 | ×WT | R<sub>42/40</sub> | fold |
|---|---|---|---|---|---|---|
| Semagacestat | +0.008 | +0.005 | 3.030 | 1.010× | 0.1806 | 1.008× |
| Avagacestat | +0.167 | +0.105 | 3.696 | 1.232× | 0.1550 | 1.174× |
| R-Flurbiprofen | +0.046 | +0.033 | 3.201 | 1.067× | 0.1733 | 1.050× |
| E2012 | +1.806 | +0.388 | 8.466 | 2.822× | 0.0769 | 2.367× |
| KMS-AD-309 | +2.130 | **+0.409** | **9.434** | **3.145×** | **0.0698** | **2.608×** |

The five separate exactly as the clinical record does, and for the reason the index says they should:

- **Semagacestat** lands in quadrant IV. Enormous K_catalytic, almost no channel rearrangement, so Notch S3 retention collapses to 13 % while **both** biomarkers move by 1 %. An inhibitor turns the enzyme down; it does not change what the enzyme makes. That is the Phase III failure mode, in two numbers.
- **Avagacestat** was sold as "Notch-sparing". The model agrees with the outcome, not the label: it *is* a milder inhibitor than Semagacestat (4.7× lower K_catalytic), but at 5.8× the safety threshold and 65 % retention it is still quadrant IV.
- **R-Flurbiprofen** (tarenflurbil) is safe and silent: quadrant III, retention 99.98 %, CKI_TM = 0.22, a fifteenth of the threshold.
- **E2012**, the second-generation bridged-imidazole class, is quadrant II: retention 99.91 %, R₄₂/₄₀ down 2.37× *with* Aβ38/Aβ42 up 2.82× — the co-movement this class reports clinically.
- **KMS-AD-309** is the design target: K_catalytic 1.7×10⁻³ (fifteen times below the safety threshold), CKI_TM 4.25, and 95 % of its covariance term is the **rotation** part D_rot — it rearranges the channel's eigenvectors rather than merely stiffening the spectrum.

**S_Notch is a ranking index, not a filter.** It saturates at CKI_TM/ε₀ as K_catalytic → 0. It scores R-flurbiprofen 204 while that compound fails constraint (2) outright, and it ranks E2012 (2952) *above* the more potent lead (1583) on a K_catalytic difference two orders of magnitude below the safety threshold. The two absolute constraints select; S_Notch only orders what survives.

### Both clinical biomarkers from one branching partition

At the last intermediate of the pathogenic line the enzyme holds E·Aβ42 and either releases it or cuts once more to Aβ38:

$$k_{\mathrm{off},42}=k^{(0)}_{\mathrm{off},42}e^{-\eta_{\mathrm{clamp}}\Gamma},\qquad k_{42\to38}=k^{(0)}_{42\to38}e^{\Delta\Delta G^\ddagger/k_BT},\qquad \rho=\frac{k_{42\to38}}{k_{\mathrm{off},42}}.$$

Proposition 3.5 then gives **both** clinical read-outs from that single partition, exactly:

$$\frac{\mathrm{A}\beta_{38}}{\mathrm{A}\beta_{42}}=\rho,\qquad R_{42/40}=R_0\frac{1+\rho_0}{1+\rho},$$

and the exponential law of the design brief is its ρ ≫ 1 limit. This replaces a *comparison* by a *derivation*: the barrier discount and the clamp are not two independent predictions but the numerator and denominator of one ratio. With ρ₀ = 3.00 fixed as a biochemical input and η_clamp = 0.22620 fitted on **one** anchor, the branching law reproduces the one-constant design-brief law across all five compounds to within 8.91 %.

The consequence is falsifiable: every compound must lie on **one curve** in the (R₄₂/₄₀, Aβ38/Aβ42) plane, with no free parameter per compound.

### The cone geodesic

Each advance of the substrate by three residues requires (x_TM3, x_TM6a, x_PAL) : (0,0,0) → (1,1,1). The exact top-down recursion for the discrete ∞-Laplacian distance on {0,1}³ reproduces the closed form to 4.4×10⁻¹⁶:

$$d_L^{\mathrm{cone}} = w\left(1+\tfrac{1}{\sqrt2}+\tfrac{1}{\sqrt3}\right) = 2.284457\,w = 0.761486 \times 3w.$$

A scan at fixed Σw shows the discount is **largest for an isotropic ligand**: d<sup>cone</sup>/Σw rises from 0.7615 at equal weights to 0.8224 when the weights are spread (1.8, 1.0, 0.2)×w̄. Cooperativity has to be shared across all three helices to be worth anything — a modulator that couples only one helix pays nearly the sequential price.

### Virtual screen, 240 ligands

| quadrant | n | K_catalytic | CKI_TM | Notch retention | R<sub>42/40</sub> |
|---|---|---|---|---|---|
| I | 60 | 2.0×10⁻¹ – 6.7×10⁻¹ | 4.07 – 7.00 | 13.4 – 54.3 % | 0.024 – 0.074 |
| II | 60 | 4.3×10⁻⁶ – 1.2×10⁻² | 4.31 – 6.64 | 96.3 – 100 % | 0.028 – 0.069 |
| III | 60 | 2.0×10⁻⁵ – 1.1×10⁻² | 0.12 – 1.55 | 96.6 – 100 % | 0.147 – 0.180 |
| IV | 60 | 2.1×10⁻¹ – 6.7×10⁻¹ | 0.10 – 1.71 | 13.4 – 53.5 % | 0.143 – 0.180 |

Exactly the 60 quadrant-II ligands satisfy both design constraints. A conventional cell-based activity read-out — a hit iff the active site is engaged, K_catalytic > 0.025 — returns 120 hits, **recall 0 % on quadrant II, and every single hit sits in quadrant I or IV**. The standard assay does not merely miss the useful class; it enriches for the Notch-liable one.

### Familial AD: gate destabilisation, and allosteric rescue

Two *PSEN1* alleles are modelled **not** as gains of activity but as losses of stiffness in the parts that hold and advance the substrate: **PS1-L166P** (helix-breaking proline in TM3) and **PS1-E280A** (TM6–TM7 hydrophilic loop, adjacent to TM6a).

Both rearrange the machinery as strongly as a potent drug does — and in the *opposite direction*. The mean step weight falls below w₀, so ζ = −1 and the clamp index goes negative:

| allele | CKI_TM | w̄ | ζ | Γ | Aβ38/Aβ42 | R<sub>42/40</sub> |
|---|---|---|---|---|---|---|
| PS1-L166P | 3.510 | 1.0131 | **−1** | −1.968 | 1.186 (from 3.00) | **0.3330** (from 0.182) |
| PS1-E280A | 3.744 | 1.0170 | **−1** | −2.142 | 1.119 | **0.3436** |

Applying each compound to each allele (Notch retention measured against the *untreated allele*, i.e. the liability the drug **adds**):

| allele | compound | Γ | Aβ38/Aβ42 | R<sub>42/40</sub> | fold | retention | m\* for R < 0.12 |
|---|---|---|---|---|---|---|---|
| PS1-L166P | untreated | −1.968 | 1.186 | 0.3330 | 1.00× | 100.00 % | — |
| | Semagacestat | −1.954 | 1.208 | 0.3297 | 1.01× | **13.30 %** | > 4 |
| | Avagacestat | −1.640 | 1.648 | 0.2749 | 1.21× | 64.94 % | 3.689 |
| | R-Flurbiprofen | −1.890 | 1.312 | 0.3149 | 1.06× | 99.98 % | > 4 |
| | E2012 | **+0.438** | 4.033 | **0.1446** | 2.30× | 99.91 % | 1.188 |
| | KMS-AD-309 | **+0.707** | 4.385 | **0.1352** | 2.46× | 99.49 % | 1.112 |
| PS1-E280A | untreated | −2.142 | 1.119 | 0.3436 | 1.00× | 100.00 % | — |
| | Semagacestat | −2.128 | 1.140 | 0.3402 | 1.01× | **13.31 %** | > 4 |
| | Avagacestat | −1.839 | 1.530 | 0.2877 | 1.19× | 64.96 % | 3.981 |
| | R-Flurbiprofen | −2.045 | 1.263 | 0.3216 | 1.07× | 99.98 % | > 4 |
| | E2012 | **+0.148** | 3.659 | **0.1563** | 2.20× | 99.91 % | 1.244 |
| | KMS-AD-309 | **+0.470** | 4.043 | **0.1444** | 2.38× | 99.49 % | 1.153 |

Only the two quadrant-II modulators **reverse the sign of the clamp**, and both bring R₄₂/₄₀ below the healthy wild-type baseline 0.182 at ≥ 99.49 % Notch retention. The stricter target of 0.12 is **not** reached at nominal occupancy (the four rescues land at 0.135–0.156); it needs 1.11–1.24× nominal exposure, which is what the m\* column reports. We state that rather than tuning a constant until the target is met at m = 1.

Semagacestat is the instructive failure: on a familial background it leaves R₄₂/₄₀ essentially untouched (1.01×) while cutting Notch retention to 13 %. It delivers the entire toxicity of the mechanism and none of the benefit.

**Why the clamp must be additive.** Γ is assembled as Γ(allele vs wild type) + Γ(drug vs untreated allele), each with its own sign. The tempting alternative — measure everything against wild type with one global sign — is catastrophically wrong: because CKI is a *distance*, the allele's own rearrangement gets counted as a therapeutic benefit, and the single-term construction has R-flurbiprofen "curing" PS1-L166P to R₄₂/₄₀ = 0.082 and **Semagacestat curing it to 0.086** — better than healthy, while inhibiting Notch by 87 %. The companion script computes this counterfactual explicitly as a check.

## Limits

- **(a)** The dynamics are a harmonic Gaussian surrogate on a *schematic* contact graph. The structures 6IYC, 6IDF and 7D8X fix **topology only** — which parts exist and how they are wired. No coordinates are used, nothing is docked, no free energy is computed, and the model does *not* place E2012 at the extracellular allosteric site 7D8X resolves.
- **(b)** All five compounds enter only through their **ligand class** — active-site, gate and channel occupancy strengths — not through their chemistry, and the same holds for the two alleles, whose softening factors encode "severe loss of gate stiffness" rather than any calculation from the mutation. The model reproduces the clinical separation; it does not predict it from structure.
- **(c)** `KMS-AD-309` is a designed point in the model's parameter space, not a synthesised compound.
- **(d)** Three constants are calibrated: η = 0.45 and R₀ = 0.182 in the design-brief law, and η_clamp = 0.22620 in the branching law (ρ₀ = 3.00 is an input, not a fit). The **severity of the two alleles was chosen** so their untreated R₄₂/₄₀ lands in the reported 0.33–0.36 range, so that range is an input. Not fitted: the sign of the alleles' effect, the co-movement of the biomarkers, the rescue, or any quadrant assignment.
- **(e)** The quadrant classification accuracy of 100 % on the screen is a statement that the sampled strength ranges realize their intended quadrants — a design check on the sampler, not a validation against experiment. The informative number is the 0 % assay recall, and within it the *empirical* parts (ii) and (iii) of Proposition 4.2 rather than the definitional part (i).
- **(e)** The quadrant classification accuracy of 100 % on the screen is a statement that the sampled strength ranges realise their intended quadrants. It is not a validation of the classifier against experiment; the informative number is the 0 % assay recall.

## Repository layout

```
.
├── kakutani_pharma/        # package inherited from Papers VII-VIII (estimators, featurizer, scaling, calibration)
├── code/
│   ├── exp09_lib.py                      # PS1 elastic model, ligand perturbation, cone geodesic, read-outs
│   ├── exp09_ad_gamma_secretase_gsm.py   # the experiment: compounds, screen, FAD rescue, figures
│   └── paper09_compute_tables.py         # companion: table rows, Schur exponents, cone and FAD checks
├── tests/
│   ├── test_exp09.py                     # 20 model tests
│   ├── test_kakutani_pharma.py           # 8 inherited
│   └── test_calibration.py               # 4 inherited
├── data/
│   ├── exp09_compound_comparison.csv
│   ├── exp09_quadrant_screen.csv
│   ├── exp09_cone_scan.csv
│   └── exp09_fad_rescue.csv
├── figures/                              # fig09_ad_gamma_secretase_gsm.{pdf,png}
│                                         # fig09b_biomarkers_and_fad_rescue.{pdf,png}
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
python code/exp09_ad_gamma_secretase_gsm.py      # about 20 seconds; prints OVERALL ACCEPTANCE: PASS
python code/paper09_compute_tables.py            # table rows and 23 checks; prints ALL CHECKS PASS
cd paper && pdflatex paper09_ad_gamma_secretase_gsm.tex   # run three times
```

## Citation

```bibtex
@misc{ChenChen2026IX,
  author    = {Chen, Zhengyi and Chen, Ruqing},
  title     = {Statistical Pharmacology via Kakutani Dichotomy IX: Notch-Sparing Gamma-Secretase Modulators for Alzheimer's Disease, Schur-Complement Confinement of the Catalytic Block, and a Cone-Geodesic Model of Processive Trimming},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23051518},
  url       = {https://doi.org/10.5281/zenodo.23051518}
}
```

## Funding

National Natural Science Foundation of China (No. 22464010), Guangxi Natural Science Foundation of China (No. 2025GXNSFAA069294), and the 2025 Bagui Youth Top Talent Project.

## License

Code under the MIT License; text, figures and data under CC BY 4.0 (see `LICENSE`).
