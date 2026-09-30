#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp09_ad_gamma_secretase_gsm.py
===============================
Statistical Pharmacology via Kakutani Dichotomy, Paper IX -- Experiment 09.
Notch-sparing gamma-secretase modulator (GSM) screen for Alzheimer's disease, built on the
second-order correlated Kakutani index (CKI) of Paper II and the discrete infinity-Laplacian
cone geodesic of Paper VI, Section 9.

Design target: a quadrant-II ligand of the (K_catalytic, CKI_TM) plane, that is, one that
    (1) leaves the catalytic dyad and the Notch S3 register undistorted,
            K_catalytic(L) < 0.025, hence Notch retention exp(-3 K_catalytic) > 92 %;
    (2) rearranges the covariance of the processive-trimming machinery (TM3, TM6a, PAL),
            CKI_TM(L) > 3.2;
    (3) turns the three-residue processive step (0,0,0) -> (1,1,1) of (x_TM3, x_TM6a, x_PAL)
            from a sequential into a synchronous displacement, whose cost is the cone geodesic
            d_L^cone = w (1 + 1/sqrt(2) + 1/sqrt(3)) = 2.284457 w < 3 w for equal weights.

Read-outs
    S_Notch(L)   = CKI_TM(L) / (K_catalytic(L) + eps_0)
    R_42/40(L)   = R_0 exp(-eta [CKI_cov(L) - d_cone_eff(L)])
    ddG barrier  = sigma(L) k_B T [sum_j w_j(L) - d_L^cone(L)]      (Abeta42 -> Abeta38)

Reference compounds: Semagacestat (orthosteric GSI, Phase III failure), Flurbiprofen
(first-generation weak GSM) and the Kakutani-spectrum lead KMS-AD-309.

Outputs
    data/exp09_compound_comparison.csv
    data/exp09_quadrant_screen.csv
    data/exp09_cone_scan.csv
    figures/fig09_ad_gamma_secretase_gsm.{pdf,png}
    results/exp09_console_log.txt

The model is a harmonic (Gaussian) surrogate on a schematic Presenilin-1 contact graph, not a
molecular-dynamics simulation, and the compounds enter only through their ligand class. The
numbers are properties of that model. See README.md, "Limits".
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp09_lib as M  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _repo_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    return d if os.path.isdir(d) else HERE


CSV_COMPOUNDS = os.path.join(_repo_dir("data"), "exp09_compound_comparison.csv")
CSV_SCREEN = os.path.join(_repo_dir("data"), "exp09_quadrant_screen.csv")
CSV_CONE = os.path.join(_repo_dir("data"), "exp09_cone_scan.csv")
CSV_FAD = os.path.join(_repo_dir("data"), "exp09_fad_rescue.csv")
FIG_PATH = os.path.join(_repo_dir("figures"), "fig09_ad_gamma_secretase_gsm.png")
FIG_PATH_PDF = os.path.splitext(FIG_PATH)[0] + ".pdf"
FIG2_PATH = os.path.join(_repo_dir("figures"), "fig09b_biomarkers_and_fad_rescue.png")
FIG2_PATH_PDF = os.path.splitext(FIG2_PATH)[0] + ".pdf"
LOG_PATH = os.path.join(_repo_dir("results"), "exp09_console_log.txt")

SEED_SCREEN = 20260929
M_PER_QUADRANT = 60
SCREEN_RANGES = {                      # s_cat,           s_gate,          s_chan
    "I": ((0.35, 1.00), (0.95, 1.35), (0.95, 1.35)),
    "II": ((0.00, 0.06), (0.95, 1.35), (0.95, 1.35)),
    "III": ((0.00, 0.06), (0.05, 0.55), (0.05, 0.55)),
    "IV": ((0.35, 1.00), (0.05, 0.55), (0.05, 0.55)),
}
CONE_ANISOTROPY = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
TOL_CONE = 1e-12
TOL_BRANCH = 0.10                      # branching law against the design-brief law, relative

# ----------------------------------------------------------------------------
# Figure style (series house style)
# ----------------------------------------------------------------------------
SURFACE, INK, INK_2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                           "#e87ba4", "#008300", "#4a3aa7", "#e34948")
Q_COLOR = {"I": VIOLET, "II": ORANGE, "III": MUTED, "IV": RED}
C_COLOR = {"Semagacestat": RED, "Avagacestat": MAGENTA, "R-Flurbiprofen": BLUE,
           "E2012": AQUA, "KMS-AD-309": ORANGE}
BG_COLOR = {"wild type": INK_2, "PS1-L166P": VIOLET, "PS1-E280A": YELLOW}
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9, "axes.unicode_minus": False,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left", "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "legend.labelcolor": INK_2, "mathtext.fontset": "dejavusans",
})


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6)


class Tee:
    """Duplicate stdout into the console log."""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s: str) -> int:
        self.out.write(s)
        self.f.write(s)
        return len(s)

    def flush(self) -> None:
        self.out.flush()
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run() -> int:
    t0 = time.time()
    ok_all = True
    print("Experiment 09: Notch-sparing gamma-secretase modulators from the second-order Kakutani index")
    print("Statistical Pharmacology via Kakutani Dichotomy, Paper IX")

    # ---------- 1. model ----------
    print("\n[1] Gaussian elastic model of the Presenilin-1 catalytic core")
    apo = M.ensemble(M.APO)
    eig = np.linalg.eigvalsh(apo.K)
    print(f"  degrees of freedom: catalytic {M.N_CAT} + gate (TM6a, PAL) {M.N_GATE} "
          f"+ channel (TM1-TM9) {M.N_CHAN} = {M.N_DOF}")
    print(f"  segments: " + ", ".join(f"{n}({s.stop - s.start})" for n, s in M.SEG.items()))
    print(f"  stiffness K: eigenvalues in [{eig[0]:.4f}, {eig[-1]:.4f}], condition number {eig[-1] / eig[0]:.2f}, "
          f"dim ker = {int(np.sum(eig < 1e-10))}")
    print(f"  weak-coupling design: K_cat-channel = {M.K_CAT_CHAN}, K_cat-gate = {M.K_CAT_GATE} against "
          f"K_path = {M.K_PATH}, K_pack = {M.K_PACK}")
    passed = (bool(eig[0] > 0) and abs(eig[0] - M.T_TETHER) < 1e-9
              and M.N_CAT == 30 and M.N_GATE == 70 and M.N_CHAN == 200 and M.N_DOF == 300)
    ok_all &= passed
    print(f"  Acceptance: K positive definite with lambda_min = tether {M.T_TETHER}, block sizes 30/70/200  "
          f"{'PASS' if passed else 'FAIL'}")

    # ---------- 2. cone geodesic ----------
    print("\n[2] Three-residue step: discrete infinity-Laplacian cone geodesic on {0,1}^3")
    exact = M.cone_equal_weight(M.W_STEP, 3)
    numeric = M.cone_geodesic([M.W_STEP] * 3)
    err = abs(numeric - exact)
    print(f"  equal weights w = {M.W_STEP}: recursion {numeric:.15f}, closed form "
          f"w(1 + 1/sqrt(2) + 1/sqrt(3)) = {exact:.15f}, error {err:.2e}")
    print(f"  discount against the sequential path: d_cone / 3w = {numeric / (3 * M.W_STEP):.6f} "
          f"(= (1 + 1/sqrt(2) + 1/sqrt(3)) / 3 = {M.cone_equal_weight(1.0, 3) / 3:.6f}), "
          f"sequential 3w = {3 * M.W_STEP:.4f}")
    print(f"  {'a':>5} {'w_TM3':>8} {'w_TM6a':>8} {'w_PAL':>8} {'sum w':>8} {'d_cone':>9} {'d/sum w':>9} {'d/max w':>9}")
    cone_rows = []
    wbar = M.W_STEP
    for a in CONE_ANISOTROPY:
        w = np.array([wbar * (1 + a), wbar, wbar * (1 - a)])
        d = M.cone_geodesic(w)
        cone_rows.append({"anisotropy": a, "w_TM3": w[0], "w_TM6a": w[1], "w_PAL": w[2],
                          "w_sum": float(w.sum()), "d_cone": d, "d_over_w_sum": d / float(w.sum()),
                          "d_over_w_max": d / float(w.max()),
                          "discount_kT": float(w.sum()) - d})
        print(f"  {a:>5.1f} {w[0]:>8.4f} {w[1]:>8.4f} {w[2]:>8.4f} {w.sum():>8.4f} {d:>9.5f} "
              f"{d / w.sum():>9.6f} {d / w.max():>9.6f}")
    write_csv(CSV_CONE, cone_rows)
    mono = all(cone_rows[i]["d_cone"] <= cone_rows[i + 1]["d_cone"] + 1e-12 for i in range(len(cone_rows) - 1))
    passed = err < TOL_CONE and all(r["d_cone"] < r["w_sum"] for r in cone_rows) and mono
    ok_all &= passed
    print(f"  Acceptance: closed form reproduced to {err:.1e} (< {TOL_CONE:.0e}), d_cone < sum w throughout, "
          f"and d_cone non-decreasing in the anisotropy at fixed sum w  {'PASS' if passed else 'FAIL'}")

    # ---------- 3. reference compounds ----------
    print("\n[3] Reference compounds against the Kakutani-spectrum lead")
    rows = [M.metrics(spec, apo) for spec in M.COMPOUNDS]
    by = {r["ligand"]: r for r in rows}
    write_csv(CSV_COMPOUNDS, rows)

    print(f"  {'compound':>14} {'s_cat':>6} {'s_gate':>7} {'s_chan':>7} {'K_catalytic':>12} {'Notch ret.':>11} "
          f"{'CKI_TM':>8} {'D_rot':>8} {'S_Notch':>10} {'quadrant':>9}")
    for r in rows:
        print(f"  {r['ligand']:>14} {r['s_cat']:>6.3f} {r['s_gate']:>7.2f} {r['s_chan']:>7.2f} "
              f"{r['K_catalytic']:>12.3e} {100 * r['notch_retention']:>10.2f}% {r['CKI_TM']:>8.4f} "
              f"{r['CKI_TM_rot']:>8.4f} {r['S_Notch']:>10.4g} {r['quadrant']:>9}")
    print(f"\n  {'compound':>14} {'w_TM3':>7} {'w_TM6a':>7} {'w_PAL':>7} {'sum w':>7} {'d_cone':>7} {'sigma':>7} "
          f"{'ddG (kcal/mol)':>15} {'k(42->38)':>10} {'R_42/40':>9} {'fold':>6}")
    for r in rows:
        print(f"  {r['ligand']:>14} {r['w_TM3']:>7.4f} {r['w_TM6a']:>7.4f} {r['w_PAL']:>7.4f} {r['w_sum']:>7.4f} "
              f"{r['d_cone']:>7.4f} {r['sigma_sync']:>7.4f} {r['ddG_barrier_kcal']:>15.4f} "
              f"{r['rate_factor_42_38']:>10.3f} {r['R_42_40']:>9.5f} {r['R_fold_reduction']:>6.2f}x")
    print(f"\n  reference ratio R_0 = {M.R0_42_40:.3f}, eta = {M.ETA_R}, eps_0 = {M.EPS0}, "
          f"k_B T = {M.KT_KCAL} kcal/mol at 310 K")
    for r in rows:
        print(f"  {r['ligand']:>14}: {M.QUADRANT_NAME[r['quadrant']]}")
        print(f"  {'':>14}  {r['note']}")

    # ---------- 3b. the two clinical biomarkers from the branching model ----------
    print("\n[3b] Branching kinetics at the E.Abeta42 intermediate: the two clinical read-outs")
    print(f"  rho_0 = {M.RHO0} (wild-type Abeta38/Abeta42), eta_clamp = {M.ETA_CLAMP}, "
          f"anchored on KMS-AD-309 at R = {M.R_LEAD_ANCHOR}")
    print(f"  {'compound':>14} {'Gamma':>8} {'ddG':>8} {'Abeta38/42':>11} {'x WT':>7} "
          f"{'R_42/40 (branch)':>17} {'fold':>7} {'R (brief law)':>14} {'disagreement':>13}")
    apo_row = M.metrics(M.APO, apo)
    for r in [apo_row] + rows:
        disc = abs(r["R_branch_fold"] / r["R_fold_reduction"] - 1)
        print(f"  {r['ligand']:>14} {r['Gamma']:>8.4f} {r['ddG_barrier_kcal']:>8.4f} "
              f"{r['ratio_38_42']:>11.4f} {r['ratio_38_42_fold']:>7.3f} {r['R_42_40_branch']:>17.5f} "
              f"{r['R_branch_fold']:>6.3f}x {r['R_42_40']:>14.5f} {100 * disc:>12.2f}%")
    max_disc = max(abs(r["R_branch_fold"] / r["R_fold_reduction"] - 1) for r in rows)
    passed = max_disc < TOL_BRANCH and abs(apo_row["ratio_38_42"] - M.RHO0) < 1e-12 \
        and abs(apo_row["R_42_40_branch"] - M.R0_42_40) < 1e-12
    ok_all &= passed
    print(f"  Acceptance: the branching law returns (rho_0, R_0) exactly at apo and agrees with the "
          f"design-brief law to {100 * max_disc:.2f}% (< {100 * TOL_BRANCH:.0f}%) on all "
          f"{len(rows)} compounds  {'PASS' if passed else 'FAIL'}")

    sema, avag = by["Semagacestat"], by["Avagacestat"]
    flur, e2012, lead = by["R-Flurbiprofen"], by["E2012"], by["KMS-AD-309"]
    passed = (sema["quadrant"] == "IV" and sema["notch_retention"] < M.NOTCH_MIN
              and avag["quadrant"] == "IV" and avag["notch_retention"] < M.NOTCH_MIN
              and flur["quadrant"] == "III" and flur["notch_retention"] > M.NOTCH_MIN
              and e2012["quadrant"] == "II" and e2012["notch_retention"] > M.NOTCH_MIN
              and lead["quadrant"] == "II" and lead["notch_retention"] > M.NOTCH_MIN
              and lead["K_catalytic"] < M.K_CAT_MAX and lead["CKI_TM"] > M.CKI_TM_MIN)
    ok_all &= passed
    print(f"\n  Acceptance: Semagacestat in IV (retention {100 * sema['notch_retention']:.2f}%), "
          f"Avagacestat in IV despite its Notch-sparing label (K_cat {avag['K_catalytic']:.3f} = "
          f"{avag['K_catalytic'] / M.K_CAT_MAX:.1f}x the threshold, retention "
          f"{100 * avag['notch_retention']:.2f}%), R-Flurbiprofen in III (CKI_TM {flur['CKI_TM']:.3f}), "
          f"E2012 and KMS-AD-309 in II  {'PASS' if passed else 'FAIL'}")

    passed = (e2012["R_branch_fold"] > 2.0 and e2012["ratio_38_42_fold"] > 2.0
              and lead["R_branch_fold"] > 2.0 and lead["ratio_38_42_fold"] > 2.0
              and sema["R_branch_fold"] < 1.05 and sema["ratio_38_42_fold"] < 1.05
              and avag["ratio_38_42_fold"] < 1.30)
    ok_all &= passed
    print(f"  Acceptance: the two clinical biomarkers move together for a modulator and not at all for "
          f"an inhibitor: E2012 {e2012['R_branch_fold']:.2f}x down / {e2012['ratio_38_42_fold']:.2f}x up, "
          f"KMS-AD-309 {lead['R_branch_fold']:.2f}x / {lead['ratio_38_42_fold']:.2f}x, Semagacestat "
          f"{sema['R_branch_fold']:.3f}x / {sema['ratio_38_42_fold']:.3f}x  {'PASS' if passed else 'FAIL'}")

    passed = lead["S_Notch"] > flur["S_Notch"] > sema["S_Notch"] and lead["S_Notch"] / sema["S_Notch"] > 1e3
    ok_all &= passed
    print(f"  Acceptance: S_Notch ordering lead {lead['S_Notch']:.1f} > R-Flurbiprofen {flur['S_Notch']:.1f} "
          f"> Semagacestat {sema['S_Notch']:.4f}, lead/Semagacestat = {lead['S_Notch'] / sema['S_Notch']:.3g} "
          f"(> 1e3)  {'PASS' if passed else 'FAIL'}")
    print(f"  Note: S_Notch saturates at CKI_TM / eps_0 as K_catalytic -> 0, so R-Flurbiprofen scores "
          f"{flur['S_Notch']:.0f} while failing constraint (2). The index ranks, the two absolute "
          f"constraints select.")

    passed = (lead["R_fold_reduction"] > 2.0 and sema["R_fold_reduction"] < 1.05
              and flur["R_fold_reduction"] < 1.10)
    ok_all &= passed
    print(f"  Acceptance: R_42/40 reduction lead {lead['R_fold_reduction']:.2f}x (> 2), Semagacestat "
          f"{sema['R_fold_reduction']:.3f}x (< 1.05, an inhibitor does not shift the ratio), R-Flurbiprofen "
          f"{flur['R_fold_reduction']:.3f}x (< 1.10)  {'PASS' if passed else 'FAIL'}")

    passed = lead["leakage_cat"] < 0.10 and sema["leakage_cat"] > 0.10
    ok_all &= passed
    print(f"  Acceptance: the weak catalytic couplings confine the allosteric perturbation: lead leakage "
          f"||dSigma_C|| / ||Sigma_C|| = {lead['leakage_cat']:.4f} (< 0.10) at CKI_TM = {lead['CKI_TM']:.2f}, "
          f"against {sema['leakage_cat']:.4f} for Semagacestat  {'PASS' if passed else 'FAIL'}")

    # ---------- 4. virtual screen ----------
    n_screen = M_PER_QUADRANT * len(SCREEN_RANGES)
    print(f"\n[4] Virtual screen of {n_screen} ligands in the (K_catalytic, CKI_TM) plane")
    rng = np.random.default_rng(SEED_SCREEN)
    screen = []
    lid = 0
    for q, (r_cat, r_gate, r_chan) in SCREEN_RANGES.items():
        for _ in range(M_PER_QUADRANT):
            lid += 1
            spec = M.LigandSpec(f"L{lid:03d}", float(rng.uniform(*r_cat)), float(rng.uniform(*r_gate)),
                                float(rng.uniform(*r_chan)), note=f"sampled for quadrant {q}")
            r = M.metrics(spec, apo)
            # a conventional cell-based gamma-secretase activity assay sees the catalytic block only
            assay_hit = int(r["K_catalytic"] > M.K_CAT_MAX)
            r |= {"quadrant_sampled": q, "correct": int(r["quadrant"] == q), "assay_hit": assay_hit,
                  "is_target_class": int(q == "II")}
            screen.append(r)
    write_csv(CSV_SCREEN, screen)

    print(f"  {'quadrant':>8} {'n':>4} {'K_catalytic range':>25} {'CKI_TM range':>20} "
          f"{'Notch retention (%)':>21} {'R_42/40 range':>19}")
    for q in SCREEN_RANGES:
        rs = [r for r in screen if r["quadrant_sampled"] == q]
        kc = [r["K_catalytic"] for r in rs]
        tm = [r["CKI_TM"] for r in rs]
        nr = [r["notch_retention"] for r in rs]
        rr = [r["R_42_40"] for r in rs]
        print(f"  {q:>8} {len(rs):>4} {min(kc):>10.2e} - {max(kc):>10.2e} {min(tm):>8.3f} - {max(tm):>8.3f} "
              f"{100 * min(nr):>9.2f} - {100 * max(nr):>8.2f} {min(rr):>8.4f} - {max(rr):>8.4f}")

    acc = float(np.mean([r["correct"] for r in screen]))
    q2 = [r for r in screen if r["quadrant_sampled"] == "II"]
    recall_assay = float(np.mean([r["assay_hit"] for r in q2]))
    hits = [r for r in screen if r["assay_hit"]]
    hits_liable = all(r["quadrant"] in ("I", "IV") for r in hits)
    n_safe_potent = sum(1 for r in screen if r["pass_notch"] and r["pass_cki"])
    print(f"  sampled strength ranges realise the intended quadrant: {100 * acc:.1f}% "
          f"({sum(r['correct'] for r in screen)}/{n_screen})")
    print(f"  ligands satisfying both design constraints: {n_safe_potent}/{n_screen}, "
          f"all of quadrant II ({len(q2)} sampled)")
    print(f"  activity-assay-only read-out (hit iff K_catalytic > {M.K_CAT_MAX}): {len(hits)} hits, "
          f"quadrant-II recall {100 * recall_assay:.0f}%, every hit in quadrant I or IV: {hits_liable}")
    passed = acc == 1.0 and recall_assay == 0.0 and hits_liable and n_safe_potent == len(q2)
    ok_all &= passed
    print(f"  Acceptance: the sampling realises all four quadrants, quadrant II is invisible to the activity "
          f"assay, and every assay hit carries a Notch liability  {'PASS' if passed else 'FAIL'}")

    # ---------- 5. familial AD: gate destabilisation and allosteric rescue ----------
    print("\n[5] Familial AD: PSEN1 gate destabilisation and allosteric rescue")
    fad_rows = []
    for bg in M.BACKGROUNDS:
        for spec in (M.APO,) + M.COMPOUNDS:
            r = M.metrics(spec, apo, bg)
            r |= {"rescue_occupancy": None}
            if bg is not M.WILD_TYPE and spec is not M.APO:
                r["rescue_occupancy"] = M.rescue_occupancy(spec, bg, apo=apo)
            fad_rows.append(r)
    write_csv(CSV_FAD, [{k: ("" if v is None else v) for k, v in r.items()} for r in fad_rows])
    fad = {(r["background"], r["ligand"]): r for r in fad_rows}

    print(f"  {'background':>11} {'ligand':>15} {'CKI_TM':>8} {'Gamma_bg':>9} {'Gamma_L':>9} "
          f"{'Abeta38/42':>11} {'R_42/40':>9} {'vs FAD':>7} {'Notch ret.':>11} {'m* for R<0.12':>14}")
    for bg in M.BACKGROUNDS:
        base = fad[(bg.name, "apo")]["R_42_40_branch"]
        for spec in (M.APO,) + M.COMPOUNDS:
            r = fad[(bg.name, spec.name)]
            mstar = r["rescue_occupancy"]
            ms = f"{mstar:.3f}" if mstar else ("--" if spec is M.APO else "> 4")
            print(f"  {bg.name:>11} {r['ligand']:>15} {r['CKI_TM']:>8.4f} {r['Gamma_bg']:>9.4f} "
                  f"{r['Gamma_ligand']:>9.4f} {r['ratio_38_42']:>11.4f} {r['R_42_40_branch']:>9.5f} "
                  f"{base / r['R_42_40_branch']:>6.2f}x {100 * r['notch_retention']:>10.2f}% {ms:>14}")

    fad_apo = [fad[(b.name, "apo")] for b in M.FAD_BACKGROUNDS]
    passed = all(M.R_FAD_LOW <= r["R_42_40_branch"] <= M.R_FAD_HIGH for r in fad_apo) \
        and all(r["zeta_bg"] < 0 and r["ratio_38_42"] < M.RHO0 for r in fad_apo)
    ok_all &= passed
    print(f"\n  Acceptance: the untreated FAD alleles loosen the processive path (zeta = -1), lower "
          f"Abeta38/Abeta42 below the wild-type {M.RHO0:.1f}, and raise R_42/40 into the reported range "
          f"[{M.R_FAD_LOW}, {M.R_FAD_HIGH}]: "
          + ", ".join(f"{r['background']} {r['R_42_40_branch']:.4f}" for r in fad_apo)
          + f"  {'PASS' if passed else 'FAIL'}")

    resc = [fad[(b.name, c)] for b in M.FAD_BACKGROUNDS for c in ("KMS-AD-309", "E2012")]
    passed = all(r["R_42_40_branch"] < M.R0_42_40 and r["notch_retention"] > M.NOTCH_RESCUE_MIN
                 and r["Gamma"] > 0 for r in resc)
    ok_all &= passed
    print(f"  Acceptance: both quadrant-II modulators reverse the sign of the clamp on both alleles and "
          f"bring R_42/40 below the wild-type baseline {M.R0_42_40} while keeping Notch retention above "
          f"{100 * M.NOTCH_RESCUE_MIN:.0f}%: "
          + ", ".join(f"{r['background']}+{r['ligand']} {r['R_42_40_branch']:.4f} "
                      f"({100 * r['notch_retention']:.2f}%)" for r in resc)
          + f"  {'PASS' if passed else 'FAIL'}")

    m_stars = [r["rescue_occupancy"] for r in resc]
    passed = all(m is not None and m < 1.5 for m in m_stars)
    ok_all &= passed
    print(f"  Acceptance: the stricter target R_42/40 < {M.R_RESCUE_MAX} is reached at an occupancy below "
          f"1.5x nominal in every case: "
          + ", ".join(f"{r['background']}+{r['ligand']} {m:.3f}x" for r, m in zip(resc, m_stars))
          + f"  {'PASS' if passed else 'FAIL'}")

    failers = [fad[(b.name, c)] for b in M.FAD_BACKGROUNDS for c in ("R-Flurbiprofen", "Semagacestat")]
    passed = all(r["Gamma"] < 0 and r["rescue_occupancy"] is None for r in failers) \
        and all(fad[(b.name, "Semagacestat")]["notch_retention"] < M.NOTCH_MIN for b in M.FAD_BACKGROUNDS)
    ok_all &= passed
    print(f"  Acceptance: neither the weak first-generation GSM nor the inhibitor rescues either allele at "
          f"any occupancy up to 4x, and the inhibitor additionally destroys Notch signalling: "
          + ", ".join(f"{r['background']}+{r['ligand']} {r['R_42_40_branch']:.4f} "
                      f"({100 * r['notch_retention']:.1f}%)" for r in failers)
          + f"  {'PASS' if passed else 'FAIL'}")

    # ---------- 6. figures ----------
    print("\n[6] Figures")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    axA, axB, axC, axD = axes.ravel()
    fig2, (axE, axF) = plt.subplots(1, 2, figsize=(11.2, 4.9))

    # A: phase diagram
    short = {"I": "non-selective allosteric inhibitor", "II": "Notch-sparing GSM (target)",
             "III": "silent / weak GSM", "IV": "orthosteric GSI"}
    for q in ("I", "IV", "III", "II"):
        rs = [r for r in screen if r["quadrant"] == q]
        axA.scatter([r["K_catalytic"] for r in rs], [r["CKI_TM"] for r in rs], s=18, color=Q_COLOR[q],
                    edgecolor=SURFACE, linewidth=0.5, zorder=3, label=f"{q}  {short[q]}")
    axA.axvline(M.K_CAT_MAX, color=AXIS, lw=1.0, ls="--")
    axA.axhline(M.CKI_TM_MIN, color=AXIS, lw=1.0, ls="--")
    # the quadrant-II cloud is dense, so the two labels that sit inside it get a leader and a
    # light background box rather than a bare offset
    offs_a = {"Semagacestat": (-10, 10, "right", False), "Avagacestat": (-10, 10, "right", False),
              "R-Flurbiprofen": (-12, 15, "right", True), "E2012": (-14, -20, "right", True),
              "KMS-AD-309": (10, -10, "left", False)}
    bbox_a = dict(boxstyle="round,pad=0.22", facecolor=SURFACE, edgecolor="none", alpha=0.88)
    for r in rows:
        dx, dy, ha, boxed = offs_a[r["ligand"]]
        axA.scatter([r["K_catalytic"]], [r["CKI_TM"]], s=150, marker="*", color=C_COLOR[r["ligand"]],
                    edgecolor=INK, linewidth=0.7, zorder=5)
        axA.annotate(r["ligand"], (r["K_catalytic"], r["CKI_TM"]), textcoords="offset points",
                     xytext=(dx, dy), ha=ha, fontsize=8, color=INK, fontweight="bold", zorder=6,
                     bbox=bbox_a if boxed else None,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=7)
                     if boxed else None)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(1e-6, 3.0)
    axA.set_ylim(0.05, 14)
    axA.set_xlabel(r"$K_{\mathrm{catalytic}}(L)$   (Asp257-Asp385 dyad and S3 register, 30 dof)")
    axA.set_ylabel(r"$\mathrm{CKI}_{\mathrm{TM}}(L)$   (TM3, TM6a, PAL machinery, 270 dof)")
    axA.set_title("A  Notch-safety phase diagram; target = quadrant II")
    for q, (xq, yq) in (("II", (1.6e-6, 11.0)), ("I", (1.6, 11.0)), ("III", (1.6e-6, 0.062)),
                        ("IV", (1.6, 0.062))):
        axA.text(xq, yq, q, ha="left", va="center", fontsize=11, color=Q_COLOR[q], fontweight="bold")
    leg = axA.legend(loc="lower left", fontsize=7, frameon=True, facecolor=SURFACE, edgecolor=GRID_C,
                     borderpad=0.5, bbox_to_anchor=(0.22, 0.02))
    leg.get_frame().set_alpha(0.92)
    tidy(axA)

    # B: Notch retention
    k = np.logspace(-6, 0.3, 400)
    axB.plot(k, [M.notch_retention(x) for x in k], color=INK_2, lw=1.6, zorder=2,
             label=r"$\exp(-3 K_{\mathrm{catalytic}})$")
    axB.axhline(M.NOTCH_MIN, color=GREEN, lw=1.0, ls="--")
    axB.axvline(M.K_CAT_MAX, color=AXIS, lw=1.0, ls="--")
    axB.scatter([r["K_catalytic"] for r in screen], [r["notch_retention"] for r in screen], s=12,
                color=[Q_COLOR[r["quadrant"]] for r in screen], alpha=0.75, zorder=3)
    # labels are parked in the empty lower-left region and tied to their markers by thin leaders,
    # so that none of them sits on the 92 % floor, the K_catalytic threshold or the curve
    pos_b = {"Semagacestat": (0.020, 0.235, "left"), "Avagacestat": (0.020, 0.405, "left"),
             "R-Flurbiprofen": (2.2e-6, 0.760, "left"), "E2012": (2.2e-6, 0.585, "left"),
             "KMS-AD-309": (2.4e-4, 0.420, "left")}
    for r in rows:
        xl, yl, ha = pos_b[r["ligand"]]
        axB.scatter([r["K_catalytic"]], [r["notch_retention"]], s=150, marker="*",
                    color=C_COLOR[r["ligand"]], edgecolor=INK, linewidth=0.7, zorder=5)
        axB.annotate(r["ligand"], xy=(r["K_catalytic"], r["notch_retention"]), xytext=(xl, yl),
                     ha=ha, va="center", fontsize=8, color=INK, fontweight="bold", zorder=6,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2, shrinkB=6))
    axB.set_xscale("log")
    axB.set_xlabel(r"$K_{\mathrm{catalytic}}(L)$")
    axB.set_ylabel("Notch S3 signal retention")
    axB.set_ylim(0, 1.05)
    axB.set_title(r"B  Notch constraint: $K_{\mathrm{catalytic}} < 0.025 \Rightarrow$ retention $> 92\%$")
    axB.text(1.3e-6, M.NOTCH_MIN - 0.045, "92 % floor", fontsize=8, color=GREEN)
    axB.legend(loc="lower left", fontsize=8)
    tidy(axB)

    # C: cone geodesic
    labels = ["apo"] + [r["ligand"] for r in rows]
    w_sums = [3 * M.W_STEP] + [r["w_sum"] for r in rows]
    d_cones = [M.cone_equal_weight(M.W_STEP, 3)] + [r["d_cone"] for r in rows]
    sig = [0.0] + [r["sigma_sync"] for r in rows]
    x = np.arange(len(labels))
    axC.bar(x - 0.19, w_sums, width=0.36, color=MUTED, label=r"sequential $\sum_j w_j$")
    axC.bar(x + 0.19, d_cones, width=0.36, color=AQUA, label=r"cone geodesic $d_L^{\mathrm{cone}}$")
    for xi, ws, dc, s in zip(x, w_sums, d_cones, sig):
        axC.annotate(f"{ws - dc:.3f} $k_BT$\n$\\sigma$ = {s:.2f}", (xi, ws),
                     textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=INK_2)
    axC.set_xticks(x)
    axC.set_xticklabels([l.replace("R-Flurbiprofen", "R-Flurbi.").replace("Semagacestat", "Semaga.")
                         .replace("Avagacestat", "Avaga.").replace("KMS-AD-309", "KMS-309")
                         for l in labels], fontsize=7.5)
    axC.set_ylabel(r"cost of the three-residue step  ($k_B T$)")
    axC.set_ylim(0, max(w_sums) * 1.45)
    axC.set_title(r"C  Cone geodesic on $\{0,1\}^3$:  $d_L^{\mathrm{cone}} = 2.284457\,w < 3w$")
    axC.legend(loc="upper left", fontsize=8)
    tidy(axC)

    # D: ratio law
    br = np.linspace(-0.2, max(r["CKI_cov"] for r in screen) + 0.3, 300)
    axD.plot(br, M.R0_42_40 * np.exp(-M.ETA_R * br), color=INK_2, lw=1.6, zorder=2,
             label=rf"$R_0 e^{{-\eta x}}$, $R_0 = {M.R0_42_40}$, $\eta = {M.ETA_R}$")
    axD.axhline(M.R0_42_40, color=AXIS, lw=1.0, ls="--")
    axD.scatter([r["CKI_cov"] - r["d_cone_eff"] for r in screen], [r["R_42_40"] for r in screen], s=12,
                color=[Q_COLOR[r["quadrant"]] for r in screen], alpha=0.75, zorder=3)
    offs_d = {"Semagacestat": (12, 10, "left"), "Avagacestat": (12, -20, "left"),
              "R-Flurbiprofen": (12, -6, "left"), "E2012": (-8, -22, "right"),
              "KMS-AD-309": (12, 8, "left")}
    for r in rows:
        xv = r["CKI_cov"] - r["d_cone_eff"]
        dx, dy, ha = offs_d[r["ligand"]]
        axD.scatter([xv], [r["R_42_40"]], s=150, marker="*", color=C_COLOR[r["ligand"]], edgecolor=INK,
                    linewidth=0.7, zorder=5)
        axD.annotate(f"{r['ligand']}\n{r['R_fold_reduction']:.2f}x", (xv, r["R_42_40"]),
                     textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8, color=INK,
                     fontweight="bold", zorder=6)
    axD.set_xlabel(r"$\mathrm{CKI}_{\mathrm{cov}}(L) - d_L^{\mathrm{cone,eff}}(L)$   (nats)")
    axD.set_ylabel(r"$R_{42/40}(L)$")
    axD.set_title(r"D  Pathogenic-ratio law; untreated $R_0$ dashed")
    axD.legend(loc="upper right", fontsize=8)
    tidy(axD)

    # E: the two clinical biomarkers, from the branching model
    gx = np.linspace(-4.0, 3.2, 400)
    curve = [M.branching(g, 0.0) for g in gx]
    axE.plot([c[1] for c in curve], [c[0] for c in curve], color=INK_2, lw=1.6, zorder=2,
             label=r"branching law, $\Delta\Delta G^{\ddagger} = 0$")
    axE.axvline(M.R0_42_40, color=AXIS, lw=1.0, ls="--")
    axE.axhline(M.RHO0, color=AXIS, lw=1.0, ls="--")
    offs_e = {"Semagacestat": (-10, 12, "right"), "Avagacestat": (-10, 10, "right"),
              "R-Flurbiprofen": (-10, -18, "right"), "E2012": (12, -14, "left"),
              "KMS-AD-309": (12, 4, "left")}
    for r in rows:
        dx, dy, ha = offs_e[r["ligand"]]
        axE.scatter([r["R_42_40_branch"]], [r["ratio_38_42"]], s=150, marker="*",
                    color=C_COLOR[r["ligand"]], edgecolor=INK, linewidth=0.7, zorder=5)
        axE.annotate(r["ligand"], (r["R_42_40_branch"], r["ratio_38_42"]), textcoords="offset points",
                     xytext=(dx, dy), ha=ha, fontsize=8, color=INK, fontweight="bold", zorder=6)
    offs_fad_e = {"PS1-L166P": (-12, 14, "right"), "PS1-E280A": (10, -14, "left")}
    for b in M.FAD_BACKGROUNDS:
        r = fad[(b.name, "apo")]
        dx, dy, ha = offs_fad_e[b.name]
        axE.scatter([r["R_42_40_branch"]], [r["ratio_38_42"]], s=110, marker="X",
                    color=BG_COLOR[b.name], edgecolor=INK, linewidth=0.7, zorder=5)
        axE.annotate(f"{b.name}\nuntreated", (r["R_42_40_branch"], r["ratio_38_42"]),
                     textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=7.5, color=INK,
                     fontweight="bold", zorder=6)
    axE.set_xscale("log")
    axE.set_yscale("log")
    axE.set_xlabel(r"$R_{42/40}$   (pathogenic ratio, falls with a modulator)")
    axE.set_ylabel(r"$\mathrm{A}\beta_{38}/\mathrm{A}\beta_{42}$   (rises with a modulator)")
    axE.set_title("A  The two clinical biomarkers move together")
    axE.legend(loc="upper right", fontsize=8)
    tidy(axE)

    # F: familial AD and its rescue
    f_labels, f_vals, f_cols = [], [], []
    for b in M.FAD_BACKGROUNDS:
        f_labels.append(f"{b.name}\nuntreated")
        f_vals.append(fad[(b.name, "apo")]["R_42_40_branch"])
        f_cols.append(BG_COLOR[b.name])
        for c in ("Semagacestat", "R-Flurbiprofen", "E2012", "KMS-AD-309"):
            f_labels.append(f"+{c.replace('R-Flurbiprofen', 'R-Flurbi.').replace('Semagacestat', 'Semaga.')}")
            f_vals.append(fad[(b.name, c)]["R_42_40_branch"])
            f_cols.append(C_COLOR[c])
    xf = np.arange(len(f_vals))
    axF.bar(xf, f_vals, color=f_cols, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    axF.axhline(M.R0_42_40, color=GREEN, lw=1.1, ls="--", zorder=4)
    axF.axhline(M.R_RESCUE_MAX, color=AQUA, lw=1.1, ls=":", zorder=4)
    axF.set_xlim(-2.5, len(f_vals) - 0.4)          # a clear strip on the left for the two rules
    axF.text(-2.35, M.R0_42_40 + 0.007, f"wild type\n{M.R0_42_40}", ha="left", va="bottom",
             fontsize=7.5, color=GREEN)
    axF.text(-2.35, M.R_RESCUE_MAX + 0.007, f"target\n{M.R_RESCUE_MAX}", ha="left", va="bottom",
             fontsize=7.5, color=AQUA)
    for xi, v in zip(xf, f_vals):
        axF.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center",
                     fontsize=6.5, color=INK_2)
    axF.set_xticks(xf)
    axF.set_xticklabels(f_labels, fontsize=6.5, rotation=45, ha="right")
    axF.set_ylabel(r"$R_{42/40}$")
    axF.set_ylim(0, max(f_vals) * 1.22)
    axF.set_title("B  Familial AD alleles and their allosteric rescue")
    tidy(axF)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    fig.savefig(FIG_PATH_PDF)
    plt.close(fig)
    fig2.tight_layout()
    fig2.savefig(FIG2_PATH, dpi=200)
    fig2.savefig(FIG2_PATH_PDF)
    plt.close(fig2)

    print("\nOutput files (relative to the repository root):")
    for pth in (CSV_COMPOUNDS, CSV_SCREEN, CSV_CONE, CSV_FAD, FIG_PATH, FIG_PATH_PDF,
                FIG2_PATH, FIG2_PATH_PDF, LOG_PATH):
        print("  ", os.path.relpath(pth, ROOT).replace(os.sep, "/"))
    print(f"\nruntime {time.time() - t0:.1f} s")
    print("OVERALL ACCEPTANCE:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def main() -> int:
    tee = Tee(LOG_PATH)
    old = sys.stdout
    sys.stdout = tee
    try:
        return run()
    finally:
        sys.stdout = old
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
