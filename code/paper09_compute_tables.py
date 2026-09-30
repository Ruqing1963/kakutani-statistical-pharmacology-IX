#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper09_compute_tables.py
=========================
Analytic companion of Paper IX. Reads the output of exp09_ad_gamma_secretase_gsm.py
(data/exp09_*.csv, results/exp09_console_log.txt), recomputes every quantity quoted in the paper
and writes the table bodies.

    Table 1   the three-zone model: block sizes, stiffnesses, spectrum of K
    Table 2   Proposition 2.2: Schur-complement screening, scan of the catalytic coupling scale s
    Table 3   Theorem 3.1 and Proposition 3.3: the cone geodesic and the anisotropy family
    Table 4   the three reference compounds: indices, cone quantities and pharmacological read-outs
    Table 5   the 240-ligand screen by quadrant, and the activity assay
    checks    equal-weight closed form for k = 1..6; second-order exponent of the Schur screening;
              strict monotonicity of d_cone along the anisotropy family on a fine grid; evenness
              of that family; the two-route consistency quotient; saturation of S_Notch
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp09_lib as M  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA, RESULTS = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
LOG = open(os.path.join(RESULTS, "exp09_console_log.txt"), encoding="utf-8").read()

COUPLING_SCALES = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)
N_FIT_SCALES = 4                   # the exponents are asymptotic: fit them on the smallest scales
N_MONO = 2001                      # grid for the monotonicity check on a in [0, 0.999]
A_MAX = 0.999


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def m(x, d=3, sign=False):
    """A number in math mode, signs included, as the series' tables require."""
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "--"
    if round(x, d) == 0:
        x = abs(x)
    return f"${x:+.{d}f}$" if sign else f"${x:.{d}f}$"


def e(x, d=3):
    """Scientific notation in math mode."""
    if x == 0:
        return "$0$"
    k = int(math.floor(math.log10(abs(x))))
    return f"${x / 10 ** k:.{d}f} \\times 10^{{{k}}}$"


def grab(pattern: str, text: str = LOG, cast=float):
    mm = re.search(pattern, text)
    if not mm:
        raise ValueError(pattern)
    return [cast(g) for g in mm.groups()]


def write_rows(name: str, rows: list[str]) -> None:
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


# ----------------------------------------------------------------------------
# Schur complement of the catalytic block
# ----------------------------------------------------------------------------
def precision_cat(K: np.ndarray) -> np.ndarray:
    """Sigma_CC^{-1} = K_CC - K_CM K_MM^{-1} K_MC, the Schur complement of the machinery block."""
    C, S = M.SL_CAT, M.SL_MACHINERY
    return K[C, C] - K[C, S] @ np.linalg.solve(K[S, S], K[S, C])


def coupling_scan(spec: M.LigandSpec) -> list[dict]:
    """Scale the catalytic couplings by s and measure what an M-supported perturbation does to the
    catalytic block. Proposition 2.2 predicts both the precision change and the index to be O(s^2)."""
    chan0, gate0 = M.K_CAT_CHAN, M.K_CAT_GATE
    out = []
    try:
        for s in COUPLING_SCALES:
            M.K_CAT_CHAN, M.K_CAT_GATE = s * chan0, s * gate0
            apo, holo = M.ensemble(M.APO), M.ensemble(spec)
            P0, P1 = precision_cat(apo.K), precision_cat(holo.K)
            r = M.metrics(spec, apo)
            out.append({"s": s, "K_cat_chan": s * chan0, "K_cat_gate": s * gate0,
                        "dP_fro": float(np.linalg.norm(P1 - P0)),
                        "dP_rel": float(np.linalg.norm(P1 - P0) / np.linalg.norm(P0)),
                        "leakage": r["leakage_cat"], "K_catalytic": r["K_catalytic"],
                        "K_cat_mean": r["K_cat_mean"], "K_cat_cov": r["K_cat_cov"],
                        "notch_retention": r["notch_retention"], "CKI_TM": r["CKI_TM"]})
    finally:
        M.K_CAT_CHAN, M.K_CAT_GATE = chan0, gate0
    return out


def main() -> int:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    summary: dict = {}
    out("=" * 96)
    out("Paper IX companion: tables and checks")
    out("=" * 96)

    compounds = read_csv("exp09_compound_comparison.csv")
    screen = read_csv("exp09_quadrant_screen.csv")
    cone = read_csv("exp09_cone_scan.csv")
    fad_rows = read_csv("exp09_fad_rescue.csv")
    by = {r["ligand"]: r for r in compounds}
    fad = {(r["background"], r["ligand"]): r for r in fad_rows}

    # ---------------- Table 1: the model ----------------
    out("\n[Table 1] Three-zone Gaussian elastic model of Presenilin-1")
    lo, hi, cond = grab(r"eigenvalues in \[([0-9.]+), ([0-9.]+)\], condition number ([0-9.]+)")
    apo = M.ensemble(M.APO)
    ev = np.linalg.eigvalsh(apo.K)
    assert abs(ev[0] - lo) < 5e-4 and abs(ev[-1] - hi) < 5e-4, "log disagrees with a fresh build"
    blocks = (("catalytic $\\mathcal{C}$", M.CAT_SEGMENTS, M.N_CAT, M.K_PATH["cat"]),
              ("gate $\\mathcal{G}$", M.GATE_SEGMENTS, M.N_GATE, M.K_PATH["gate"]),
              ("channel $\\mathcal{H}$", M.CHAN_SEGMENTS, M.N_CHAN, M.K_PATH["chan"]))
    t1 = []
    for label, segs, n, kp in blocks:
        names = ", ".join(f"{s.replace('_', ' ')} ({k})" for s, k in segs)
        if len(segs) > 3:      # the nine channel helices do not fit in a table cell one by one
            # each size in its own math group, so the cell can be broken across lines
            tex_names = (f"{segs[0][0]}--{segs[-1][0]} "
                         f"({', '.join(f'${k}$' for _, k in segs)})")
        else:
            tex_names = ", ".join(f"{s.replace('_', ' ')} (${k}$)" for s, k in segs)
        t1.append(f"{label} & ${n}$ & {tex_names} & ${kp:.1f}$ \\\\")
        out(f"  {label:>24}: {n:>3} dof, path stiffness {kp:.1f}, segments {names}")
    out(f"  K = graph Laplacian + {M.T_TETHER} I: spectrum [{ev[0]:.4f}, {ev[-1]:.4f}], condition number "
        f"{ev[-1] / ev[0]:.2f}, dim ker = {int(np.sum(ev < 1e-10))}")
    out(f"  inter-block couplings: gate-channel {M.K_GATE_CHAN}, channel packing {M.K_PACK}, "
        f"catalytic-channel {M.K_CAT_CHAN}, catalytic-gate {M.K_CAT_GATE}")
    write_rows("paper09_table1_rows.tex", t1)
    summary["model"] = {"lambda_min": ev[0], "lambda_max": ev[-1], "cond": ev[-1] / ev[0],
                        "dim_ker": int(np.sum(ev < 1e-10)), "n": [M.N_CAT, M.N_GATE, M.N_CHAN]}

    # ---------------- Table 2: Schur screening ----------------
    out("\n[Table 2] Proposition 2.2: Schur-complement screening of the catalytic block")
    allo = M.LigandSpec("allosteric-only", 0.0, 1.0, 1.0)
    scan = coupling_scan(allo)
    out(f"  perturbation: {allo.name} (s_cat = 0, s_gate = s_chan = 1), supported on "
        f"$\\mathcal{{M}} = \\mathcal{{G}} \\cup \\mathcal{{H}}$")
    out(f"  {'s':>8} {'K_cat-chan':>11} {'||dP||_F':>11} {'||dP||/||P||':>13} {'leakage':>10} "
        f"{'K_cat mean':>12} {'K_cat cov':>12} {'CKI_TM':>8}")
    t2 = []
    for r in scan:
        out(f"  {r['s']:>8.5f} {r['K_cat_chan']:>11.5f} {r['dP_fro']:>11.3e} {r['dP_rel']:>13.3e} "
            f"{r['leakage']:>10.6f} {r['K_cat_mean']:>12.3e} {r['K_cat_cov']:>12.3e} {r['CKI_TM']:>8.4f}")
        t2.append(f"{m(r['s'], 5)} & {m(r['K_cat_chan'], 5)} & {e(r['dP_fro'], 2)} & {e(r['leakage'], 2)} & "
                  f"{e(r['K_cat_mean'], 2)} & {e(r['K_cat_cov'], 2)} & {m(r['CKI_TM'], 4)} \\\\")
    s_arr = np.array([r["s"] for r in scan])

    def slope(key: str) -> float:
        """Asymptotic log-log exponent, fitted on the N_FIT_SCALES smallest couplings."""
        y = np.log([r[key] for r in scan[:N_FIT_SCALES]])
        return float(np.polyfit(np.log(s_arr[:N_FIT_SCALES]), y, 1)[0])

    def local(key: str) -> list[float]:
        y = np.log([r[key] for r in scan])
        return list(np.diff(y) / np.diff(np.log(s_arr)))

    slope_P, slope_leak = slope("dP_fro"), slope("leakage")
    slope_mean, slope_cov = slope("K_cat_mean"), slope("K_cat_cov")
    cki_spread = max(r["CKI_TM"] for r in scan) / min(r["CKI_TM"] for r in scan)
    out(f"  asymptotic exponents (fitted on s <= {s_arr[N_FIT_SCALES - 1]}): ||dP||_F {slope_P:.4f}, "
        f"leakage {slope_leak:.4f}, mean term {slope_mean:.4f}, covariance term {slope_cov:.4f}")
    out(f"  Proposition 2.2 predicts 2, 2, 2 and 4; successive local exponents of ||dP||_F: "
        + ", ".join(f"{v:.3f}" for v in local("dP_fro")))
    out(f"  successive local exponents of the covariance term: "
        + ", ".join(f"{v:.3f}" for v in local("K_cat_cov")))
    out(f"  CKI_TM is untouched by the screening: it varies by a factor {cki_spread:.4f} over a "
        f"64-fold range of s")
    write_rows("paper09_table2_rows.tex", t2)
    summary["schur"] = {"scan": scan, "slope_dP": slope_P, "slope_leakage": slope_leak,
                        "slope_mean": slope_mean, "slope_cov": slope_cov, "cki_spread": cki_spread,
                        "local_dP": local("dP_fro"), "local_cov": local("K_cat_cov")}

    # ---------------- Table 3: cone geodesic ----------------
    out("\n[Table 3] Theorem 3.1 and Proposition 3.3: the cone geodesic on {0,1}^3")
    err_k = {}
    for k in range(1, 7):
        num, ref = M.cone_geodesic([M.W_STEP] * k), M.cone_equal_weight(M.W_STEP, k)
        err_k[k] = abs(num - ref)
    out("  equal-weight closed form d_k = w sum_{j<=k} 1/sqrt(j), k = 1..6, max error "
        f"{max(err_k.values()):.2e}")
    out(f"  k = 3: d_cone = {M.cone_geodesic([M.W_STEP] * 3):.9f} kT = {M.cone_equal_weight(1.0, 3):.6f} w, "
        f"sequential 3w = {3 * M.W_STEP:.4f} kT, relative {M.cone_equal_weight(1.0, 3) / 3:.6f}")
    t3 = []
    for r in cone:
        t3.append(f"{m(r['anisotropy'], 1)} & {m(r['w_TM3'], 4)} & {m(r['w_TM6a'], 4)} & {m(r['w_PAL'], 4)} & "
                  f"{m(r['d_cone'], 5)} & {m(r['d_over_w_sum'], 6)} & {m(r['discount_kT'], 5)} \\\\")
    write_rows("paper09_table3_rows.tex", t3)

    a_grid = np.linspace(0.0, A_MAX, N_MONO)
    wbar = M.W_STEP
    d_grid = np.array([M.cone_geodesic([wbar * (1 + a), wbar, wbar * (1 - a)]) for a in a_grid])
    inc = np.diff(d_grid)
    even_err = max(abs(M.cone_geodesic([wbar * (1 + a), wbar, wbar * (1 - a)])
                       - M.cone_geodesic([wbar * (1 - a), wbar, wbar * (1 + a)]))
                   for a in (0.1, 0.3, 0.5, 0.7, 0.9))
    out(f"  monotonicity on {N_MONO} points a in [0, {A_MAX}]: min increment {inc.min():.3e} "
        f"(> 0: {bool(inc.min() > 0)}), d_cone rises from {d_grid[0]:.6f} to {d_grid[-1]:.6f} kT")
    out(f"  evenness d_cone(w(a)) = d_cone(w(-a)) to {even_err:.2e} (the family is a permutation of itself), "
        f"so a = 0 is a critical point")
    out(f"  bounds on the 240 screened ligands and the three compounds: max_j w_j <= d_cone < sum_j w_j")
    bound_ok = all(max(r["w_TM3"], r["w_TM6a"], r["w_PAL"]) - 1e-12 <= r["d_cone"] < r["w_sum"]
                   for r in compounds + screen)
    out(f"    holds in {len(compounds) + len(screen)}/{len(compounds) + len(screen)} cases: {bound_ok}")
    slack_family = min(M.cone_slack([wbar * (1 + a), wbar, wbar * (1 - a)]) for a in a_grid[::10])
    slack_ligand = min(M.cone_slack([r["w_TM3"], r["w_TM6a"], r["w_PAL"]]) for r in compounds + screen)
    out(f"  well-posedness margin of the recursion (Theorem 3.1 hypothesis): min over the anisotropy "
        f"family {slack_family:.6f}, min over all 243 ligands {slack_ligand:.6f} (both >= 0)")
    summary["cone"] = {"err_k": err_k, "min_increment": float(inc.min()), "even_err": even_err,
                       "d_a0": float(d_grid[0]), "d_amax": float(d_grid[-1]), "bounds_hold": bound_ok,
                       "slack_family": slack_family, "slack_ligand": slack_ligand}

    # ---------------- Table 4: reference compounds ----------------
    out(f"\n[Table 4] The {len(compounds)} reference compounds")
    t4a, t4b, t4c = [], [], []
    for spec in M.COMPOUNDS:
        name = spec.name
        r = by[name]
        out(f"  {name:>14}: K_cat {r['K_catalytic']:.4e}, retention {100 * r['notch_retention']:.2f}%, "
            f"CKI_TM {r['CKI_TM']:.4f} (rot {r['CKI_TM_rot']:.4f} = {100 * r['CKI_TM_rot'] / r['CKI_cov']:.2f}% "
            f"of the covariance term), S_Notch {r['S_Notch']:.4g}, quadrant {r['quadrant']}")
        out(f"  {'':>14}  sum w {r['w_sum']:.4f}, d_cone {r['d_cone']:.4f}, sigma {r['sigma_sync']:.4f}, "
            f"ddG {r['ddG_barrier_kcal']:.4f} kcal/mol, k-factor {r['rate_factor_42_38']:.3f}, "
            f"R_42/40 {r['R_42_40']:.5f} ({r['R_fold_reduction']:.2f}x), leakage {r['leakage_cat']:.4f}")
        t4a.append(f"{name} & {m(r['s_cat'], 3)} & {m(r['s_gate'], 2)} & {m(r['s_chan'], 2)} & "
                   f"{e(r['K_catalytic'])} & {m(100 * r['notch_retention'], 2)} & {m(r['CKI_TM'], 4)} & "
                   f"{m(r['CKI_TM_rot'], 4)} & ${r['S_Notch']:.4g}$ & {r['quadrant']} \\\\")
        t4b.append(f"{name} & {m(r['w_TM3'], 4)} & {m(r['w_TM6a'], 4)} & {m(r['w_PAL'], 4)} & "
                   f"{m(r['w_sum'], 4)} & {m(r['d_cone'], 4)} & {m(r['sigma_sync'], 4)} & "
                   f"{m(r['ddG_barrier_kcal'], 4)} & {m(r['rate_factor_42_38'], 3)} & "
                   f"{m(r['R_42_40'], 5)} & {m(r['R_fold_reduction'], 3)} \\\\")
        t4c.append(f"{name} & {m(r['Gamma'], 4, True)} & {m(r['ddG_barrier_kcal'], 4, True)} & "
                   f"{m(r['ratio_38_42'], 4)} & {m(r['ratio_38_42_fold'], 3)} & "
                   f"{m(r['R_42_40_branch'], 5)} & {m(r['R_branch_fold'], 3)} & "
                   f"{m(100 * abs(r['R_branch_fold'] / r['R_fold_reduction'] - 1), 2)} \\\\")
    write_rows("paper09_table4a_rows.tex", t4a)
    write_rows("paper09_table4b_rows.tex", t4b)
    write_rows("paper09_table4c_rows.tex", t4c)

    lead = by["KMS-AD-309"]
    rot_share = lead["CKI_TM_rot"] / lead["CKI_cov"]
    max_disc = max(abs(r["R_branch_fold"] / r["R_fold_reduction"] - 1) for r in compounds)
    apo_wt = fad[("wild type", "apo")]
    out(f"  rotation share of the lead's covariance term: {100 * rot_share:.2f}%")
    out(f"  branching law: rho_0 = {M.RHO0}, eta_clamp = {M.ETA_CLAMP}; at wild-type apo it returns "
        f"rho = {apo_wt['ratio_38_42']:.6f} and R = {apo_wt['R_42_40_branch']:.6f}")
    out(f"  largest disagreement with the design-brief law over the {len(compounds)} compounds: "
        f"{100 * max_disc:.2f}% ({max(compounds, key=lambda r: abs(r['R_branch_fold'] / r['R_fold_reduction'] - 1))['ligand']})")
    out(f"  the two clinical biomarkers: "
        + ", ".join(f"{r['ligand']} {r['R_branch_fold']:.2f}x down / {r['ratio_38_42_fold']:.2f}x up"
                    for r in compounds))
    summary["compounds"] = {r["ligand"]: r for r in compounds}
    summary["branching"] = {"rho0": M.RHO0, "eta_clamp": M.ETA_CLAMP, "max_disagreement": max_disc,
                            "rot_share": rot_share}

    # ---------------- Table 5: the screen ----------------
    out("\n[Table 5] Virtual screen of 240 ligands and the activity assay")
    t5 = []
    for q in ("I", "II", "III", "IV"):
        rs = [r for r in screen if r["quadrant_sampled"] == q]
        kc = [r["K_catalytic"] for r in rs]
        tm = [r["CKI_TM"] for r in rs]
        nr = [100 * r["notch_retention"] for r in rs]
        rr = [r["R_42_40"] for r in rs]
        hits = sum(int(r["assay_hit"]) for r in rs)
        both = sum(1 for r in rs if r["pass_notch"] and r["pass_cki"])
        out(f"  {q:>4}: n {len(rs)}, K_cat [{min(kc):.2e}, {max(kc):.2e}], CKI_TM [{min(tm):.3f}, {max(tm):.3f}], "
            f"retention [{min(nr):.2f}, {max(nr):.2f}]%, R [{min(rr):.4f}, {max(rr):.4f}], "
            f"assay hits {hits}, both constraints {both}")
        t5.append(f"{q} & ${len(rs)}$ & {e(min(kc), 2)} & {e(max(kc), 2)} & {m(min(tm), 3)} & {m(max(tm), 3)} & "
                  f"{m(min(nr), 2)} & {m(max(nr), 2)} & ${hits}$ & ${both}$ \\\\")
    write_rows("paper09_table5_rows.tex", t5)

    n_hits = sum(int(r["assay_hit"]) for r in screen)
    q2 = [r for r in screen if r["quadrant_sampled"] == "II"]
    recall = sum(int(r["assay_hit"]) for r in q2) / len(q2)
    liable = all(r["quadrant"] in ("I", "IV") for r in screen if r["assay_hit"])
    n_both = sum(1 for r in screen if r["pass_notch"] and r["pass_cki"])
    out(f"  assay (hit iff K_catalytic > {M.K_CAT_MAX}): {n_hits}/{len(screen)} hits, quadrant-II recall "
        f"{100 * recall:.0f}% ({sum(int(r['assay_hit']) for r in q2)}/{len(q2)}), every hit in I or IV: {liable}")
    out(f"  ligands meeting both design constraints: {n_both}/{len(screen)}, all in quadrant II")
    sat = max(r["S_Notch"] * (M.EPS0 / r["CKI_TM"]) for r in screen + compounds)
    out(f"  saturation bound S_Notch <= CKI_TM / eps_0: max ratio S_Notch eps_0 / CKI_TM = {sat:.6f} (<= 1)")
    out(f"  R-Flurbiprofen scores S_Notch = {by['R-Flurbiprofen']['S_Notch']:.1f} with "
        f"CKI_TM = {by['R-Flurbiprofen']['CKI_TM']:.4f}, far below the threshold {M.CKI_TM_MIN}")
    summary["screen"] = {"n": len(screen), "hits": n_hits, "recall_II": recall, "all_hits_liable": liable,
                         "both_constraints": n_both, "saturation_max": sat}

    # ---------------- Table 6: familial AD and rescue ----------------
    out("\n[Table 6] Familial AD alleles and their allosteric rescue")
    t6 = []
    for bg in M.FAD_BACKGROUNDS:
        base = fad[(bg.name, "apo")]
        out(f"  {bg.name}: {bg.note}")
        out(f"    untreated: Gamma_bg {base['Gamma_bg']:+.4f}, ddG_bg {base['ddG_bg_kcal']:+.4f}, "
            f"Abeta38/42 {base['ratio_38_42']:.4f}, R_42/40 {base['R_42_40_branch']:.5f}")
        t6.append(f"{bg.name} & untreated & {m(base['Gamma'], 4, True)} & {m(base['ratio_38_42'], 4)} & "
                  f"{m(base['R_42_40_branch'], 5)} & $1.00$ & {m(100 * base['notch_retention'], 2)} & -- \\\\")
        for spec in M.COMPOUNDS:
            r = fad[(bg.name, spec.name)]
            mstar = r["rescue_occupancy"]
            ms = m(mstar, 3) if isinstance(mstar, float) else "$>4$"
            out(f"    + {spec.name:<15} Gamma {r['Gamma']:+.4f}, Abeta38/42 {r['ratio_38_42']:.4f}, "
                f"R {r['R_42_40_branch']:.5f} ({base['R_42_40_branch'] / r['R_42_40_branch']:.2f}x), "
                f"retention {100 * r['notch_retention']:.2f}%, m* {ms.strip('$')}")
            t6.append(f" & {spec.name} & {m(r['Gamma'], 4, True)} & {m(r['ratio_38_42'], 4)} & "
                      f"{m(r['R_42_40_branch'], 5)} & {m(base['R_42_40_branch'] / r['R_42_40_branch'], 2)} & "
                      f"{m(100 * r['notch_retention'], 2)} & {ms} \\\\")
    write_rows("paper09_table6_rows.tex", t6)

    fad_apo = [fad[(b.name, "apo")] for b in M.FAD_BACKGROUNDS]
    resc = [fad[(b.name, c)] for b in M.FAD_BACKGROUNDS for c in ("E2012", "KMS-AD-309")]
    fails = [fad[(b.name, c)] for b in M.FAD_BACKGROUNDS for c in ("R-Flurbiprofen", "Semagacestat")]
    additive = max(abs(r["Gamma"] - (r["Gamma_bg"] + r["Gamma_ligand"])) for r in fad_rows)
    bg_indep = max(
        abs(fad[(b.name, s.name)]["Gamma_bg"] - fad[(b.name, "apo")]["Gamma_bg"])
        for b in M.BACKGROUNDS for s in M.COMPOUNDS)
    out(f"  clamp additivity |Gamma - (Gamma_bg + Gamma_L)| <= {additive:.2e}; the background term is "
        f"ligand-independent to {bg_indep:.2e}")
    out(f"  rescue occupancies for the quadrant-II modulators: "
        + ", ".join(f"{r['background']}+{r['ligand']} {r['rescue_occupancy']:.3f}x" for r in resc))

    # the single-term clamp of Remark 4.x: everything against wild type, one global sign
    out("  counterfactual, the non-additive clamp (one term against wild type, one sign):")
    apo_ens = M.ensemble(M.APO, M.WILD_TYPE)
    naive = {}
    for bg in M.FAD_BACKGROUNDS:
        for cn in ("R-Flurbiprofen", "Semagacestat", "KMS-AD-309"):
            t = M._clamp_term(M.ensemble(M.COMPOUND[cn], bg), apo_ens)
            _, r_naive = M.branching(t["Gamma"], t["ddG"])
            naive[(bg.name, cn)] = {"zeta": t["zeta"], "Gamma": t["Gamma"], "R": r_naive}
            add = fad[(bg.name, cn)]
            out(f"    {bg.name:>11} + {cn:<15} naive zeta {t['zeta']:+.0f}, Gamma {t['Gamma']:+.4f}, "
                f"R {r_naive:.4f}   against additive Gamma {add['Gamma']:+.4f}, R {add['R_42_40_branch']:.4f}")
    naive_absurd = all(naive[(b.name, c)]["R"] < M.R0_42_40 for b in M.FAD_BACKGROUNDS
                       for c in ("R-Flurbiprofen", "Semagacestat"))
    out(f"    the non-additive clamp makes a failed modulator and an inhibitor both appear to cure both "
        f"alleles below the wild-type baseline: {naive_absurd}")
    summary["naive_clamp"] = {f"{k[0]}+{k[1]}": v for k, v in naive.items()}
    summary["fad"] = {"untreated": {r["background"]: r["R_42_40_branch"] for r in fad_apo},
                      "rescue": {f"{r['background']}+{r['ligand']}":
                                 {"R": r["R_42_40_branch"], "retention": r["notch_retention"],
                                  "m_star": r["rescue_occupancy"]} for r in resc},
                      "additivity": additive, "bg_independence": bg_indep}

    # ---------------- overall ----------------
    out("\n[Checks]")
    checks = {
        "closed form, k = 1..6, max error < 1e-12": max(err_k.values()) < 1e-12,
        "Schur exponent of ||dP||_F is 2 to within 0.05": abs(slope_P - 2.0) < 0.05,
        "Schur exponent of the leakage is 2 to within 0.05": abs(slope_leak - 2.0) < 0.05,
        "Schur exponent of the mean term is 2 to within 0.05": abs(slope_mean - 2.0) < 0.05,
        "Schur exponent of the covariance term is 4 to within 0.10": abs(slope_cov - 4.0) < 0.10,
        "CKI_TM varies by under 1 % over the coupling scan": cki_spread < 1.01,
        "d_cone strictly increasing along the anisotropy family": bool(inc.min() > 0),
        "the anisotropy family is even in a": even_err < 1e-12,
        f"max_j w_j <= d_cone < sum_j w_j on all {len(compounds) + len(screen)} ligands": bound_ok,
        "the recursion is well posed on the whole family and all ligands": min(slack_family, slack_ligand) >= 0,
        "the lead acts through rotation (> 90 % of the covariance term)": rot_share > 0.90,
        "branching law returns (rho_0, R_0) at wild-type apo": (
            abs(apo_wt["ratio_38_42"] - M.RHO0) < 1e-9 and abs(apo_wt["R_42_40_branch"] - M.R0_42_40) < 1e-9),
        "branching and design-brief laws agree to 10 % on all compounds": max_disc < 0.10,
        "the clamp is additive over background and ligand": additive < 1e-12,
        "the background term does not depend on the ligand": bg_indep < 1e-12,
        "both FAD alleles land in the reported R_42/40 range": all(
            M.R_FAD_LOW <= r["R_42_40_branch"] <= M.R_FAD_HIGH and r["zeta_bg"] < 0 for r in fad_apo),
        "both quadrant-II modulators rescue both alleles below wild type": all(
            r["Gamma"] > 0 and r["R_42_40_branch"] < M.R0_42_40
            and r["notch_retention"] > M.NOTCH_RESCUE_MIN for r in resc),
        "rescue to R < 0.12 needs under 1.5x occupancy": all(
            isinstance(r["rescue_occupancy"], float) and r["rescue_occupancy"] < 1.5 for r in resc),
        "neither the weak GSM nor the inhibitor rescues at any occupancy": all(
            not isinstance(r["rescue_occupancy"], float) for r in fails),
        "the non-additive clamp gives the absurd result of Remark 4.6": naive_absurd,
        "assay recall on quadrant II is 0": recall == 0.0,
        "every assay hit is Notch-liable": liable,
        "exactly the 60 quadrant-II ligands meet both constraints": n_both == len(q2) == 60,
        "S_Notch never exceeds CKI_TM / eps_0": sat <= 1.0,
    }
    for k, v in checks.items():
        out(f"  {'PASS' if v else 'FAIL':>4}  {k}")
    ok = all(checks.values())
    summary["checks"] = checks
    out(f"\nALL CHECKS {'PASS' if ok else 'FAILED'}")

    with open(os.path.join(RESULTS, "paper09_tables.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(RESULTS, "paper09_tables_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwritten: results/paper09_table{1,2,3,4a,4b,4c,5,6}_rows.tex, paper09_tables.json, "
          "paper09_tables_summary.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
