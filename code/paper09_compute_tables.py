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
    by = {r["ligand"]: r for r in compounds}

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
    out("\n[Table 4] The three reference compounds")
    t4a, t4b = [], []
    for name in ("Semagacestat", "Flurbiprofen", "KMS-AD-309"):
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
    write_rows("paper09_table4a_rows.tex", t4a)
    write_rows("paper09_table4b_rows.tex", t4b)

    lead = by["KMS-AD-309"]
    quotient = lead["R_fold_reduction"] / lead["rate_factor_42_38"]
    rot_share = lead["CKI_TM_rot"] / lead["CKI_cov"]
    out(f"  two routes to the same effect: ratio law {lead['R_fold_reduction']:.4f}x, cone-barrier rate "
        f"factor {lead['rate_factor_42_38']:.4f}x, quotient {quotient:.4f}")
    out(f"  rotation share of the lead's covariance term: {100 * rot_share:.2f}%")
    summary["compounds"] = {n: by[n] for n in ("Semagacestat", "Flurbiprofen", "KMS-AD-309")}
    summary["consistency"] = {"quotient": quotient, "rot_share": rot_share}

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
    out(f"  Flurbiprofen scores S_Notch = {by['Flurbiprofen']['S_Notch']:.1f} with "
        f"CKI_TM = {by['Flurbiprofen']['CKI_TM']:.4f}, far below the threshold {M.CKI_TM_MIN}")
    summary["screen"] = {"n": len(screen), "hits": n_hits, "recall_II": recall, "all_hits_liable": liable,
                         "both_constraints": n_both, "saturation_max": sat}

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
        "max_j w_j <= d_cone < sum_j w_j on all 243 ligands": bound_ok,
        "the recursion is well posed on the whole family and all ligands": min(slack_family, slack_ligand) >= 0,
        "two routes agree within a factor 2": 0.5 <= quotient <= 2.0,
        "the lead acts through rotation (> 90 % of the covariance term)": rot_share > 0.90,
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
    print("\nwritten: results/paper09_table{1,2,3,4a,4b,5}_rows.tex, paper09_tables.json, "
          "paper09_tables_summary.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
