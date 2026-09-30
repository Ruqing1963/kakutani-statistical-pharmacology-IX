"""Unit tests for the Paper IX model library. Run with `python -m pytest tests` or `python tests/test_exp09.py`."""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
import exp09_lib as M  # noqa: E402


def test_block_sizes_and_partition():
    assert (M.N_CAT, M.N_GATE, M.N_CHAN, M.N_DOF) == (30, 70, 200, 300)
    covered = np.zeros(M.N_DOF, int)
    for sl in M.SEG.values():
        covered[sl] += 1
    assert np.all(covered == 1), "the segments must partition the 300 coordinates"
    assert M.SL_MACHINERY.start == M.N_CAT and M.SL_MACHINERY.stop == M.N_DOF


def test_stiffness_is_positive_definite_with_tether_floor():
    for bg in M.BACKGROUNDS:
        for spec in (M.APO,) + M.COMPOUNDS + (M.LigandSpec("extreme", 1.0, 1.35, 1.35),):
            e = np.linalg.eigvalsh(M.stiffness(spec, bg))
            assert e[0] > 0, (bg.name, spec.name)
            # K = (graph Laplacian, positive semi-definite) + T_TETHER I, so lambda_min >= T_TETHER
            assert e[0] >= M.T_TETHER - 1e-9, (bg.name, spec.name, e[0])


def test_apo_has_zero_mean_and_symmetric_covariance():
    apo = M.ensemble(M.APO)
    assert np.allclose(apo.mu, 0.0)
    assert np.allclose(apo.Sigma, apo.Sigma.T, atol=1e-12)
    assert np.allclose(apo.Sigma @ apo.K, np.eye(M.N_DOF), atol=1e-8)


def test_cone_geodesic_equal_weights_closed_form():
    for w0 in (0.5, 1.0, 1.05, 3.7):
        for k in (1, 2, 3, 4, 5):
            num = M.cone_geodesic([w0] * k)
            ref = M.cone_equal_weight(w0, k)
            assert abs(num - ref) < 1e-12, (w0, k, num, ref)


def test_cone_geodesic_bounds():
    """max_j w_j <= d_cone < sum_j w_j: at least one coordinate must flip, and the synchronous
    path is strictly cheaper than the sequential one."""
    rng = np.random.default_rng(3)
    for _ in range(25):
        w = rng.uniform(0.2, 2.0, size=3)
        d = M.cone_geodesic(w)
        assert w.max() - 1e-12 <= d < w.sum()


def test_cone_geodesic_is_homogeneous():
    w = np.array([1.3, 0.8, 1.1])
    assert abs(M.cone_geodesic(2.5 * w) - 2.5 * M.cone_geodesic(w)) < 1e-12


def test_step_weights_are_w0_for_apo():
    apo = M.ensemble(M.APO)
    assert np.allclose(M.step_weights(apo, apo), M.W_STEP, atol=1e-10)


def test_notch_retention_at_the_design_threshold():
    r = M.notch_retention(M.K_CAT_MAX)
    assert abs(r - math.exp(-3 * 0.025)) < 1e-15
    assert r > M.NOTCH_MIN, "K_catalytic < 0.025 must give retention above 92 %"
    assert M.notch_retention(0.0) == 1.0


def test_synchrony_is_increasing_and_bounded():
    xs = [0.0, 0.1, 1.0, 3.2, 10.0]
    ss = [M.synchrony(x) for x in xs]
    assert ss[0] == 0.0
    assert all(a < b for a, b in zip(ss, ss[1:]))
    assert ss[-1] < 1.0


def test_quadrant_assignment():
    assert M.quadrant(1e-4, 5.0) == "II"
    assert M.quadrant(0.5, 5.0) == "I"
    assert M.quadrant(1e-4, 0.5) == "III"
    assert M.quadrant(0.5, 0.5) == "IV"


def test_apo_against_itself_is_the_zero_ligand():
    """A ligand with all three strengths zero reproduces the apo ensemble exactly, so every index
    vanishes and R_42/40 returns the untreated ratio."""
    m = M.metrics(M.APO)
    assert m["K_catalytic"] < 1e-12
    assert m["CKI_TM"] < 1e-12
    assert abs(m["R_42_40"] - M.R0_42_40) < 1e-12
    assert m["ddG_barrier_kcal"] == 0.0 or abs(m["ddG_barrier_kcal"]) < 1e-12


def test_reference_compounds_fall_in_the_expected_quadrants():
    apo = M.ensemble(M.APO)
    q = {s.name: M.metrics(s, apo) for s in M.COMPOUNDS}
    assert q["Semagacestat"]["quadrant"] == "IV"
    assert q["Semagacestat"]["notch_retention"] < M.NOTCH_MIN
    # Avagacestat was described as Notch-sparing; in the model it is not
    assert q["Avagacestat"]["quadrant"] == "IV"
    assert q["Avagacestat"]["K_catalytic"] > M.K_CAT_MAX
    assert q["Avagacestat"]["notch_retention"] < M.NOTCH_MIN
    assert q["R-Flurbiprofen"]["quadrant"] == "III"
    for name in ("E2012", "KMS-AD-309"):
        assert q[name]["quadrant"] == "II", name
        assert q[name]["K_catalytic"] < M.K_CAT_MAX
        assert q[name]["CKI_TM"] > M.CKI_TM_MIN
        assert q[name]["notch_retention"] > M.NOTCH_MIN
    assert q["KMS-AD-309"]["S_Notch"] > q["R-Flurbiprofen"]["S_Notch"] > q["Semagacestat"]["S_Notch"]


def test_branching_law_is_anchored_at_the_wild_type_apo_state():
    m = M.metrics(M.APO)
    assert abs(m["ratio_38_42"] - M.RHO0) < 1e-12
    assert abs(m["R_42_40_branch"] - M.R0_42_40) < 1e-12
    assert m["Gamma"] == 0.0 and m["ddG_barrier_kcal"] == 0.0
    # rho and R_42/40 move in opposite directions, by construction
    assert M.branching(2.0, 0.3)[0] > M.RHO0 > M.branching(-2.0, -0.3)[0]
    assert M.branching(2.0, 0.3)[1] < M.R0_42_40 < M.branching(-2.0, -0.3)[1]


def test_branching_law_agrees_with_the_design_brief_law_on_wild_type():
    """The two laws share one anchor, KMS-AD-309; everywhere else agreement is a check."""
    apo = M.ensemble(M.APO)
    for spec in M.COMPOUNDS:
        m = M.metrics(spec, apo)
        assert abs(m["R_branch_fold"] / m["R_fold_reduction"] - 1) < 0.10, spec.name
    lead = M.metrics(M.COMPOUND["KMS-AD-309"], apo)
    assert abs(lead["R_42_40_branch"] - M.R_LEAD_ANCHOR) < 1e-4


def test_fad_backgrounds_loosen_the_processive_path():
    apo = M.ensemble(M.APO)
    for bg in M.FAD_BACKGROUNDS:
        m = M.metrics(M.APO, apo, bg)
        assert m["zeta_bg"] < 0, bg.name                       # the allele loosens, it does not clamp
        assert m["Gamma"] < 0 and m["ddG_barrier_kcal"] < 0
        assert m["ratio_38_42"] < M.RHO0                       # less Abeta38 relative to Abeta42
        assert M.R_FAD_LOW <= m["R_42_40_branch"] <= M.R_FAD_HIGH, (bg.name, m["R_42_40_branch"])
        # the ligand term vanishes for an untreated background
        assert m["Gamma_ligand"] == 0.0 and m["CKI_TM"] < 1e-12


def test_quadrant_two_modulators_rescue_fad_and_the_others_do_not():
    apo = M.ensemble(M.APO)
    for bg in M.FAD_BACKGROUNDS:
        base = M.metrics(M.APO, apo, bg)["R_42_40_branch"]
        for name in ("E2012", "KMS-AD-309"):
            m = M.metrics(M.COMPOUND[name], apo, bg)
            assert m["Gamma"] > 0, (bg.name, name)             # the clamp changes sign
            assert m["R_42_40_branch"] < M.R0_42_40 < base
            assert m["notch_retention"] > M.NOTCH_RESCUE_MIN
        for name in ("R-Flurbiprofen", "Semagacestat"):
            m = M.metrics(M.COMPOUND[name], apo, bg)
            assert m["Gamma"] < 0, (bg.name, name)
            assert m["R_42_40_branch"] > M.R0_42_40
            assert M.rescue_occupancy(M.COMPOUND[name], bg, apo=apo) is None


def test_the_clamp_is_additive_over_background_and_ligand():
    """Gamma = Gamma_background + Gamma_ligand, and on wild type the background term vanishes."""
    apo = M.ensemble(M.APO)
    for spec in M.COMPOUNDS:
        wt = M.metrics(spec, apo)
        assert wt["Gamma_bg"] == 0.0 and wt["ddG_bg_kcal"] == 0.0
        assert abs(wt["Gamma"] - wt["Gamma_ligand"]) < 1e-12
        for bg in M.FAD_BACKGROUNDS:
            m = M.metrics(spec, apo, bg)
            assert abs(m["Gamma"] - (m["Gamma_bg"] + m["Gamma_ligand"])) < 1e-12
            assert abs(m["ddG_barrier_kcal"] - (m["ddG_bg_kcal"] + m["ddG_ligand_kcal"])) < 1e-12
            # the background term does not depend on which ligand is present
            assert abs(m["Gamma_bg"] - M.metrics(M.APO, apo, bg)["Gamma_bg"]) < 1e-12


def test_occupancy_scaling_is_monotone():
    apo = M.ensemble(M.APO)
    lead = M.COMPOUND["KMS-AD-309"]
    rs = [M.metrics(M.scaled(lead, m), apo)["R_42_40_branch"] for m in (0.25, 0.5, 1.0, 1.5)]
    assert all(a > b for a, b in zip(rs, rs[1:])), rs
    m_star = M.rescue_occupancy(lead, M.PS1_L166P, apo=apo)
    assert m_star is not None and 1.0 < m_star < 1.5
    got = M.metrics(M.scaled(lead, m_star), apo, M.PS1_L166P)["R_42_40_branch"]
    assert abs(got - M.R_RESCUE_MAX) < 1e-3


def test_allosteric_perturbation_stays_out_of_the_catalytic_block():
    """The whole design rests on the weak catalytic couplings: a gate-and-channel ligand with no
    active-site contact must leave the catalytic covariance nearly unchanged."""
    apo = M.ensemble(M.APO)
    allo = M.metrics(M.LigandSpec("allosteric-only", 0.0, 1.0, 1.0), apo)
    orth = M.metrics(M.LigandSpec("orthosteric-only", 1.0, 0.0, 0.0), apo)
    assert allo["CKI_TM"] > M.CKI_TM_MIN
    assert allo["K_catalytic"] < M.K_CAT_MAX
    assert allo["leakage_cat"] < 0.10 < orth["leakage_cat"]
    assert orth["CKI_TM"] < M.CKI_TM_MIN


def test_cki_rotation_term_is_non_negative():
    """D_rot = D_cov - D_comm >= 0 (Fiedler); the lead acts mainly through eigenvector rotation."""
    apo = M.ensemble(M.APO)
    for spec in M.COMPOUNDS:
        m = M.metrics(spec, apo)
        assert m["CKI_TM_rot"] >= -1e-9, spec.name
    lead = M.metrics(M.COMPOUND["KMS-AD-309"], apo)
    assert lead["CKI_TM_rot"] > 0.8 * lead["CKI_cov"], "the lead must rearrange, not merely rescale"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL  {name}: {exc!r}")
    print("ALL TESTS PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
