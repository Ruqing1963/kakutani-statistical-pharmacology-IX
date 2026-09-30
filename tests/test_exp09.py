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
    for spec in (M.APO,) + M.COMPOUNDS + (M.LigandSpec("extreme", 1.0, 1.35, 1.35),):
        e = np.linalg.eigvalsh(M.stiffness(spec))
        assert e[0] > 0, spec.name
        # K = (graph Laplacian, positive semi-definite) + T_TETHER I, so lambda_min >= T_TETHER
        assert e[0] >= M.T_TETHER - 1e-9, (spec.name, e[0])


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
    assert q["Flurbiprofen"]["quadrant"] == "III"
    assert q["KMS-AD-309"]["quadrant"] == "II"
    assert q["KMS-AD-309"]["K_catalytic"] < M.K_CAT_MAX
    assert q["KMS-AD-309"]["CKI_TM"] > M.CKI_TM_MIN
    assert q["KMS-AD-309"]["S_Notch"] > q["Flurbiprofen"]["S_Notch"] > q["Semagacestat"]["S_Notch"]


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
    lead = M.metrics(M.COMPOUNDS[2], apo)
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
