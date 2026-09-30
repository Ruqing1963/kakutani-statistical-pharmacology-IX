#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp09_lib.py
============
Statistical Pharmacology via Kakutani Dichotomy, Paper IX -- model library.

A Gaussian elastic model of the Presenilin-1 / gamma-secretase catalytic core, the
perturbation of that model by a ligand, and the discrete infinity-Laplacian cone
geodesic on the three-residue processive-step lattice {0, 1}^3.

Topology (D = 300 harmonic degrees of freedom, three blocks)
    catalytic block      30 dof   D257 loop (TM6), D385 loop (TM7), S3 register
    gate block           70 dof   TM6a (40) + PAL motif (30)
    channel block       200 dof   TM1 ... TM9, the processive-trimming path

    The contact graph places the catalytic aspartates on TM6 and TM7, as in PS1, and
    couples them to the rest only weakly (K_CAT_CHAN, K_CAT_GATE). That weak coupling is
    what makes a Notch-sparing quadrant-II ligand possible at all: a perturbation
    supported on the gate and the channel reaches the catalytic block only at second
    order in those couplings. The leakage is measured, not assumed (see `leakage`).

Energetics
    K       stiffness (Hessian) = weighted graph Laplacian of the contact graph + tether
    Sigma   covariance = K^-1 in units of k_B T
    mu      mean displacement = Sigma f, with f the ligand binding force

Cone geodesic
    Each advance of the substrate by three residues requires the synchronous displacement
    (x_TM3, x_TM6a, x_PAL): (0,0,0) -> (1,1,1). The discrete infinity-Laplacian distance on
    {0,1}^3 with step weights w is computed by the exact top-down recursion of Paper VI,
    `cone_geodesic`. For equal weights it equals w (1 + 1/sqrt(2) + 1/sqrt(3)) = 2.284457 w,
    strictly below the sequential cost 3 w.

The dynamics are a harmonic (Gaussian) surrogate on a schematic contact graph, not
molecular dynamics, and the three named compounds enter only through their ligand class
(active-site occupancy, gate coupling, channel coupling). See the README for the limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

SEED_MODEL = 20260929

# ----------------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------------
CAT_SEGMENTS = (("D257_loop", 8), ("D385_loop", 8), ("S3_register", 14))          # 30
GATE_SEGMENTS = (("TM6a", 40), ("PAL", 30))                                       # 70
CHAN_SEGMENTS = (("TM1", 24), ("TM2", 24), ("TM3", 26), ("TM4", 22), ("TM5", 22),
                 ("TM6", 24), ("TM7", 22), ("TM8", 20), ("TM9", 16))              # 200

N_CAT = sum(n for _, n in CAT_SEGMENTS)
N_GATE = sum(n for _, n in GATE_SEGMENTS)
N_CHAN = sum(n for _, n in CHAN_SEGMENTS)
N_DOF = N_CAT + N_GATE + N_CHAN

SL_CAT = slice(0, N_CAT)
SL_GATE = slice(N_CAT, N_CAT + N_GATE)
SL_CHAN = slice(N_CAT + N_GATE, N_DOF)
SL_MACHINERY = slice(N_CAT, N_DOF)          # gate + channel: the trimming machinery, 270 dof


def _segment_index() -> dict[str, slice]:
    out, off = {}, 0
    for name, n in CAT_SEGMENTS + GATE_SEGMENTS + CHAN_SEGMENTS:
        out[name] = slice(off, off + n)
        off += n
    assert off == N_DOF
    return out


SEG = _segment_index()
SEG_BLOCK = ({name: "cat" for name, _ in CAT_SEGMENTS} | {name: "gate" for name, _ in GATE_SEGMENTS}
             | {name: "chan" for name, _ in CHAN_SEGMENTS})

# ----------------------------------------------------------------------------
# Apo force field
# ----------------------------------------------------------------------------
K_PATH = {"cat": 6.0, "gate": 4.0, "chan": 5.0}      # backbone path stiffness inside a segment
T_TETHER = 0.90                                      # bilayer / scaffold restraint, makes K positive definite
K_PACK = 1.20                                        # helix-helix packing inside the channel
K_GATE_INT = 1.50                                    # TM6a - PAL
K_GATE_CHAN = 1.00                                   # gate - channel
K_CAT_INT = 0.80                                     # inside the catalytic block
K_CAT_CHAN = 0.35                                    # catalytic - channel (deliberately weak)
K_CAT_GATE = 0.25                                    # catalytic - gate (deliberately weak)

CHANNEL_PACKING = (("TM1", "TM2", 5), ("TM2", "TM3", 5), ("TM3", "TM4", 5), ("TM4", "TM5", 5),
                   ("TM5", "TM6", 5), ("TM6", "TM7", 5), ("TM7", "TM8", 4), ("TM8", "TM9", 4),
                   ("TM1", "TM9", 4), ("TM2", "TM6", 3), ("TM3", "TM5", 3))
GATE_CONTACTS = (("TM6a", "PAL", 6),)
GATE_CHANNEL_CONTACTS = (("TM6a", "TM6", 5), ("TM6a", "TM7", 4), ("PAL", "TM8", 3), ("PAL", "TM9", 3))
CAT_INTERNAL = (("D257_loop", "D385_loop", 3), ("D385_loop", "S3_register", 3), ("D257_loop", "S3_register", 2))
CAT_CHANNEL_CONTACTS = (("D257_loop", "TM6", 2), ("D385_loop", "TM7", 2))      # Asp257 on TM6, Asp385 on TM7
CAT_GATE_CONTACTS = (("S3_register", "PAL", 2),)

# ----------------------------------------------------------------------------
# Ligand perturbation
# ----------------------------------------------------------------------------
G_CAT_DYAD = 5.00        # transition-state analogue bridging D257 and D385
G_CAT_S3 = 3.00          # occlusion of the S3 register
G_GATE = 5.75            # TM6a - PAL gate coupling created by an allosteric ligand
G_CHAN = 7.20            # new cross-helix couplings in the channel
G_CHAN_GATE = 4.50       # TM3 - TM6a coupling: the origin of the synchronous three-residue step
SOFT_GATE = 0.50         # fractional softening of the TM6a backbone at s_gate = 1
SOFT_CHAN = 0.50         # fractional softening of the TM3 and TM6 backbones at s_chan = 1
SOFT_FLOOR = 0.15        # lower bound of a softening factor, so K stays a Laplacian with positive weights
F_CAT, F_GATE, F_CHAN = 0.60, 0.25, 0.20     # binding-force amplitudes

LIGAND_CAT_EDGES = (("D257_loop", "D385_loop", 8, G_CAT_DYAD), ("D385_loop", "S3_register", 6, G_CAT_S3))
LIGAND_GATE_EDGES = (("TM6a", "PAL", 12, G_GATE),)
LIGAND_CHAN_EDGES = (("TM3", "TM6", 6, G_CHAN), ("TM3", "TM7", 5, G_CHAN), ("TM5", "TM9", 4, G_CHAN),
                     ("TM2", "TM8", 4, G_CHAN), ("TM3", "TM6a", 4, G_CHAN_GATE))

# collective three-residue step coordinates of the processive path
STEP_SEGMENTS = ("TM3", "TM6a", "PAL")

# ----------------------------------------------------------------------------
# Thermodynamic and phenomenological constants
# ----------------------------------------------------------------------------
KT_KCAL = 0.6162         # k_B T at 310 K, kcal/mol
W_STEP = 1.05            # apo step weight of one helix, in k_B T
EPS0 = 1.0e-3            # regulariser of the Notch selectivity index
N_S3 = 3                 # subsites of the S3 register: Notch retention = exp(-N_S3 K_catalytic)
CKI_TM_REF = 3.20        # reference of the synchrony law; equals the quadrant-II threshold
R0_42_40 = 0.182         # untreated Abeta42/Abeta40 ratio
ETA_R = 0.45             # phenomenological coupling of the ratio law (calibrated, not derived)

# acceptance thresholds of the design brief
K_CAT_MAX = 0.025        # Notch safety constraint
CKI_TM_MIN = 3.20        # covariance-rearrangement constraint
NOTCH_MIN = 0.92         # Notch signal retention


# ----------------------------------------------------------------------------
# Graph and stiffness
# ----------------------------------------------------------------------------
def _pairs(a: str, b: str, m: int) -> list[tuple[int, int]]:
    """m evenly spaced contact pairs between segments a and b."""
    ia, ib = SEG[a], SEG[b]
    na, nb = ia.stop - ia.start, ib.stop - ib.start
    ja = np.linspace(0, na - 1, m).round().astype(int)
    jb = np.linspace(0, nb - 1, m).round().astype(int)
    return [(ia.start + int(x), ib.start + int(y)) for x, y in zip(ja, jb)]


def _add(K: np.ndarray, i: int, j: int, w: float) -> None:
    """Laplacian edge: a harmonic spring between the two coordinates."""
    K[i, i] += w
    K[j, j] += w
    K[i, j] -= w
    K[j, i] -= w


@dataclass
class LigandSpec:
    """A ligand as three occupancy strengths, all zero for apo."""
    name: str
    s_cat: float = 0.0        # active-site (catalytic dyad / S3 register) occupancy
    s_gate: float = 0.0       # TM6a / PAL gate coupling
    s_chan: float = 0.0       # TM1-TM9 channel coupling rearrangement
    note: str = ""


APO = LigandSpec("apo", 0.0, 0.0, 0.0, "unliganded reference")


def _force_patterns() -> dict[str, np.ndarray]:
    """Fixed unit-norm binding-force patterns, drawn once from a seeded generator."""
    rng = np.random.default_rng(SEED_MODEL)
    out = {}
    for key, sl in (("cat", SL_CAT), ("gate", SL_GATE), ("chan", SL_CHAN)):
        v = np.zeros(N_DOF)
        x = rng.standard_normal(sl.stop - sl.start)
        v[sl] = x / np.linalg.norm(x)
        out[key] = v
    return out


FORCE = _force_patterns()


def stiffness(spec: LigandSpec = APO) -> np.ndarray:
    """Stiffness matrix K(L) of the harmonic model, in units of k_B T per squared displacement."""
    soft = {name: 1.0 for name in SEG}
    soft["TM6a"] -= SOFT_GATE * spec.s_gate
    soft["TM3"] -= SOFT_CHAN * spec.s_chan
    soft["TM6"] -= SOFT_CHAN * spec.s_chan
    soft = {k: max(v, SOFT_FLOOR) for k, v in soft.items()}    # a backbone never softens past SOFT_FLOOR
    K = np.zeros((N_DOF, N_DOF))

    for name, sl in SEG.items():
        w = K_PATH[SEG_BLOCK[name]] * soft[name]
        for i in range(sl.start, sl.stop - 1):
            _add(K, i, i + 1, w)

    for groups, w in ((CHANNEL_PACKING, K_PACK), (GATE_CONTACTS, K_GATE_INT),
                      (GATE_CHANNEL_CONTACTS, K_GATE_CHAN), (CAT_INTERNAL, K_CAT_INT),
                      (CAT_CHANNEL_CONTACTS, K_CAT_CHAN), (CAT_GATE_CONTACTS, K_CAT_GATE)):
        for a, b, m in groups:
            for i, j in _pairs(a, b, m):
                _add(K, i, j, w)

    for edges, s in ((LIGAND_CAT_EDGES, spec.s_cat), (LIGAND_GATE_EDGES, spec.s_gate),
                     (LIGAND_CHAN_EDGES, spec.s_chan)):
        if s == 0.0:
            continue
        for a, b, m, g in edges:
            for i, j in _pairs(a, b, m):
                _add(K, i, j, g * s)

    K[np.diag_indices(N_DOF)] += T_TETHER
    return K


def binding_force(spec: LigandSpec) -> np.ndarray:
    return (F_CAT * spec.s_cat * FORCE["cat"] + F_GATE * spec.s_gate * FORCE["gate"]
            + F_CHAN * spec.s_chan * FORCE["chan"])


@dataclass
class Ensemble:
    """A Gaussian conformational ensemble N(mu, Sigma) of the 300 coordinates."""
    spec: LigandSpec
    mu: np.ndarray
    Sigma: np.ndarray
    K: np.ndarray = field(repr=False, default=None)

    def block(self, sl: slice) -> tuple[np.ndarray, np.ndarray]:
        """Marginal of a coordinate block: a Gaussian marginal is the sub-block of Sigma."""
        return self.mu[sl], self.Sigma[sl, sl]


def ensemble(spec: LigandSpec = APO) -> Ensemble:
    K = stiffness(spec)
    c = cho_factor(K, lower=True, check_finite=False)
    Sigma = cho_solve(c, np.eye(N_DOF), check_finite=False)
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = cho_solve(c, binding_force(spec), check_finite=False)
    return Ensemble(spec, mu, Sigma, K)


def leakage(holo: Ensemble, apo: Ensemble) -> float:
    """Relative change of the catalytic-block covariance, ||dSigma_C|| / ||Sigma_C|| (Frobenius)."""
    a, b = apo.Sigma[SL_CAT, SL_CAT], holo.Sigma[SL_CAT, SL_CAT]
    return float(np.linalg.norm(b - a) / np.linalg.norm(a))


# ----------------------------------------------------------------------------
# Collective step coordinates
# ----------------------------------------------------------------------------
def step_directions() -> np.ndarray:
    """Unit vectors e_j of the uniform displacement of TM3, TM6a and PAL (3, D)."""
    E = np.zeros((len(STEP_SEGMENTS), N_DOF))
    for r, name in enumerate(STEP_SEGMENTS):
        sl = SEG[name]
        E[r, sl] = 1.0 / math.sqrt(sl.stop - sl.start)
    return E


STEP_DIRS = step_directions()


def step_weights(holo: Ensemble, apo: Ensemble, w0: float = W_STEP) -> np.ndarray:
    """Step weights w_j = w0 sqrt(Var_apo(e_j) / Var_holo(e_j)).

    Var(e_j) = e_j^T Sigma e_j is the variance of the collective displacement of helix j with
    every other coordinate relaxed, so 1 / Var is its effective free-energy curvature. The
    weights are w0 for the apo model by construction.
    """
    va = np.einsum("ij,jk,ik->i", STEP_DIRS, apo.Sigma, STEP_DIRS)
    vb = np.einsum("ij,jk,ik->i", STEP_DIRS, holo.Sigma, STEP_DIRS)
    return w0 * np.sqrt(va / vb)


# ----------------------------------------------------------------------------
# Cone geodesic on {0, 1}^k  (Paper VI, exact top-down recursion)
# ----------------------------------------------------------------------------
def cone_geodesic(w) -> float:
    """Exact discrete infinity-Laplacian distance d_L(0, 1) on the lattice {0,1}^k.

    At every vertex the value solves sum_j ((h(mask | e_j) - h(mask)) / w_j)^2 = 1 over the
    coordinates not yet flipped, the Aronsson / infinity-harmonic condition of Paper VI,
    Section 9. For equal weights the value is w sum_{m=1..k} 1/sqrt(m).
    """
    w = [float(x) for x in w]
    k = len(w)
    full = (1 << k) - 1
    h = np.zeros(1 << k)
    for mask in sorted(range(1 << k), key=lambda x: -bin(x).count("1")):
        if mask == full:
            continue
        free = [j for j in range(k) if not mask >> j & 1]
        hs = np.array([h[mask | 1 << j] for j in free])
        ws = np.array([w[j] for j in free])
        f = lambda x: np.sum(((x - hs) / ws) ** 2) - 1
        lo = hs.max()
        h[mask] = brentq(f, lo, lo + ws.sum() + 1, xtol=1e-15, rtol=1e-15)
    return float(h[0])


def cone_slack(w) -> float:
    """Well-posedness margin of the recursion.

    At a vertex with free set F the root is sought in [max_j h_j, infinity), where the left-hand
    side is strictly increasing. A root exists there only if the value at the left endpoint does
    not already exceed 1, that is if

        g(x) = 1 - sum_{j in F} ((max_i h_i - h_j) / w_j)^2 >= 0.

    This returns min_x g(x) over all vertices. A non-negative value certifies that the plain
    equation, without a positive part, determines the distance.
    """
    w = [float(x) for x in w]
    k = len(w)
    full = (1 << k) - 1
    h = np.zeros(1 << k)
    slack = np.inf
    for mask in sorted(range(1 << k), key=lambda x: -bin(x).count("1")):
        if mask == full:
            continue
        free = [j for j in range(k) if not mask >> j & 1]
        hs = np.array([h[mask | 1 << j] for j in free])
        ws = np.array([w[j] for j in free])
        slack = min(slack, 1.0 - float(np.sum(((hs.max() - hs) / ws) ** 2)))
        f = lambda x: np.sum(((x - hs) / ws) ** 2) - 1
        lo = hs.max()
        h[mask] = brentq(f, lo, lo + ws.sum() + 1, xtol=1e-15, rtol=1e-15)
    return float(slack)


def cone_equal_weight(w0: float, k: int = 3) -> float:
    """Closed form for equal weights: w0 (1 + 1/sqrt(2) + ... + 1/sqrt(k))."""
    return w0 * sum(1.0 / math.sqrt(m) for m in range(1, k + 1))


# ----------------------------------------------------------------------------
# Pharmacological read-outs
# ----------------------------------------------------------------------------
def notch_retention(k_catalytic: float) -> float:
    """Retention of Notch S3 signalling.

    S3 proteolysis needs the correct register at N_S3 consecutive subsites, and each subsite
    keeps its register with probability equal to the Bhattacharyya affinity exp(-K_catalytic)
    of the catalytic ensembles, so retention = exp(-N_S3 K_catalytic). The design threshold
    K_catalytic < 0.025 gives retention > exp(-0.075) = 0.9277.
    """
    return float(math.exp(-N_S3 * k_catalytic))


def synchrony(cki_tm: float) -> float:
    """Fraction sigma of three-residue steps that proceed synchronously rather than one helix
    at a time: sigma = 1 - exp(-CKI_TM / CKI_TM_REF). Only the synchronous fraction travels the
    cone geodesic, so sigma weights both the cone penalty and the barrier discount."""
    return float(-np.expm1(-cki_tm / CKI_TM_REF))


def quadrant(k_catalytic: float, cki_tm: float) -> str:
    """Four-quadrant class in the (K_catalytic, CKI_TM) plane, in the convention of Paper IV."""
    hi_cat = k_catalytic > K_CAT_MAX
    hi_tm = cki_tm > CKI_TM_MIN
    return {(True, True): "I", (False, True): "II", (False, False): "III", (True, False): "IV"}[(hi_cat, hi_tm)]


QUADRANT_NAME = {
    "I": "non-selective allosteric inhibitor (channel modulation with Notch liability)",
    "II": "Notch-sparing gamma-secretase modulator (target class)",
    "III": "silent / first-generation weak GSM",
    "IV": "orthosteric gamma-secretase inhibitor (GSI)",
}


def metrics(spec: LigandSpec, apo: Ensemble | None = None) -> dict:
    """All read-outs of one ligand. Imports the index estimators of the package lazily so that
    the library can be imported without the package on the path."""
    from kakutani_pharma import compute_cki

    if apo is None:
        apo = ensemble(APO)
    holo = ensemble(spec)
    cat = compute_cki(*apo.block(SL_CAT), *holo.block(SL_CAT))
    tm = compute_cki(*apo.block(SL_MACHINERY), *holo.block(SL_MACHINERY))

    w = step_weights(holo, apo)
    d_cone = cone_geodesic(w)
    w_sum = float(w.sum())
    sigma = synchrony(tm.total)
    d_cone_eff = sigma * d_cone                      # cone penalty actually paid, in index units
    ddg_barrier = sigma * KT_KCAL * (w_sum - d_cone)  # Abeta42 -> Abeta38 barrier discount, kcal/mol
    cki_cov = tm.cov_term
    r_ratio = R0_42_40 * math.exp(-ETA_R * (cki_cov - d_cone_eff))

    return {
        "ligand": spec.name, "note": spec.note,
        "s_cat": spec.s_cat, "s_gate": spec.s_gate, "s_chan": spec.s_chan,
        "K_catalytic": cat.total, "K_cat_mean": cat.mean_term, "K_cat_cov": cat.cov_term,
        "CKI_TM": tm.total, "CKI_TM_mean": tm.mean_term, "CKI_cov": cki_cov,
        "CKI_TM_comm": tm.comm_term, "CKI_TM_rot": tm.rot_term,
        "S_Notch": tm.total / (cat.total + EPS0),
        "notch_retention": notch_retention(cat.total),
        "leakage_cat": leakage(holo, apo),
        "w_TM3": float(w[0]), "w_TM6a": float(w[1]), "w_PAL": float(w[2]), "w_sum": w_sum,
        "d_cone": d_cone, "d_cone_over_wsum": d_cone / w_sum,
        "sigma_sync": sigma, "d_cone_eff": d_cone_eff,
        "ddG_barrier_kcal": ddg_barrier,
        "rate_factor_42_38": math.exp(ddg_barrier / KT_KCAL),
        "R_42_40": r_ratio, "R_fold_reduction": R0_42_40 / r_ratio,
        "pass_notch": int(cat.total < K_CAT_MAX), "pass_cki": int(tm.total > CKI_TM_MIN),
        "quadrant": quadrant(cat.total, tm.total),
    }


# ----------------------------------------------------------------------------
# Reference compounds
# ----------------------------------------------------------------------------
COMPOUNDS = (
    LigandSpec("Semagacestat", s_cat=1.00, s_gate=0.05, s_chan=0.05,
               note="orthosteric GSI, transition-state analogue; Phase III failure (IDENTITY)"),
    LigandSpec("Flurbiprofen", s_cat=0.004, s_gate=0.18, s_chan=0.12,
               note="first-generation NSAID-derived GSM; weak channel coupling"),
    LigandSpec("KMS-AD-309", s_cat=0.02, s_gate=1.00, s_chan=1.00,
               note="Kakutani-spectrum optimised lead; gate and channel coupling, no active-site contact"),
)
