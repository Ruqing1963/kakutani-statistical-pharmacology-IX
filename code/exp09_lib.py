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
R0_42_40 = 0.182         # untreated (wild-type) Abeta42/Abeta40 ratio
ETA_R = 0.45             # phenomenological coupling of the ratio law (calibrated, not derived)

# branching kinetics at the E.Abeta42 intermediate of the pathogenic product line
RHO0 = 3.00              # wild-type apo Abeta38/Abeta42 partition k_42->38 / k_off,42 (biochemical input)
ETA_CLAMP = 0.22620      # clamp coupling; the one constant fitted, on the wild-type lead anchor below
R_LEAD_ANCHOR = 0.06978  # R_42/40 of KMS-AD-309 on wild type under the design-brief law, Eq. (3.8)

# acceptance thresholds of the design brief
K_CAT_MAX = 0.025        # Notch safety constraint
CKI_TM_MIN = 3.20        # covariance-rearrangement constraint
NOTCH_MIN = 0.92         # Notch signal retention
R_FAD_LOW, R_FAD_HIGH = 0.33, 0.36   # reported Abeta42/Abeta40 range of aggressive PSEN1 FAD alleles
R_RESCUE_MAX = 0.12      # rescue target: back below the healthy wild-type baseline
NOTCH_RESCUE_MIN = 0.99  # a rescue must not add Notch inhibition of its own


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


@dataclass
class Background:
    """A genetic background: wild-type presenilin-1, or a familial-AD PSEN1 allele.

    A FAD mutation is not modelled as a gain of catalytic activity. It is modelled as a loss of
    stiffness in the parts of the enzyme that hold and advance the substrate, which is what the
    clinical phenotype of these alleles implies: less processive trimming, earlier release of the
    long products, and a raised Abeta42/Abeta40 ratio.
    """
    name: str
    label: str
    path_soft: dict = field(default_factory=dict)   # multiplier on a segment's backbone stiffness
    gate_int: float = 1.0                           # multiplier on the TM6a-PAL coupling
    gate_chan: float = 1.0                          # multiplier on the gate-channel couplings
    pack: float = 1.0                               # multiplier on the channel packing contacts
    note: str = ""


WILD_TYPE = Background("wild type", "WT", note="reference presenilin-1")
PS1_L166P = Background(
    "PS1-L166P", "FAD",
    path_soft={"TM3": 0.14, "TM6": 0.36}, pack=0.65, gate_chan=0.55,
    note="helix-breaking proline in TM3; among the most aggressive PSEN1 alleles")
PS1_E280A = Background(
    "PS1-E280A", "FAD",
    path_soft={"TM6a": 0.20, "PAL": 0.40}, gate_int=0.60, gate_chan=0.70,
    note="TM6-TM7 hydrophilic loop, adjacent to TM6a; the Antioquia kindred allele")
BACKGROUNDS = (WILD_TYPE, PS1_L166P, PS1_E280A)
FAD_BACKGROUNDS = (PS1_L166P, PS1_E280A)


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


def stiffness(spec: LigandSpec = APO, bg: Background = WILD_TYPE) -> np.ndarray:
    """Stiffness matrix K(L, b) of the harmonic model, in units of k_B T per squared displacement.

    The background acts first, softening the backbone of the segments it names and weakening the
    couplings it names; the ligand then acts on top of the resulting force field.
    """
    soft = {name: bg.path_soft.get(name, 1.0) for name in SEG}
    soft["TM6a"] -= SOFT_GATE * spec.s_gate
    soft["TM3"] -= SOFT_CHAN * spec.s_chan
    soft["TM6"] -= SOFT_CHAN * spec.s_chan
    soft = {k: max(v, SOFT_FLOOR) for k, v in soft.items()}    # a backbone never softens past SOFT_FLOOR
    K = np.zeros((N_DOF, N_DOF))

    for name, sl in SEG.items():
        w = K_PATH[SEG_BLOCK[name]] * soft[name]
        for i in range(sl.start, sl.stop - 1):
            _add(K, i, i + 1, w)

    for groups, w in ((CHANNEL_PACKING, K_PACK * bg.pack), (GATE_CONTACTS, K_GATE_INT * bg.gate_int),
                      (GATE_CHANNEL_CONTACTS, K_GATE_CHAN * bg.gate_chan), (CAT_INTERNAL, K_CAT_INT),
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
    bg: Background = field(default_factory=lambda: WILD_TYPE)

    def block(self, sl: slice) -> tuple[np.ndarray, np.ndarray]:
        """Marginal of a coordinate block: a Gaussian marginal is the sub-block of Sigma."""
        return self.mu[sl], self.Sigma[sl, sl]


def ensemble(spec: LigandSpec = APO, bg: Background = WILD_TYPE) -> Ensemble:
    K = stiffness(spec, bg)
    c = cho_factor(K, lower=True, check_finite=False)
    Sigma = cho_solve(c, np.eye(N_DOF), check_finite=False)
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = cho_solve(c, binding_force(spec), check_finite=False)
    return Ensemble(spec, mu, Sigma, K, bg)


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


def branching(gamma: float, ddg_kcal: float) -> tuple[float, float]:
    """Branching kinetics at the E.Abeta42 intermediate of the pathogenic product line.

    Along that line the enzyme reaches E.Abeta42 and then either releases the pathogenic peptide,

        k_off,42  = k_off,42^(0) exp(-eta_clamp * Gamma),

    or completes one more synchronous three-residue cut to the benign Abeta38,

        k_42->38  = k_42->38^(0) exp(ddG / k_B T).

    Their ratio is the branching partition rho = k_42->38 / k_off,42, which is the Abeta38/Abeta42
    product ratio. If the flux into the intermediate and the Abeta40 output of the other product
    line are unchanged, the fraction of that flux released as Abeta42 is 1/(1 + rho), so

        R_42/40 = R_0 (1 + rho_0) / (1 + rho).

    Returns (rho, R_42/40). At Gamma = 0 and ddG = 0 this returns (rho_0, R_0) exactly.
    """
    rho = RHO0 * math.exp(ETA_CLAMP * gamma + ddg_kcal / KT_KCAL)
    return rho, R0_42_40 * (1.0 + RHO0) / (1.0 + rho)


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


def _clamp_term(state: Ensemble, ref: Ensemble) -> dict:
    """One signed contribution to the clamp on the substrate, for `state` measured against `ref`.

    The correlated index is a distance and cannot tell a ligand that stiffens the processive path
    from a mutation that loosens it. The sign is supplied by the step weights, which are directed:
    zeta = +1 when the state raises the mean step weight above the reference, -1 when it lowers it.
    The clamp contribution is then

        Gamma = zeta (CKI_cov - sigma d_cone),        ddG = zeta sigma k_B T (sum_j w_j - d_cone),

    both of which vanish when the state equals the reference.
    """
    from kakutani_pharma import compute_cki

    tm = compute_cki(*ref.block(SL_MACHINERY), *state.block(SL_MACHINERY))
    w0 = float(step_weights(ref, ref).mean())          # W_STEP by construction
    w = step_weights(state, ref)
    d_cone = cone_geodesic(w)
    w_sum = float(w.sum())
    sigma = synchrony(tm.total)
    zeta = 1.0 if float(w.mean()) >= w0 else -1.0
    return {"tm": tm, "w": w, "w_sum": w_sum, "w_mean": float(w.mean()), "d_cone": d_cone,
            "sigma": sigma, "d_cone_eff": sigma * d_cone, "zeta": zeta,
            "Gamma": zeta * (tm.cov_term - sigma * d_cone),
            "ddG": zeta * sigma * KT_KCAL * (w_sum - d_cone)}


def metrics(spec: LigandSpec, apo: Ensemble | None = None, bg: Background = WILD_TYPE) -> dict:
    """All read-outs of one state, a ligand acting on a genetic background.

    The clamp on the substrate is additive over the two perturbations, each measured against its own
    reference and carrying its own sign:

        Gamma(b, L) = Gamma_background(b vs wild type) + Gamma_ligand(L on b vs b untreated).

    Measuring both against the wild type instead would let a mutation's own large rearrangement be
    counted as a benefit once a ligand flipped the overall sign, and a weak modulator would appear
    to rescue a severe allele. The ligand terms, and the quadrant coordinates, are therefore taken
    against the background's own untreated ensemble: what a drug does is what it does to the enzyme
    the patient has. For the wild-type background the background term vanishes identically and every
    quantity reduces to the single-perturbation case.
    """
    from kakutani_pharma import compute_cki

    if apo is None:
        apo = ensemble(APO, WILD_TYPE)
    bg_apo = apo if bg is WILD_TYPE else ensemble(APO, bg)
    holo = ensemble(spec, bg)

    back = _clamp_term(bg_apo, apo)                      # the allele, against wild type
    lig = _clamp_term(holo, bg_apo)                      # the ligand, against the untreated allele
    cat = compute_cki(*bg_apo.block(SL_CAT), *holo.block(SL_CAT))
    tm = lig["tm"]

    gamma = back["Gamma"] + lig["Gamma"]
    ddg_barrier = back["ddG"] + lig["ddG"]
    rho, r_branch = branching(gamma, ddg_barrier)
    r_ratio = R0_42_40 * math.exp(-ETA_R * gamma)        # design-brief law, for comparison
    w = lig["w"]

    return {
        "ligand": spec.name, "background": bg.name, "note": spec.note,
        "s_cat": spec.s_cat, "s_gate": spec.s_gate, "s_chan": spec.s_chan,
        "K_catalytic": cat.total, "K_cat_mean": cat.mean_term, "K_cat_cov": cat.cov_term,
        "CKI_TM": tm.total, "CKI_TM_mean": tm.mean_term, "CKI_cov": tm.cov_term,
        "CKI_TM_comm": tm.comm_term, "CKI_TM_rot": tm.rot_term,
        "S_Notch": tm.total / (cat.total + EPS0),
        "notch_retention": notch_retention(cat.total),
        "leakage_cat": leakage(holo, bg_apo),
        "w_TM3": float(w[0]), "w_TM6a": float(w[1]), "w_PAL": float(w[2]), "w_sum": lig["w_sum"],
        "w_mean": lig["w_mean"], "zeta": lig["zeta"], "zeta_bg": back["zeta"],
        "d_cone": lig["d_cone"], "d_cone_over_wsum": lig["d_cone"] / lig["w_sum"],
        "sigma_sync": lig["sigma"], "d_cone_eff": lig["d_cone_eff"],
        "CKI_TM_bg": back["tm"].total, "sigma_bg": back["sigma"], "w_mean_bg": back["w_mean"],
        "Gamma_bg": back["Gamma"], "Gamma_ligand": lig["Gamma"], "Gamma": gamma,
        "ddG_bg_kcal": back["ddG"], "ddG_ligand_kcal": lig["ddG"], "ddG_barrier_kcal": ddg_barrier,
        "rate_factor_42_38": math.exp(ddg_barrier / KT_KCAL),
        "ratio_38_42": rho, "ratio_38_42_fold": rho / RHO0,
        "R_42_40": r_ratio, "R_fold_reduction": R0_42_40 / r_ratio,
        "R_42_40_branch": r_branch, "R_branch_fold": R0_42_40 / r_branch,
        "pass_notch": int(cat.total < K_CAT_MAX), "pass_cki": int(tm.total > CKI_TM_MIN),
        "quadrant": quadrant(cat.total, tm.total),
    }


# ----------------------------------------------------------------------------
# Reference compounds
# ----------------------------------------------------------------------------
COMPOUNDS = (
    LigandSpec("Semagacestat", s_cat=1.00, s_gate=0.05, s_chan=0.05,
               note="orthosteric GSI, transition-state analogue; Phase III failure (IDENTITY)"),
    LigandSpec("Avagacestat", s_cat=0.28, s_gate=0.32, s_chan=0.28,
               note="aryl sulfonamide marketed as a Notch-sparing GSI; Phase II, Notch-type adverse events"),
    LigandSpec("R-Flurbiprofen", s_cat=0.004, s_gate=0.18, s_chan=0.12,
               note="tarenflurbil, first-generation NSAID-derived GSM; Phase III failure on efficacy"),
    LigandSpec("E2012", s_cat=0.008, s_gate=0.95, s_chan=0.92,
               note="second-generation bridged-imidazole GSM class; lowers CSF Abeta42, raises Abeta38"),
    LigandSpec("KMS-AD-309", s_cat=0.02, s_gate=1.00, s_chan=1.00,
               note="Kakutani-spectrum optimised lead; gate and channel coupling, no active-site contact"),
)
COMPOUND = {c.name: c for c in COMPOUNDS}


def scaled(spec: LigandSpec, m: float) -> LigandSpec:
    """The same ligand at occupancy m: every strength scaled, which is what raising the exposure
    of a reversible binder does in this model."""
    return LigandSpec(f"{spec.name} x{m:.3g}", m * spec.s_cat, m * spec.s_gate, m * spec.s_chan,
                      note=f"{spec.note} (occupancy {m:.3g})")


def rescue_occupancy(spec: LigandSpec, bg: Background, target: float = R_RESCUE_MAX,
                     apo: Ensemble | None = None, hi: float = 4.0) -> float | None:
    """Smallest occupancy multiplier m at which the ligand brings R_42/40 of the background down to
    `target`. Returns None if the target is out of reach at m <= hi."""
    if apo is None:
        apo = ensemble(APO, WILD_TYPE)

    def f(m: float) -> float:
        return metrics(scaled(spec, m), apo, bg)["R_42_40_branch"] - target

    if f(hi) > 0:
        return None
    lo = 1e-3
    if f(lo) < 0:
        return lo
    return float(brentq(f, lo, hi, xtol=1e-4))
