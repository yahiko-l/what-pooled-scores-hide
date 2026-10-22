"""matched_construction.py — PLAN_AMENDMENT_4 §C shared production constructor.

SINGLE OWNER of the displacement-matched dose-grid construction (prereg-v4,
design review PASS R8). matched_pairs.py, matched_feasibility_v4.py
(F-1) and every v4 runner import THIS module; nothing else may construct arms.

Contract (PLAN_AMENDMENT_4 §C-1/2/4/5):
  * Legacy RNG bit-exactness — the v3 construction stream (W-unit shuffles,
    best-of-20 greedy X-matching restarts, per-edge Xm option coins, Xfull) is
    reproduced with an IDENTICAL numpy call sequence on the same
    default_rng(draw_seed) stream. The legacy primitives are IMPORTED from the
    frozen v3 module matched_feasibility (zero copy divergence); only the
    build_matched_pair BODY is re-expressed here so that master records can be
    emitted without consuming any extra legacy randomness. Certificate (xvii)
    plus the golden fixtures in test_matched_construction.py pin byte equality
    of pi_W / pi_X / pi_Xfull against the pristine v3 path.
  * Master records — one record per X-matching edge, in the WINNING greedy
    pass's acceptance order (canonical order): record = {w1=(r1,r2), w2=(r3,r4),
    x1=(x1a,x1b), x2=(x2a,x2b)} with the literal chosen Xm transpositions
    (no index indirection; §C-1 [R3-8-fix]). Serialization is ASCII
    "r1,r2,r3,r4,x1a,x1b,x2a,x2b" joined by ";" -> master_bytes/master_hash.
  * Nested dose grid — order substream: order_seed = first 8 bytes (little,
    unsigned) of sha256("dose_order|{backbone}|{graph_id}|{draw_seed}");
    sigma_d = Generator(PCG64(order_seed)).permutation(U). Dose q selects the
    WHOLE records sigma_d[:ceil(q*U)]; every lower dose is a restriction.
  * Wcap anchor — ALL pre-trim W-units of the draw (class-internal
    transpositions at capacity), zero extra legacy RNG (§C-4 [R1-19-fix]).
  * Certificates (xi)-(xx) raise CertificateError (terminal, never retried).

Runs under plain CPU; zero checkpoint/model contact in this module.
"""
import hashlib
import math

import numpy as np

import matched_feasibility as MF

if not __debug__:  # §C certificates rely on assert — refuse optimizer mode
    raise SystemExit("matched_construction: refusing to run under python -O "
                     "(certificate asserts would be disabled)")


class CertificateError(Exception):
    """A §C-5 construction certificate failed. NEVER retried (fail-closed)."""


class DoseInfeasible(CertificateError):
    """The matched pool is too small for the frozen dose grid (U == 0, or the
    quartile prefixes collide so strict f-monotonicity is undefined). An
    EXPECTED capacity state (drives B-4 ineligibility / report-only
    exclusion), NOT invariant corruption — distinguished so F-1 can route it
    without aborting (C2-9-fix)."""


# ------------------------------------------------------------ frozen constants ----

# Canonical home as of prereg-v4 (matched_pairs re-imports these; values are
# byte-identical to the prereg-v3 constants they supersede).
BACKBONE_PRESET = {
    "ultra": dict(joint=True, strong=True, compute_xfull=True),
    "trix": dict(joint=True, strong=True, compute_xfull=True),
    "flock": dict(joint=False, strong=False, compute_xfull=False),
}

DRAW_SEEDS = {
    "ultra": list(range(20)),
    "trix": list(range(20)),
    "flock": list(range(10)),
}

# PLAN_AMENDMENT_4 §A/§C-2: ULTRA/TRIX four nested relative-capacity dose
# levels; FLOCK two (budget-driven, §B-6/§G — disclosed).
DOSE_GRID = {
    "ultra": (0.25, 0.50, 0.75, 1.00),
    "trix": (0.25, 0.50, 0.75, 1.00),
    "flock": (0.50, 1.00),
}


# ------------------------------------------------------------------ primitives ----

def fingerprint(arr):
    """sha256 of the int64 byte image of a permutation / index array (§C-5(xviii);
    identical convention to prereg-v3 matched_pairs.fingerprint)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def order_seed_int(backbone, graph_id, draw_seed):
    """§C-2 literal derivation of the dose-order substream seed."""
    key = f"dose_order|{backbone}|{graph_id}|{int(draw_seed)}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8],
                          "little", signed=False)


def dose_sigma(backbone, graph_id, draw_seed, n_records):
    """sigma_d: one seeded permutation of the canonical master-record indices,
    derived AFTER legacy construction from an independent PCG64 stream."""
    oseed = order_seed_int(backbone, graph_id, draw_seed)
    gen = np.random.Generator(np.random.PCG64(oseed))
    return oseed, gen.permutation(int(n_records))


def record_ascii(rec):
    """§C-1 frozen record serialization (literal chosen Xm transpositions)."""
    (r1, r2), (r3, r4) = rec["w1"], rec["w2"]
    (x1a, x1b), (x2a, x2b) = rec["x1"], rec["x2"]
    return f"{r1},{r2},{r3},{r4},{x1a},{x1b},{x2a},{x2b}"


def master_serialization(records):
    """master_bytes = ';'-joined record ASCII in canonical (acceptance) order."""
    mb = ";".join(record_ascii(r) for r in records).encode("ascii")
    return mb, hashlib.sha256(mb).hexdigest()


# ------------------------------------------------- legacy body + master records ----

def _build_legacy_with_records(prof, R_base, seed, joint, strong, compute_xfull):
    """Bit-exact re-expression of MF.build_matched_pair exposing master records.

    The numpy RNG call sequence is IDENTICAL to the frozen v3 constructor:
      #1 MF._w_units          (per-class permutation of members)
      #2 MF._greedy_maximal_matching over W-units (20 restarts)
      #3 one rng.integers(len(opts)) per matching edge, in the winning pass's
         acceptance order (the coin is consumed even when len(opts) == 1,
         exactly as in v3)
      #4 MF._greedy_maximal_matching for Xfull (ULTRA/TRIX only)
    No other randomness is consumed. AssertionError from the imported v3
    primitives propagates to the public API where it becomes CertificateError.
    """
    rng = np.random.default_rng(seed)
    classes = MF._classes_of(prof, R_base, joint=joint)
    units = MF._w_units(classes, rng)                                # legacy #1
    nu = len(units)
    cls_of = {}
    for key, members in classes.items():
        for r in members:
            cls_of[r] = key
    adj = [[] for _ in range(nu)]
    for i in range(nu):
        for j in range(i + 1, nu):
            if cls_of[units[i][0]] == cls_of[units[j][0]]:
                continue
            if MF._unit_pair_options(prof, R_base, units[i], units[j], strong):
                adj[i].append(j)
                adj[j].append(i)
    matching = MF._greedy_maximal_matching(nu, adj, rng)             # legacy #2
    kept_units = sorted({i for ij in matching for i in ij})
    records = []
    x_pairs = []
    for i, j in matching:  # canonical order = winning-pass acceptance order
        opts = MF._unit_pair_options(prof, R_base, units[i], units[j], strong)
        chosen = opts[int(rng.integers(len(opts)))]                  # legacy #3
        x_pairs.extend(chosen)
        records.append({
            "w1": (int(units[i][0]), int(units[i][1])),
            "w2": (int(units[j][0]), int(units[j][1])),
            "x1": (int(chosen[0][0]), int(chosen[0][1])),
            "x2": (int(chosen[1][0]), int(chosen[1][1])),
        })
    w_pairs = [units[i] for i in kept_units]
    pi_w = MF._lift(w_pairs, R_base)
    pi_x = MF._lift(x_pairs, R_base)
    M_w = MF._certify_lifted(pi_w, w_pairs, R_base, "Wm")
    M_x = MF._certify_lifted(pi_x, x_pairs, R_base, "Xm")
    assert M_w == M_x, "moved-set mismatch between arms"
    for a, b in w_pairs:
        assert cls_of[a] == cls_of[b], "W pair crosses class"
        assert MF._profile_l1(prof, a, b, R_base) == 0, \
            "W pair joint profile dist != 0"
    for a, b in x_pairs:
        assert MF._cross_ok(prof, R_base, a, b, strong), "X pair not cross"
    x_dist = [MF._profile_l1(prof, a, b, R_base) for a, b in x_pairs]
    xfull_pairs = []
    pi_xfull = None
    if compute_xfull:
        all_adj = [[] for _ in range(R_base)]
        for x in range(R_base):
            for y in range(x + 1, R_base):
                if MF._cross_ok(prof, R_base, x, y, strong):
                    all_adj[x].append(y)
                    all_adj[y].append(x)
        xfull_pairs = MF._greedy_maximal_matching(R_base, all_adj, rng)  # legacy #4
        pi_xfull = MF._lift(xfull_pairs, R_base)
        MF._certify_lifted(pi_xfull, xfull_pairs, R_base, "Xfull")
        for a, b in xfull_pairs:
            assert MF._cross_ok(prof, R_base, a, b, strong), "Xfull pair not cross"
    return {
        "classes": classes, "cls_of": cls_of, "units": units,
        "records": records, "w_pairs": w_pairs, "x_pairs": x_pairs,
        "x_dist": x_dist, "pi_w": pi_w, "pi_x": pi_x, "M": M_w,
        "xfull_pairs": xfull_pairs, "pi_xfull": pi_xfull,
        "n_units_raw": nu, "n_units_kept": len(kept_units),
    }


# --------------------------------------------------------------- dose machinery ----

def _dose_slices(base, R_base, backbone, graph_id, draw_seed, grid, prof, strong):
    """Nested restriction of the master-record list to each dose level, with
    certificates (xi)-(xvi) and per-dose pi fingerprints (xviii)."""
    records = base["records"]
    U = len(records)
    if U < 1:
        raise DoseInfeasible("no matched records — dose construction "
                             "undefined (capacity, not corruption)")
    if len({math.ceil(q * U) for q in grid}) < len(grid):
        raise DoseInfeasible(
            f"U={U} collapses the dose grid {grid} (prefix sizes "
            f"{sorted({math.ceil(q * U) for q in grid})}) — strict "
            "f-monotonicity undefined (capacity, not corruption)")
    oseed, sigma = dose_sigma(backbone, graph_id, draw_seed, U)
    # cert (xi): the substream is recomputable from its literal key parts
    oseed2, sigma2 = dose_sigma(backbone, graph_id, draw_seed, U)
    assert oseed2 == oseed and np.array_equal(sigma2, sigma), \
        "order substream not reproducible"
    master_bytes, master_hash = master_serialization(records)
    doses = {}
    prev_sel = set()
    prev_f = -1.0
    for q in grid:
        assert 0.0 < q <= 1.0, f"dose level {q} outside (0,1]"
        k = math.ceil(q * U)
        sel = [int(t) for t in sigma[:k]]
        # cert (xii): selected set == sigma[:ceil(q*U)] exact indices
        assert sel == [int(t) for t in sigma[:k]], "selection != sigma prefix"
        recs = [records[t] for t in sel]
        # cert (xv): retained records carry their literal Xm mapping unchanged
        for t, rec in zip(sel, recs):
            assert rec is records[t], "record identity broken across doses"
        w_pairs_q = [p for r in recs for p in (r["w1"], r["w2"])]
        x_pairs_q = [p for r in recs for p in (r["x1"], r["x2"])]
        pi_w_q = MF._lift(w_pairs_q, R_base)
        pi_x_q = MF._lift(x_pairs_q, R_base)
        M_w_q = MF._certify_lifted(pi_w_q, w_pairs_q, R_base, f"Wm@q{q}")
        M_x_q = MF._certify_lifted(pi_x_q, x_pairs_q, R_base, f"Xm@q{q}")
        # cert (xiv): per-dose moved-set equality across arms
        assert M_w_q == M_x_q, f"moved-set mismatch between arms at q={q}"
        for a, b in x_pairs_q:
            assert MF._cross_ok(prof, R_base, a, b, strong), \
                f"X pair not cross at q={q}"
        # cert (xiii): whole records only — |moved base| == 4*ceil(q*U) and the
        # moved set is exactly the union of the selected records' identity sets
        idset = set()
        for r in recs:
            idset |= {r["w1"][0], r["w1"][1], r["w2"][0], r["w2"][1]}
        assert len(idset) == 4 * k, f"partial/overlapping record at q={q}"
        assert set(M_w_q) == idset, f"moved set != record identity union at q={q}"
        # nesting (strict prefixes)
        assert prev_sel.issubset(set(sel)), f"dose nesting violated at q={q}"
        f_q = len(M_w_q) / max(R_base, 1)
        # cert (xvi): realized dose recomputed, strictly monotone across levels
        assert f_q > prev_f, f"realized f not strictly monotone at q={q}"
        doses[q] = {
            "n_records": k,
            "sel_indices": sel,
            "M": [int(m) for m in sorted(M_w_q)],
            "moved_base": len(M_w_q),
            "f": f_q,
            "pi_w": pi_w_q, "pi_x": pi_x_q,
            "sha": {"Wm": fingerprint(pi_w_q), "Xm": fingerprint(pi_x_q)},
        }
        prev_sel = set(sel)
        prev_f = f_q
    return {
        "U": U, "order_seed": oseed, "sigma": [int(t) for t in sigma],
        "sigma_sha": fingerprint(sigma),
        "master_bytes": master_bytes, "master_hash": master_hash,
        "doses": doses,
    }


def _wcap_anchor(base, R_base, prof):
    """§C-4 Wcap: ALL pre-trim W-units, class-internal transpositions at
    capacity; zero extra legacy RNG. Certified + serialized (cert (xix))."""
    pairs = [(int(a), int(b)) for a, b in base["units"]]
    if not pairs:
        return None
    pi = MF._lift(pairs, R_base)
    M = MF._certify_lifted(pi, pairs, R_base, "Wcap")
    cls_of = base["cls_of"]
    for a, b in pairs:
        assert cls_of[a] == cls_of[b], "Wcap pair crosses class"
        assert MF._profile_l1(prof, a, b, R_base) == 0, \
            "Wcap pair joint profile dist != 0"
    serial = ";".join(f"{a},{b}" for a, b in pairs).encode("ascii")
    return {
        "pairs": pairs, "pi": pi, "sha": fingerprint(pi),
        "serial_hash": hashlib.sha256(serial).hexdigest(),
        "moved_base": len(M), "f": len(M) / max(R_base, 1),
    }


# ------------------------------------------------------------------- public API ----

def build_dose_construction(prof, R_base, draw_seed, backbone, graph_id,
                            dose_grid=None):
    """One draw's full §C construction: legacy arms + master records + nested
    dose slices + anchors + certificates. CertificateError on any violation.

    dose_grid is accepted ONLY as a redundant caller assertion: it must equal
    the frozen DOSE_GRID[backbone] (no production override exists — C1-17-fix).
    Wcap/Xfull anchors exist iff the backbone preset computes Xfull
    (ULTRA/TRIX; §C-4 anchors are ULTRA/TRIX-only)."""
    if backbone not in BACKBONE_PRESET:
        raise SystemExit(f"unknown backbone {backbone!r}")
    preset = BACKBONE_PRESET[backbone]
    grid = DOSE_GRID[backbone]
    if dose_grid is not None:
        try:  # one guarded block: EVERY malformed normalization/comparison
            given = tuple(float(q) for q in tuple(dose_grid))
            matches = (given == grid)
        except Exception as e:  # noqa: BLE001 (C2-16/C3-10-fix)
            raise CertificateError(f"malformed dose_grid {dose_grid!r}: "
                                   f"{e!r}") from e
        if not matches:
            raise CertificateError(
                f"dose grid override {given} != frozen {grid} "
                f"for backbone {backbone!r} (§C-2; no unfrozen grids)")
    try:
        # cert (xx): FLOCK global partition invariant — checked on EVERY build
        # (per-pair L1==0 alone can miss an odd-dropped mismatch; C1-16-fix)
        partition_equal = None
        if backbone == "flock":
            partition_equal = bool(MF._partition_equal(prof, R_base))
            assert partition_equal, \
                "FLOCK joint/base partition mismatch (cert xx)"
        base = _build_legacy_with_records(
            prof, R_base, draw_seed, joint=preset["joint"],
            strong=preset["strong"], compute_xfull=preset["compute_xfull"])
        dose = _dose_slices(base, R_base, backbone, graph_id, draw_seed,
                            grid, prof, preset["strong"])
        # cert (xvii): q=1.00 legacy equality — the top dose reproduces the v3
        # constructor's full-matched permutations byte-for-byte
        top = dose["doses"][1.00]
        assert np.array_equal(top["pi_w"], base["pi_w"]), \
            "q=1.00 pi_W != legacy pi_W"
        assert np.array_equal(top["pi_x"], base["pi_x"]), \
            "q=1.00 pi_X != legacy pi_X"
        wcap = _wcap_anchor(base, R_base, prof) if preset["compute_xfull"] \
            else None
    except AssertionError as e:
        raise CertificateError(f"construction certificate failed: {e}") from e
    xfull = None
    if base["pi_xfull"] is not None:
        xfull = {
            "pairs": [(int(a), int(b)) for a, b in base["xfull_pairs"]],
            "pi": base["pi_xfull"], "sha": fingerprint(base["pi_xfull"]),
            "moved_base": 2 * len(base["xfull_pairs"]),
            "f": 2 * len(base["xfull_pairs"]) / max(R_base, 1),
        }
    x_dist = base["x_dist"]
    return {
        "backbone": backbone, "graph_id": graph_id,
        "draw_seed": int(draw_seed), "dose_grid": grid,
        "R_base": int(R_base), "partition_equal": partition_equal,
        "records": base["records"],
        "U": dose["U"],
        "order_seed": dose["order_seed"],
        "sigma": dose["sigma"], "sigma_sha": dose["sigma_sha"],
        "master_hash": dose["master_hash"],
        "doses": dose["doses"],
        "wcap": wcap, "xfull": xfull,
        "legacy": {
            "pi_w": base["pi_w"], "pi_x": base["pi_x"],
            "pi_xfull": base["pi_xfull"], "M": base["M"],
            "w_pairs": base["w_pairs"], "x_pairs": base["x_pairs"],
        },
        "stats": {
            "n_units_raw": base["n_units_raw"],
            "n_units_kept": base["n_units_kept"],
            "moved_base": len(base["M"]),
            "f_matched": len(base["M"]) / max(R_base, 1),
            "x_l1_median": float(np.median(x_dist)) if x_dist else 0.0,
            "x_l1_q25": float(np.percentile(x_dist, 25)) if x_dist else 0.0,
            "x_l1_q75": float(np.percentile(x_dist, 75)) if x_dist else 0.0,
            "f_Xfull": (2 * len(base["xfull_pairs"]) / max(R_base, 1)
                        if preset["compute_xfull"] else None),
        },
    }


def legacy_arms(prof, R_base, draw_seed, backbone):
    """prereg-v3 make_arms-compatible view (matched_pairs thin delegation).

    C1-1-fix: this path performs the LEGACY construction ONLY — no dose
    slicing, no Wcap, no strict-dose-monotonicity certificates. v3 cells with
    small matched pools (e.g. the real TRIX fb-v2/fb-v3/wk-75 report-only
    cells) are valid v3 constructions that a quartile grid would wrongly
    reject; the frozen v3 semantics must be reproduced verbatim. Byte
    equality with MF.build_matched_pair is enforced by the golden fixtures."""
    if backbone not in BACKBONE_PRESET:
        raise SystemExit(f"unknown backbone {backbone!r}")
    preset = BACKBONE_PRESET[backbone]
    try:
        base = _build_legacy_with_records(
            prof, R_base, draw_seed, joint=preset["joint"],
            strong=preset["strong"], compute_xfull=preset["compute_xfull"])
    except AssertionError as e:
        raise CertificateError(f"constructor certificate failed: {e}") from e
    x_dist = base["x_dist"]
    res = {
        "pi_w": np.asarray(base["pi_w"], dtype=np.int64),
        "pi_x": np.asarray(base["pi_x"], dtype=np.int64),
        "pi_xfull": (None if base["pi_xfull"] is None
                     else np.asarray(base["pi_xfull"], dtype=np.int64)),
        "M": list(map(int, base["M"])),
        "stats": {
            "moved_base": len(base["M"]),
            "f_matched": len(base["M"]) / max(R_base, 1),
            "n_units_raw": base["n_units_raw"],
            "n_units_kept": base["n_units_kept"],
            "x_l1_median": float(np.median(x_dist)) if x_dist else 0.0,
            "x_l1_q25": float(np.percentile(x_dist, 25)) if x_dist else 0.0,
            "x_l1_q75": float(np.percentile(x_dist, 75)) if x_dist else 0.0,
        },
    }
    if preset["compute_xfull"]:
        res["stats"]["f_Xfull"] = 2 * len(base["xfull_pairs"]) / max(R_base, 1)
    res["sha"] = {"Wm": fingerprint(res["pi_w"]), "Xm": fingerprint(res["pi_x"])}
    if res["pi_xfull"] is not None:
        res["sha"]["Xfull"] = fingerprint(res["pi_xfull"])
    return res


ARCHIVE_TOP_KEYS = frozenset({
    "backbone", "graph_id", "draw_seed", "dose_grid", "R_base", "U",
    "order_seed", "sigma_sha", "master_hash", "partition_equal", "doses",
    "wcap_sha", "wcap_serial_hash", "wcap_moved_base", "wcap_f",
    "xfull_sha", "xfull_moved_base", "xfull_f",
})
ARCHIVE_DOSE_KEYS = frozenset({"n_records", "moved_base", "f", "sha"})


def archive_entry(out):
    """JSON-safe fingerprint record for the F-1 pre-archive (§C-6): everything a
    runner must re-verify BEFORE loading any checkpoint. Exact key schema
    (ARCHIVE_TOP_KEYS / ARCHIVE_DOSE_KEYS) — C1-18-fix."""
    wc, xf = out["wcap"], out["xfull"]
    e = {
        "backbone": out["backbone"], "graph_id": out["graph_id"],
        "draw_seed": out["draw_seed"], "dose_grid": list(out["dose_grid"]),
        "R_base": out["R_base"], "U": out["U"],
        "order_seed": out["order_seed"], "sigma_sha": out["sigma_sha"],
        "master_hash": out["master_hash"],
        "partition_equal": out["partition_equal"],
        "doses": {str(q): {"n_records": d["n_records"],
                           "moved_base": d["moved_base"], "f": d["f"],
                           "sha": dict(d["sha"])}
                  for q, d in out["doses"].items()},
        "wcap_sha": None if wc is None else wc["sha"],
        "wcap_serial_hash": None if wc is None else wc["serial_hash"],
        "wcap_moved_base": None if wc is None else wc["moved_base"],
        "wcap_f": None if wc is None else wc["f"],
        "xfull_sha": None if xf is None else xf["sha"],
        "xfull_moved_base": None if xf is None else xf["moved_base"],
        "xfull_f": None if xf is None else xf["f"],
    }
    assert set(e) == ARCHIVE_TOP_KEYS
    return e


def verify_against_archive(prof, R_base, draw_seed, backbone, graph_id,
                           archived):
    """§C-6 pre-model verification: rebuild the construction (frozen grid) and
    deep-compare the FULL archive entry under an exact key schema. Any
    mismatch, missing/extra key, or malformed archive -> CertificateError
    (routes BLOCKED per §A; regeneration after model contact is forbidden).
    Returns the rebuilt construction so runners reuse it without a second
    build."""
    try:
        top_keys = set(archived)
        if top_keys != ARCHIVE_TOP_KEYS:
            raise CertificateError(
                "archive schema violation: missing "
                f"{sorted(ARCHIVE_TOP_KEYS - top_keys)}, unknown "
                f"{sorted(top_keys - ARCHIVE_TOP_KEYS)}")
        for q, d in archived["doses"].items():
            if set(d) != ARCHIVE_DOSE_KEYS:
                raise CertificateError(
                    f"archive dose schema violation at q={q}: "
                    f"{sorted(set(d) ^ ARCHIVE_DOSE_KEYS)}")
    except (TypeError, AttributeError) as e:
        raise CertificateError(f"malformed archive entry: {e!r}") from e
    out = build_dose_construction(prof, R_base, draw_seed, backbone, graph_id)
    got = archive_entry(out)
    if got != archived:
        diffs = [k for k in ARCHIVE_TOP_KEYS if got[k] != archived[k]]
        raise CertificateError(
            "F-1 archive mismatch (pre-model check) in fields "
            f"{sorted(diffs)}: rebuilt "
            f"{ {k: got[k] for k in diffs} } != archived "
            f"{ {k: archived[k] for k in diffs} }")
    return out
