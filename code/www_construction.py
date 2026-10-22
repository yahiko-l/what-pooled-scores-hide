"""Displacement-matched arms with edgeless relations excluded ("populated").

A base relation is edgeless when its typed degree profile is all zero on both
the base and the inverse side: it has no edge in the graph the profile was
computed from. The legacy constructor (matched_construction.legacy_arms, via
matched_pairs.make_arms) puts all edgeless relations in one zero-profile role
class, so they enter W-units, and X-matching pairs them with populated
relations. populated_arms removes them from the role classes before the
W-units are formed and is otherwise the legacy construction: the same numpy
call sequence on default_rng(draw_seed) as
matched_construction._build_legacy_with_records, the same certificates, and
Xfull matched over populated relations only. On a graph without edgeless
relations it returns the legacy arms byte for byte.
"""
import numpy as np

import matched_construction as MC
import matched_feasibility as MF
import matched_pairs as MP

CONSTRUCTIONS = ("legacy", "populated")


def edgeless_base(prof, R_base):
    return {r for r in range(R_base)
            if int(prof[r].sum()) == 0 and int(prof[r + R_base].sum()) == 0}


def _build_populated(prof, R_base, seed, joint, strong, compute_xfull, edgeless):
    rng = np.random.default_rng(seed)
    classes = {key: [r for r in members if r not in edgeless]
               for key, members in MF._classes_of(prof, R_base, joint=joint).items()}
    units = MF._w_units(classes, rng)                                    # legacy #1
    nu = len(units)
    cls_of = {r: key for key, members in classes.items() for r in members}
    adj = [[] for _ in range(nu)]
    for i in range(nu):
        for j in range(i + 1, nu):
            if cls_of[units[i][0]] == cls_of[units[j][0]]:
                continue
            if MF._unit_pair_options(prof, R_base, units[i], units[j], strong):
                adj[i].append(j)
                adj[j].append(i)
    matching = MF._greedy_maximal_matching(nu, adj, rng)                # legacy #2
    kept_units = sorted({i for ij in matching for i in ij})
    x_pairs = []
    for i, j in matching:
        opts = MF._unit_pair_options(prof, R_base, units[i], units[j], strong)
        x_pairs.extend(opts[int(rng.integers(len(opts)))])             # legacy #3
    w_pairs = [units[i] for i in kept_units]
    pi_w = MF._lift(w_pairs, R_base)
    pi_x = MF._lift(x_pairs, R_base)
    M_w = MF._certify_lifted(pi_w, w_pairs, R_base, "Wm")
    M_x = MF._certify_lifted(pi_x, x_pairs, R_base, "Xm")
    assert M_w == M_x, "moved-set mismatch between arms"
    assert not set(M_w) & edgeless, "edgeless relation in the moved set"
    for a, b in w_pairs:
        assert cls_of[a] == cls_of[b], "W pair crosses class"
        assert MF._profile_l1(prof, a, b, R_base) == 0, "W pair joint profile dist != 0"
    for a, b in x_pairs:
        assert MF._cross_ok(prof, R_base, a, b, strong), "X pair not cross"
    x_dist = [MF._profile_l1(prof, a, b, R_base) for a, b in x_pairs]
    xfull_pairs, pi_xfull = [], None
    if compute_xfull:
        # Edgeless nodes stay in the node set with no neighbours, so the
        # matching consumes the legacy random stream when none exist.
        all_adj = [[] for _ in range(R_base)]
        for x in range(R_base):
            if x in edgeless:
                continue
            for y in range(x + 1, R_base):
                if y not in edgeless and MF._cross_ok(prof, R_base, x, y, strong):
                    all_adj[x].append(y)
                    all_adj[y].append(x)
        xfull_pairs = MF._greedy_maximal_matching(R_base, all_adj, rng)   # legacy #4
        pi_xfull = MF._lift(xfull_pairs, R_base)
        MF._certify_lifted(pi_xfull, xfull_pairs, R_base, "Xfull")
        for a, b in xfull_pairs:
            assert MF._cross_ok(prof, R_base, a, b, strong), "Xfull pair not cross"
            assert a not in edgeless and b not in edgeless, "edgeless relation in Xfull"
    res = {
        "pi_w": np.asarray(pi_w, dtype=np.int64),
        "pi_x": np.asarray(pi_x, dtype=np.int64),
        "pi_xfull": None if pi_xfull is None else np.asarray(pi_xfull, dtype=np.int64),
        "M": list(map(int, M_w)),
        "stats": {
            "moved_base": len(M_w),
            "f_matched": len(M_w) / max(R_base, 1),
            "n_units_raw": nu,
            "n_units_kept": len(kept_units),
            "x_l1_median": float(np.median(x_dist)) if x_dist else 0.0,
            "x_l1_q25": float(np.percentile(x_dist, 25)) if x_dist else 0.0,
            "x_l1_q75": float(np.percentile(x_dist, 75)) if x_dist else 0.0,
        },
    }
    if compute_xfull:
        res["stats"]["f_Xfull"] = 2 * len(xfull_pairs) / max(R_base, 1)
    res["sha"] = {"Wm": MC.fingerprint(res["pi_w"]), "Xm": MC.fingerprint(res["pi_x"])}
    if res["pi_xfull"] is not None:
        res["sha"]["Xfull"] = MC.fingerprint(res["pi_xfull"])
    return res


def populated_arms(prof, R_base, draw_seed, backbone):
    """matched_pairs.make_arms-compatible arms over populated relations only."""
    if backbone not in MC.BACKBONE_PRESET:
        raise SystemExit(f"unknown backbone {backbone!r}")
    preset = MC.BACKBONE_PRESET[backbone]
    try:
        return _build_populated(prof, R_base, draw_seed, preset["joint"], preset["strong"],
                                preset["compute_xfull"], edgeless_base(prof, R_base))
    except AssertionError as e:
        raise MC.CertificateError(f"populated constructor certificate failed: {e}") from e


def make_arms(construction, prof, R_base, draw_seed, backbone):
    if construction == "legacy":
        return MP.make_arms(prof, R_base, draw_seed, backbone)
    if construction == "populated":
        return populated_arms(prof, R_base, draw_seed, backbone)
    raise SystemExit(f"unknown construction {construction!r}; choose from {CONSTRUCTIONS}")
