import re

import h5py
import hist
import numpy as np

from wums import ioutils  # isort: skip


def get_fitresult(fitresult_filename, result=None, meta=False):
    if isinstance(fitresult_filename, str):
        h5file = h5py.File(fitresult_filename, mode="r")
    else:
        h5file = fitresult_filename
    key = "results"
    if result is not None and result is not "":
        key = f"{key}_{result}"
    elif key not in h5file.keys():  # fallback in case only asimov was fit
        key = f"{key}_asimov"
    if key not in h5file.keys():
        raise ValueError(f"'{key}' not in h5file, available keys are {h5file.keys()}")
    h5results = ioutils.pickle_load_h5py(h5file[key])
    if meta:
        meta = ioutils.pickle_load_h5py(h5file["meta"])
        return h5results, meta
    return h5results


def get_poi_names(meta):
    return np.concatenate((meta.get("pois", meta.get("signals")), meta["nois"])).astype(
        str
    )


def get_syst_labels(fitresult):
    h = fitresult["parms"].get()
    return np.array(h.axes["parms"])


def read_impacts_poi(
    fitresult,
    poi,
    grouped=False,
    impact_type="traditional",
    pulls=False,
    add_total=True,
    asym=False,
):
    # read impacts of a single POI

    if asym and impact_type == "traditional" and "impacts_asym" not in fitresult.keys():
        # Fallback: read asymmetric traditional impacts from a generic
        # contour scan output if --asymImpacts wasn't run.
        h_impacts = fitresult["contour_scans"].get()[{"confidence_level": "1.0"}]
    else:
        impact_name = "impacts"
        if impact_type != "traditional":
            impact_name = f"{impact_type}_{impact_name}"
        if asym:
            impact_name += "_asym"
        if grouped:
            impact_name += "_grouped"

        h_impacts = fitresult[impact_name].get()

    h_impacts = h_impacts[{"parms": poi}]

    impacts = h_impacts.values()
    labels = np.array(h_impacts.axes["impacts"])

    # if add_total and poi not in labels:
    if add_total:
        h_parms = fitresult["parms"].get()
        total = np.sqrt(h_parms[{"parms": poi}].variance)
        impacts = np.append(impacts, total)
        labels = np.append(labels, "Total")

    if pulls:
        pulls_labels, pulls, constraints = get_pulls_and_constraints(
            fitresult, asym=asym and impact_type == "traditional"
        )
        pulls_labels, pulls_prefit, constraints_prefit = get_pulls_and_constraints(
            fitresult, asym=asym and impact_type == "traditional", prefit=True
        )

        if len(pulls_labels) != len(labels):
            mask = [l in labels for l in pulls_labels]
            pulls = pulls[mask]
            pulls_prefit = pulls_prefit[mask]
            constraints = constraints[mask]
            constraints_prefit = constraints_prefit[mask]
        return pulls, pulls_prefit, constraints, constraints_prefit, impacts, labels

    return impacts, labels


def _filter_nuisance_data(
    labels,
    pulls,
    constraints,
    keep_patterns=None,
    exclude_patterns=None,
):
    if keep_patterns is None and exclude_patterns is None:
        return labels, pulls, constraints

    if isinstance(keep_patterns, str):
        keep_patterns = [keep_patterns]
    if isinstance(exclude_patterns, str):
        exclude_patterns = [exclude_patterns]

    keep_patterns = keep_patterns or []
    exclude_patterns = exclude_patterns or []

    def matches_any(patterns, label):
        return any(re.search(pattern, label) for pattern in patterns)

    mask = np.ones(len(labels), dtype=bool)
    if keep_patterns:
        mask &= np.array([matches_any(keep_patterns, label) for label in labels])
    if exclude_patterns:
        mask &= ~np.array([matches_any(exclude_patterns, label) for label in labels])

    filtered_labels = labels[mask]
    filtered_pulls = pulls[mask]

    if np.ndim(constraints) == 0:
        filtered_constraints = constraints
    else:
        indices = np.nonzero(mask)[0]
        filtered_constraints = np.take(constraints, indices, axis=0)

    return filtered_labels, filtered_pulls, filtered_constraints


def get_pulls_and_constraints(
    fitresult,
    prefit=False,
    asym=False,
    keep_nuisances=None,
    exclude_nuisances=None,
):
    hist_name = "parms_prefit" if prefit else "parms"
    h_parms = fitresult[hist_name].get()
    labels = np.array(h_parms.axes["parms"])
    pulls = h_parms.values()

    if asym:
        h_intervals = fitresult["contour_scans"].get()
        intervals = h_intervals[{"confidence_level": "1.0"}].values()
        constraints = np.einsum("i j i -> i j", intervals)
    else:
        constraints = np.sqrt(h_parms.variances())

    labels, pulls, constraints = _filter_nuisance_data(
        labels,
        pulls,
        constraints,
        keep_patterns=keep_nuisances,
        exclude_patterns=exclude_nuisances,
    )

    return labels, pulls, constraints


def get_postfit_hist_cov(fitresult, mapping="BaseMapping", channels=None):
    """
    Return postfit histogram and covariance matrix from selected channels (all if channel is None)
    """
    print(f"Load postfit histogram and covariance matrix")

    result = fitresult.get("mappings", fitresult.get("physics_models"))
    if mapping not in result.keys():
        raise IOError(
            f"{mapping} not found in fitresults, available mappings are {result.keys()}"
        )
    result = result[mapping]

    cov = result["hist_postfit_inclusive_cov"].get().values()
    if channels is not None:
        found_channels = [c for c in result["channels"].keys() if c in channels]
        if list(channels) != list(found_channels):
            raise RuntimeError(
                f"Not all channels found in fitresult or the order is wrong, requested: {channels} and found {found_channels}. Available: {result['channels'].keys()}."
            )
        h_data = [
            result["channels"][c]["hist_postfit_inclusive"].get() for c in channels
        ]

        # select submatric that corresponds to selected channels
        channel_idxs = [
            i for i, c in enumerate(result["channels"].keys()) if c in channels
        ]

        stops = np.cumsum(
            [
                len(c["hist_postfit_inclusive"].get().values().flatten())
                for c in result["channels"].values()
            ]
        )
        starts = np.array([0, *stops[:-1]])
        starts = starts[channel_idxs]
        stops = stops[channel_idxs]

        idxs = np.concat([np.arange(s, e) for s, e in zip(starts, stops)])

        cov = cov[np.ix_(idxs, idxs)]
    else:
        found_channels = [c for c in result["channels"].keys()]
        h_data = [
            c["hist_postfit_inclusive"].get() for k, c in result["channels"].items()
        ]

    return h_data, cov, found_channels


# =============================================================================
# Merging fitresults files
# =============================================================================
# rabbit_fit writes one fitresults.hdf5 per invocation, while every consumer
# (rabbit_plot_hists, rabbit_print_*) reads a single file. merge_fitresults()
# recombines results spread over several invocations, content-agnostically:
# results groups with the same name are recursively unioned (guarded to come
# from the same postfit point), groups unique to one input are copied through.
# Typical uses: partial post-processing runs (the --externalPostfit --noFit
# pattern splits expensive covariance / impacts / hist errors / saturated
# tests across runs), runs each computing a different mapping/projection, or
# packaging fits of different datasets (data + asimov) into one multi-result
# file.


def _proxy_value(v):
    return v.get() if isinstance(v, ioutils.H5PickleProxy) else v


# Leaf-merge semantics: rabbit encodes "not computed" as NaN (e.g. parms
# variances / edmval in a --noHessian run) or an absent/None entry, so a
# computed value FILLS a missing one instead of conflicting — only
# computed-vs-computed disagreements are conflicts. Float comparisons use a
# relative tolerance: the same quantity evaluated at the same postfit point
# through different computation graphs (exact fold vs straight-through,
# different op ordering) differs by last-ulp noise, which must not read as a
# conflict; genuine disagreements are many orders of magnitude larger.
MERGE_RTOL = 1e-9


def _fill_close(a, b, rtol):
    """Elementwise merge of two same-shape arrays: NaN entries are filled from
    the other side; entries finite on both sides must agree within rtol.
    Returns (merged_array, conflict)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return a, True
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        return a, not np.array_equal(a, b)
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    both = ~a_nan & ~b_nan
    if not np.allclose(a[both], b[both], rtol=rtol, atol=0.0):
        return a, True
    return np.where(a_nan, b, a), False


def _hist_variances(h):
    """Computed variances of a hist, or None. Double storage is 'no variances
    computed': its variances() returns the VALUES (Poisson assumption), not
    stored uncertainties — e.g. hists saved without --computeHistErrors — so
    only Weight storage counts as carrying variances."""
    if getattr(h, "storage_type", None) != hist.storage.Weight:
        return None
    return h.variances()


def _merge_hists(ha, hb, rtol):
    """Merge two hists: values and variances via _fill_close, a variance set
    absent on one side taken from the other. Returns (hist, conflict); the
    returned hist IS ``ha`` when nothing was filled (fast path, keeps the
    original object/proxy upstream)."""
    if ha.axes != hb.axes:
        return ha, True
    vals, c_vals = _fill_close(ha.values(), hb.values(), rtol)
    if c_vals:
        return ha, True
    va = _hist_variances(ha)
    vb = _hist_variances(hb)
    if va is None or vb is None:
        var = va if vb is None else vb
    else:
        var, c_var = _fill_close(va, vb, rtol)
        if c_var:
            return ha, True
    unchanged = np.array_equal(vals, ha.values(), equal_nan=True) and (
        (var is None and va is None)
        or (
            var is not None
            and va is not None
            and np.array_equal(var, va, equal_nan=True)
        )
    )
    if unchanged:
        return ha, False
    storage = hist.storage.Weight() if var is not None else hist.storage.Double()
    h = hist.Hist(*ha.axes, storage=storage, name=ha.name, label=ha.label)
    h.values()[...] = vals
    if var is not None:
        h.variances()[...] = var
    return h, False


def _merge_leaf(a, b, rtol):
    """Merge two results-dict leaves. Returns (merged_value, conflict); on
    conflict the first value is returned (the prefer policy decides upstream).
    Proxies are kept whenever their content is unchanged; a hist rebuilt with
    filled entries is re-wrapped so readers can keep calling .get()."""
    av = _proxy_value(a)
    bv = _proxy_value(b)
    if av is None:
        return b, False
    if bv is None:
        return a, False
    if hasattr(av, "values") and hasattr(bv, "values"):  # hist-like
        merged, conflict = _merge_hists(av, bv, rtol)
        if conflict:
            return a, True
        if merged is av:
            return a, False
        return ioutils.H5PickleProxy(merged), False
    if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
        merged, conflict = _fill_close(av, bv, rtol)
        return (a, True) if conflict else (merged, False)
    if isinstance(av, (float, np.floating)) and isinstance(bv, (float, np.floating)):
        if np.isnan(av):
            return b, False
        if np.isnan(bv):
            return a, False
        return a, not np.isclose(av, bv, rtol=rtol, atol=0.0)
    try:
        return a, not bool(av == bv)
    except Exception:
        return a, True


def _materialize_proxies(obj):
    """Recursively force-load every H5PickleProxy. Required before re-dumping
    a loaded results dict: wums pickles proxy.obj, which is None while lazy."""
    if isinstance(obj, ioutils.H5PickleProxy):
        obj.get()
    elif isinstance(obj, dict):
        for v in obj.values():
            _materialize_proxies(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _materialize_proxies(v)


def merge_results_dicts(dicts, prefer="error", rtol=MERGE_RTOL):
    """Recursive union of results dicts (later inputs merged onto earlier).

    Keys present in one input are taken as-is; nested dicts are merged
    recursively. Leaves present in several inputs are merged with rabbit's
    not-computed semantics (NaN/None filled from the computed side, floats
    compared within ``rtol``); genuinely differing leaves are conflicts,
    resolved by ``prefer``: "first" keeps the earlier value, "last" the later,
    "error" collects them.

    Returns (merged_dict, conflict_paths). With prefer="error" the caller is
    expected to raise if conflict_paths is non-empty.
    """

    def rec(a, b, path):
        out = dict(a)
        conflicts = []
        for k, vb in b.items():
            if k not in out:
                out[k] = vb
                continue
            va = out[k]
            if isinstance(va, dict) and isinstance(vb, dict):
                out[k], sub = rec(va, vb, path + [k])
                conflicts += sub
            else:
                merged, conflict = _merge_leaf(va, vb, rtol)
                if conflict:
                    conflicts.append("/".join(path + [k]))
                    if prefer == "last":
                        out[k] = vb
                else:
                    out[k] = merged
        return out, conflicts

    merged, conflicts = dicts[0], []
    for d in dicts[1:]:
        merged, sub = rec(merged, d, [])
        conflicts += sub
    return merged, conflicts


def _parms_values(results_dict):
    p = results_dict.get("parms")
    return None if p is None else np.asarray(_proxy_value(p).values())


def merge_fitresults(inputs, output, prefer="error", force=False, rtol=MERGE_RTOL):
    """Merge several fitresults.hdf5 files into ``output``.

    inputs
        Paths, in precedence order (relevant for prefer="first"/"last").
    prefer
        Conflict policy for genuinely differing leaves within same-name results
        groups (after NaN/None filling and the ``rtol`` float comparison):
        "error" (default) raises listing every conflicting path, "first"/"last"
        keep the earlier/later input's value.
    force
        Skip the same-postfit guard (same-name results groups must otherwise
        agree on their stored parms values).
    rtol
        Relative tolerance for float leaf comparisons (see MERGE_RTOL).

    Top-level non-results datasets (x, parms, cov, ...) are copied first-input-
    wins. The merged meta is the first input's, with a ``merged_inputs``
    provenance list (file, command, time of every input).
    """
    files = [h5py.File(p, mode="r") for p in inputs]
    try:
        all_results = []  # per input: {group_name: results dict}
        metas = []
        for f in files:
            groups = {
                k: ioutils.pickle_load_h5py(f[k])
                for k in f.keys()
                if k.startswith("results")
            }
            if not groups:
                raise ValueError(
                    f"{f.filename} has no results* group (aborted-fit stub?)"
                )
            all_results.append(groups)
            metas.append(
                ioutils.pickle_load_h5py(f["meta"]) if "meta" in f.keys() else {}
            )

        group_names = []
        for groups in all_results:
            group_names += [g for g in groups.keys() if g not in group_names]

        merged_groups = {}
        for gname in group_names:
            members = [g[gname] for g in all_results if gname in g]
            if len(members) > 1 and not force:
                ref = None
                for m in members:
                    vals = _parms_values(m)
                    if vals is None:
                        continue
                    if ref is None:
                        ref = vals
                    elif ref.shape != vals.shape or not np.array_equal(ref, vals):
                        raise ValueError(
                            f"'{gname}': inputs disagree on the postfit parameter "
                            "values — not the same postfit point (--force to "
                            "override)."
                        )
            merged, conflicts = merge_results_dicts(members, prefer=prefer, rtol=rtol)
            if conflicts and prefer == "error":
                raise ValueError(
                    f"'{gname}': conflicting entries (use --prefer first/last):\n  "
                    + "\n  ".join(conflicts)
                )
            merged_groups[gname] = merged

        meta = metas[0]
        meta["merged_inputs"] = [
            {
                "file": fname,
                "command": (m.get("meta_info", {}) or {}).get("command", ""),
                "time": (m.get("meta_info", {}) or {}).get("time", ""),
            }
            for fname, m in zip(inputs, metas)
        ]

        for merged in merged_groups.values():
            _materialize_proxies(merged)

        with h5py.File(output, "w") as fout:
            rewritten = set(merged_groups) | {"meta"}
            for f in files:
                for k in f.keys():
                    if k not in rewritten and k not in fout.keys():
                        f.copy(k, fout)
            ioutils.pickle_dump_h5py("meta", meta, fout)
            for gname, merged in merged_groups.items():
                ioutils.pickle_dump_h5py(gname, merged, fout)
    finally:
        for f in files:
            f.close()

    return merged_groups
