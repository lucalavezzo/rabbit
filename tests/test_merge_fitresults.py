"""Test merging of fitresults files (io_tools.merge_fitresults).

Covers:
  * union of complementary partial outputs (the --externalPostfit --noFit
    split: one file with mapping hists, one with the saturated-test scalars)
    including nested mappings/channels dicts and lazy H5PickleProxy hists;
  * equal entries present in both inputs pass silently (validation for free);
  * conflicting entries error by default and resolve with prefer first/last;
  * the same-postfit guard (differing parms) raises, force=True overrides;
  * results groups with disjoint names are packaged side by side;
  * merged meta carries the merged_inputs provenance.

No fit is run: the files are built directly in the Workspace layout
(meta + results* group of nested dicts with H5PickleProxy-wrapped hists),
so the test stays TensorFlow-free and fast.
"""

import h5py
import hist
import numpy as np
import pytest

from rabbit import io_tools

from wums import ioutils  # isort: skip


def make_parms_hist(values):
    ax = hist.axis.StrCategory([f"p{i}" for i in range(len(values))], name="parms")
    h = hist.Hist(ax, storage=hist.storage.Weight(), name="parms")
    h.values()[...] = values
    h.variances()[...] = 1.0
    return h


def make_yield_hist(values, name):
    ax = hist.axis.Regular(len(values), 0.0, 1.0, name="x")
    h = hist.Hist(ax, storage=hist.storage.Weight(), name=name)
    h.values()[...] = values
    h.variances()[...] = np.abs(values)
    return h


def write_fitresults(path, results, group="results", command="cmd"):
    with h5py.File(path, "w") as f:
        ioutils.pickle_dump_h5py(
            "meta", {"meta_info": {"command": command, "time": "t"}}, f
        )
        ioutils.pickle_dump_h5py(group, results, f)


def base_results(parms_values=(1.0, 2.0)):
    """A cov-pass-like results dict: parms + a mapping with channel hists."""
    return {
        "parms": ioutils.H5PickleProxy(make_parms_hist(list(parms_values))),
        "nllvalreduced": 12.5,
        "mappings": {
            "Project ch0 x": {
                "channels": {
                    "ch0": {
                        "hist_postfit_inclusive": ioutils.H5PickleProxy(
                            make_yield_hist(np.array([10.0, 20.0, 30.0]), "postfit")
                        ),
                    }
                },
                "chi2": 3.0,
                "ndf": 3,
            }
        },
    }


def saturated_results(parms_values=(1.0, 2.0), nll=12.5):
    """A saturated-step-like results dict: same postfit, saturated scalars."""
    return {
        "parms": ioutils.H5PickleProxy(make_parms_hist(list(parms_values))),
        "nllvalreduced": nll,
        "mappings": {
            "Project ch0 x": {
                "chi2_saturated": 7.5,
                "ndf_saturated": 3,
            }
        },
    }


def test_union_partial_outputs(tmpdir):
    f_cov = str(tmpdir / "cov.hdf5")
    f_sat = str(tmpdir / "sat.hdf5")
    f_out = str(tmpdir / "merged.hdf5")
    write_fitresults(f_cov, base_results(), command="cov cmd")
    write_fitresults(f_sat, saturated_results(), command="sat cmd")

    io_tools.merge_fitresults([f_cov, f_sat], f_out)

    with h5py.File(f_out, "r") as f:
        res = ioutils.pickle_load_h5py(f["results"])
        mapping = res["mappings"]["Project ch0 x"]
        # union: hists from cov, saturated scalars from sat, shared keys once
        assert mapping["chi2"] == 3.0
        assert mapping["chi2_saturated"] == 7.5
        assert mapping["ndf_saturated"] == 3
        h = mapping["channels"]["ch0"]["hist_postfit_inclusive"].get()
        assert h is not None  # lazy proxies must be materialized on re-dump
        np.testing.assert_array_equal(h.values(), [10.0, 20.0, 30.0])
        # equal duplicate entries (nllvalreduced, parms) pass silently
        assert res["nllvalreduced"] == 12.5
        meta = ioutils.pickle_load_h5py(f["meta"])
        assert [m["command"] for m in meta["merged_inputs"]] == ["cov cmd", "sat cmd"]


def test_conflict_policies(tmpdir):
    f_a = str(tmpdir / "a.hdf5")
    f_b = str(tmpdir / "b.hdf5")
    write_fitresults(f_a, base_results())
    write_fitresults(f_b, saturated_results(nll=99.0))  # conflicting leaf

    with pytest.raises(ValueError, match="nllvalreduced"):
        io_tools.merge_fitresults([f_a, f_b], str(tmpdir / "err.hdf5"))

    io_tools.merge_fitresults([f_a, f_b], str(tmpdir / "first.hdf5"), prefer="first")
    with h5py.File(str(tmpdir / "first.hdf5"), "r") as f:
        assert ioutils.pickle_load_h5py(f["results"])["nllvalreduced"] == 12.5

    io_tools.merge_fitresults([f_a, f_b], str(tmpdir / "last.hdf5"), prefer="last")
    with h5py.File(str(tmpdir / "last.hdf5"), "r") as f:
        assert ioutils.pickle_load_h5py(f["results"])["nllvalreduced"] == 99.0


def test_postfit_guard(tmpdir):
    f_a = str(tmpdir / "a.hdf5")
    f_b = str(tmpdir / "b.hdf5")
    write_fitresults(f_a, base_results(parms_values=(1.0, 2.0)))
    write_fitresults(f_b, saturated_results(parms_values=(1.0, 2.5)))

    with pytest.raises(ValueError, match="postfit parameter values"):
        io_tools.merge_fitresults([f_a, f_b], str(tmpdir / "err.hdf5"))

    # force skips the guard; the differing parms leaf then needs a policy too
    io_tools.merge_fitresults(
        [f_a, f_b], str(tmpdir / "forced.hdf5"), force=True, prefer="first"
    )
    with h5py.File(str(tmpdir / "forced.hdf5"), "r") as f:
        res = ioutils.pickle_load_h5py(f["results"])
        np.testing.assert_array_equal(res["parms"].get().values(), [1.0, 2.0])


def test_nan_variances_not_a_conflict(tmpdir):
    # rabbit writes NaN parms variances in --noHessian runs (uncomputed
    # uncertainties); two such runs on the same postfit must merge cleanly
    def results_with_nan_parms():
        h = make_parms_hist([1.0, 2.0])
        h.variances()[...] = np.nan
        return {
            "parms": ioutils.H5PickleProxy(h),
            "edmval": float("nan"),  # NaN scalar leaf must not conflict either
        }

    f_a = str(tmpdir / "a.hdf5")
    f_b = str(tmpdir / "b.hdf5")
    write_fitresults(f_a, results_with_nan_parms())
    write_fitresults(f_b, {**results_with_nan_parms(), "chi2_saturated": 5.0})

    f_out = str(tmpdir / "merged.hdf5")
    io_tools.merge_fitresults([f_a, f_b], f_out)  # prefer="error": must not raise
    with h5py.File(f_out, "r") as f:
        res = ioutils.pickle_load_h5py(f["results"])
        assert res["chi2_saturated"] == 5.0
        assert np.all(np.isnan(res["parms"].get().variances()))


def test_disjoint_groups(tmpdir):
    f_a = str(tmpdir / "a.hdf5")
    f_b = str(tmpdir / "b.hdf5")
    # different datasets: parms may differ freely across group names
    write_fitresults(f_a, base_results(parms_values=(1.0, 2.0)), group="results")
    write_fitresults(f_b, base_results(parms_values=(3.0, 4.0)), group="results_asimov")

    f_out = str(tmpdir / "merged.hdf5")
    io_tools.merge_fitresults([f_a, f_b], f_out)

    with h5py.File(f_out, "r") as f:
        assert "results" in f.keys() and "results_asimov" in f.keys()
        res_a = ioutils.pickle_load_h5py(f["results"])
        res_b = ioutils.pickle_load_h5py(f["results_asimov"])
        np.testing.assert_array_equal(res_a["parms"].get().values(), [1.0, 2.0])
        np.testing.assert_array_equal(res_b["parms"].get().values(), [3.0, 4.0])
