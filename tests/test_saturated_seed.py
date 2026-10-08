"""--saturatedSeed: seed the saturated sub-fit by parameter name.

The sub-fit of --computeSaturatedProjectionTests starts from the main fit with
every bin scale at 1. When the projection is in tension the bin scales have far
to travel and the sub-fit can take hours; a converged sub-fit of a nearby model
is a much better start. The seed is applied by NAME (layouts differ between
fits), only to the selected entries, and kept only if it does not raise the
starting loss.
"""

import ast
import importlib.util
import pathlib

import numpy as np
import tensorflow as tf

from rabbit import snapshot

DRIVER = pathlib.Path(__file__).resolve().parents[1] / "bin" / "rabbit_fit.py"


def test_seed_by_name_matches_names_not_positions():
    target = ["a", "sat0", "b", "sat1", "c"]
    x = np.arange(5.0)
    seed_names = [
        "sat1",
        "zzz",
        "b",
        "sat0",
    ]  # different order, an extra, a missing one
    seed_vals = [10.0, 99.0, 20.0, 30.0]
    out, seeded = snapshot.seed_by_name(target, x, seed_names, seed_vals)
    assert out.tolist() == [0.0, 30.0, 20.0, 10.0, 4.0]
    assert sorted(seeded) == ["b", "sat0", "sat1"]
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]  # input untouched


def test_seed_by_name_select():
    out, seeded = snapshot.seed_by_name(
        ["a", "sat0"],
        [1.0, 1.0],
        ["a", "sat0"],
        [5.0, 7.0],
        select=lambda n: n.startswith("sat"),
    )
    assert out.tolist() == [1.0, 7.0] and seeded == ["sat0"]


def test_read_vector_snapshot(tmp_path):
    p = tmp_path / "s.hdf5"
    snapshot.write_snapshot(
        str(p), ["a", "b"], [0.5, -1.5], meta=dict(reason="converged")
    )
    names, vals = snapshot.read_vector(str(p))
    assert list(names) == ["a", "b"] and vals.tolist() == [0.5, -1.5]


def _driver_module():
    spec = importlib.util.spec_from_file_location("rabbit_fit_driver", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeFitter:
    """x, parms and a loss: the three things seed_saturated_fit touches."""

    def __init__(self, names, x, loss):
        self.parms = np.array(names)
        self.x = tf.Variable(np.asarray(x, np.float64))
        self._loss = loss

    def _compute_loss(self, profile=True):
        return self._loss(self.x.numpy())


def test_seed_kept_when_it_lowers_the_loss(tmp_path):
    drv = _driver_module()
    p = tmp_path / "seed.hdf5"
    snapshot.write_snapshot(str(p), ["m", "sat0", "sat1"], [9.0, 2.0, 3.0])
    loss = (
        lambda x: (x[1] - 2.0) ** 2 + (x[2] - 3.0) ** 2 + (x[0] - 1.0) ** 2
    )  # noqa: E731
    f = _FakeFitter(["m", "sat0", "sat1"], [1.0, 1.0, 1.0], loss)
    drv.seed_saturated_fit(f, str(p), "projection", ["sat0", "sat1"])
    # projection scope: the model parameter 'm' keeps the main-fit value
    assert f.x.numpy().tolist() == [1.0, 2.0, 3.0]
    f = _FakeFitter(["m", "sat0", "sat1"], [1.0, 1.0, 1.0], lambda x: (x[0] - 9.0) ** 2)
    drv.seed_saturated_fit(f, str(p), "all", ["sat0", "sat1"])
    assert f.x.numpy().tolist() == [9.0, 2.0, 3.0]


def test_seed_rejected_when_it_raises_the_loss(tmp_path):
    drv = _driver_module()
    p = tmp_path / "seed.hdf5"
    snapshot.write_snapshot(str(p), ["sat0"], [5.0])
    f = _FakeFitter(["m", "sat0"], [1.0, 1.0], lambda x: (x[1] - 1.0) ** 2)
    drv.seed_saturated_fit(f, str(p), "projection", ["sat0"])
    assert f.x.numpy().tolist() == [1.0, 1.0]


def test_driver_seeds_after_the_warm_start_and_before_minimize():
    src = DRIVER.read_text()
    tree = ast.parse(src)
    body = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "save_hists"
    )
    text = ast.get_source_segment(src, body)
    warm = text.find("fitter_saturated.x[composite_model.nparams :].assign(")
    seed = text.find("seed_saturated_fit(")
    mini = text.find("cb = fitter_saturated.minimize()")
    assert -1 not in (warm, seed, mini)
    assert warm < seed < mini
