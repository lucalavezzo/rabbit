import numpy as np


class Regularizer:

    def __init__(self, mapping, dtype):
        """
        Initialize the regularization depending on the mapping
        """

    def set_expectations(self, initial_params, initial_observables, parms=None):
        """
        Set the expectations to use in the regularization. Called once per
        parameter layout: the fitter can swap its ParamModel mid-session (the
        saturated goodness-of-fit path wraps it in a CompositeParamModel), which
        reorders and resizes the parameter vector. Do not cache positions across
        calls; re-resolve them by name from ``parms``, the names of every entry
        of ``initial_params``.
        """

    def compute_nll_penalty(self, params, observables):
        """
        Compute the penalty term that gets added to -ln(L), this function should be called in each step of the minimization
        """
        return 0

    def constraint_spec(self, params, observables):
        """Express this regularizer as HARD inequality constraints instead.

        Return ``(values, lower_bounds)``: two 1-D tensors of equal length, the
        feasible set being ``values >= lower_bounds``. A soft regularizer whose
        penalty is a hinge ``relu(lb - c)**2`` has the same feasible set as the
        constraint ``c >= lb``, so the two describe identical physics -- the
        penalty only approaches it from outside, scaled by ``exp(2*tau)``.

        Used by the constrained minimizers (``trust-constr``), which take the
        constraints directly and drop the penalty from the loss. That removes
        the ``exp(2*tau)`` stiffness from the objective, and with
        ``keep_feasible`` it keeps every iterate inside the feasible region --
        which matters when leaving it produces a NaN likelihood rather than
        merely a large one.

        Returning ``None`` (the default) means this regularizer can only be
        expressed as a penalty; the fitter then refuses the constrained path
        rather than silently dropping it.
        """
        return None

    @staticmethod
    def resolve_indices(parms, names, who="Regularizer"):
        """Map parameter names to positions in ``parms``, raising if any is absent."""
        if parms is None:
            raise ValueError(
                f"{who}: parameter names are required to resolve positions"
            )
        parms = np.asarray(parms).astype(str)
        index = {name: i for i, name in enumerate(parms)}
        missing = [n for n in names if n not in index]
        if missing:
            raise ValueError(f"{who}: {missing} not in the fit's parameter vector")
        return {n: index[n] for n in names}
