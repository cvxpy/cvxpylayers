"""Custom DGP to DCP reduction that works without parameter values.

This module provides a custom implementation of CVXPY's DGP→DCP reduction
that allows cvxpylayers to build computation graphs without requiring
parameter values to be set upfront.
"""

from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy.reductions.dgp2dcp.canonicalizers import DgpCanonMethods


class _DgpCanonMethodsNoValueCheck(DgpCanonMethods):  # type: ignore[misc]
    """Custom DGP canonicalization methods that work without parameter values."""

    def parameter_canon(
        self, parameter: cp.Parameter, args: list[Any]
    ) -> tuple[cp.Parameter, list[Any]]:
        """Canonicalize a parameter without requiring it to have a value.

        Args:
            parameter: The parameter to canonicalize
            args: Arguments (unused)

        Returns:
            Tuple of (log-space parameter, constraints)
        """
        del args
        # Swaps out positive parameters for unconstrained parameters.
        if parameter in self._parameters:
            return self._parameters[parameter], []
        else:
            # Create log-space parameter, preserving None value if present
            log_parameter = cp.Parameter(
                parameter.shape,
                name=parameter.name(),
                value=np.log(parameter.value) if parameter.value is not None else None,
            )
            self._parameters[parameter] = log_parameter
            return log_parameter, []


class _Dgp2DcpNoValueCheck(cp.reductions.Dgp2Dcp):  # type: ignore[misc]
    """DGP to DCP reduction that works without parameter values.

    This is an internal cvxpylayers class that bypasses CVXPY's requirement
    for parameters to have values during the DGP→DCP transformation.

    CVXPY's Dgp2Dcp.accepts() checks that all parameters have values, but
    this is unnecessary - the transformation is purely symbolic and doesn't
    actually need the values until solve time.

    This class is NOT monkey patching - it's a separate class used only
    within cvxpylayers. CVXPY's original Dgp2Dcp remains unchanged.
    """

    def accepts(self, problem: cp.Problem) -> bool:
        """Accept DGP problems even without parameter values.

        Args:
            problem: The CVXPY problem to check

        Returns:
            True if the problem is DGP, False otherwise
        """
        return problem.is_dgp()

    def apply(self, problem: cp.Problem) -> tuple[cp.Problem, Any]:
        """Apply DGP to DCP reduction using custom canon methods.

        Args:
            problem: The DGP problem to reduce

        Returns:
            Tuple of (DCP problem, inverse data)
        """
        if not self.accepts(problem):
            raise ValueError("The supplied problem is not DGP.")

        # Use our custom canon methods that handle None parameter values
        self.canon_methods = _DgpCanonMethodsNoValueCheck()
        equiv_problem, inverse_data = super(cp.reductions.Dgp2Dcp, self).apply(  # type: ignore[misc]
            problem
        )
        inverse_data._problem = problem
        return equiv_problem, inverse_data
