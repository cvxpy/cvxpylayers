"""Monkey-patches for parametric quad_form(x, P) DPP support (issue #136).

Stock CVXPY (<=1.8) rejects quad_form(x, P) where P is cp.Parameter(PSD=True)
because is_atom_convex() checks P.is_constant(), which is False under DPP.

We add a quad_form_dpp_scope: when active, the convexity check accepts
parametric P.  The scope is only activated by cvxpylayers during problem
canonicalization for QP-capable solvers, so normal CVXPY behaviour is
completely unaffected.

If CVXPY already provides quad_form_dpp_scope (future versions), all
patches are skipped.
"""

import contextlib

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.quad_form import QuadForm
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities import performance_utils as perf
from cvxpy.utilities import scopes
from cvxpy.utilities.coeff_extractor import CoeffExtractor

# Solvers that handle quadratic objectives directly (quad_obj=True).
# For these solvers, quad_form_dpp_scope is entered during canonicalization.
QP_SOLVERS = frozenset({"MOREAU", "SCS", "CLARABEL", "CUCLARABEL"})

# Check if CVXPY already has quad_form_dpp_scope support.
# The dev build (1.9.0.dev0) has it; stock 1.8.x does not.
_NEEDS_PATCH = not hasattr(scopes, "quad_form_dpp_scope_active")

if _NEEDS_PATCH:
    # -- Patch 0: Add quad_form_dpp_scope to scopes module ---------------------
    scopes._quad_form_dpp_scope_active = False

    @contextlib.contextmanager
    def _quad_form_dpp_scope():
        prev = scopes._quad_form_dpp_scope_active
        scopes._quad_form_dpp_scope_active = True
        yield
        scopes._quad_form_dpp_scope_active = prev

    def _quad_form_dpp_scope_active():
        return scopes._quad_form_dpp_scope_active

    scopes.quad_form_dpp_scope = _quad_form_dpp_scope
    scopes.quad_form_dpp_scope_active = _quad_form_dpp_scope_active

    # -- Patch _cache_key to include quad_form_dpp_scope in cache keys ---------
    # compute_once resolves _cache_key through module globals at call time,
    # so replacing it here affects all existing @compute_once decorators
    # (is_convex, is_concave, is_dcp, etc.).
    _orig_cache_key = perf._cache_key

    def _patched_cache_key(args, kwargs):
        key = _orig_cache_key(args, kwargs)
        if scopes._quad_form_dpp_scope_active:
            key = ("__quad_form_dpp_scope_active__",) + key
        return key

    perf._cache_key = _patched_cache_key

    # -- Patch 1: Scope-aware QuadForm.is_atom_convex/concave ------------------
    # Outside the scope, behaviour is identical to stock CVXPY.
    _orig_is_atom_convex = QuadForm.is_atom_convex
    _orig_is_atom_concave = QuadForm.is_atom_concave

    def _scoped_is_atom_convex(self):
        if scopes._quad_form_dpp_scope_active:
            P = self.args[1]
            x = self.args[0]
            return (
                isinstance(P, Parameter)
                and len(x.parameters()) == 0
                and P.is_psd()
            )
        return _orig_is_atom_convex(self)

    def _scoped_is_atom_concave(self):
        if scopes._quad_form_dpp_scope_active:
            P = self.args[1]
            x = self.args[0]
            return (
                isinstance(P, Parameter)
                and len(x.parameters()) == 0
                and P.is_nsd()
            )
        return _orig_is_atom_concave(self)

    QuadForm.is_atom_convex = _scoped_is_atom_convex
    QuadForm.is_atom_concave = _scoped_is_atom_concave

    # -- Patch 2: extract_quadratic_coeffs with parametric P -------------------
    # When P is a bare Parameter, build a TensorRepresentation mapping P's
    # flattened parameter columns to the QP's P matrix entries.

    _orig_extract_quadratic_coeffs = CoeffExtractor.extract_quadratic_coeffs

    def _patched_extract_quadratic_coeffs(self, affine_expr, quad_forms):
        """extract_quadratic_coeffs with parametric-P support."""
        for var in affine_expr.variables():
            if var.id in quad_forms:
                if isinstance(quad_forms[var.id][2].args[1], Parameter):
                    return _extract_with_param_P(self, affine_expr, quad_forms)
        return _orig_extract_quadratic_coeffs(self, affine_expr, quad_forms)

    def _extract_with_param_P(self, affine_expr, quad_forms):
        """Full extract_quadratic_coeffs with parametric P path."""
        assert affine_expr.is_dpp()
        affine_id_map, affine_offsets, x_length, affine_var_shapes = (
            InverseData.get_var_offsets(affine_expr.variables())
        )
        param_coeffs = canonInterface.get_problem_matrix(
            [affine_expr.canonical_form[0]],
            x_length,
            affine_offsets,
            self.param_to_size,
            self.param_id_map,
            affine_expr.size,
            self.canon_backend,
        )
        constant = param_coeffs[[-1], :]
        c = param_coeffs[:-1, :].toarray()
        num_params = param_coeffs.shape[1]

        coeffs = {}
        for var in affine_expr.variables():
            if var.id in quad_forms:
                var_id = var.id
                orig_id = quad_forms[var_id][2].args[0].id
                var_offset = affine_id_map[var_id][0]
                var_size = affine_id_map[var_id][1]
                c_part = c[var_offset : var_offset + var_size, :]

                P_expr = quad_forms[var_id][2].args[1]

                if isinstance(P_expr, Parameter):
                    # -- Parametric P path --
                    assert var_size == 1, (
                        "quad_form with parametric P only supports scalar output"
                    )
                    n = P_expr.shape[0]
                    P_param_offset = self.param_id_map[P_expr.id]
                    rows = np.tile(np.arange(n), n)
                    cols = np.repeat(np.arange(n), n)
                    param_offsets = P_param_offset + np.arange(n * n)

                    nonzero_idxs = c_part[0] != 0
                    if not np.any(nonzero_idxs):
                        P_tup = TensorRepresentation.empty_with_shape((n, n))
                    else:
                        c_nz_vals = c_part[0, nonzero_idxs]
                        c_nz_idxs = np.arange(num_params)[nonzero_idxs]
                        if np.any(c_nz_idxs != (num_params - 1)):
                            raise ValueError(
                                "quad_form with parametric P requires x to be "
                                "parameter-free"
                            )
                        P_tup = TensorRepresentation(
                            np.full(n * n, c_nz_vals[0]),
                            rows,
                            cols,
                            param_offsets,
                            (n, n),
                        )
                else:
                    # -- Constant P path (identical to stock) --
                    P_val = P_expr.value
                    assert P_val is not None, (
                        "P must be instantiated before extract_quadratic_coeffs"
                    )
                    P = (
                        P_val.tocoo()
                        if sp.issparse(P_val) and not isinstance(P_val, sp.coo_matrix)
                        else sp.coo_matrix(P_val)
                    )

                    block_indices = quad_forms[var_id][2].block_indices
                    if var_size == 1:
                        nonzero_idxs = c_part[0] != 0
                        data = P.data[:, None] * c_part[:, nonzero_idxs]
                        param_idxs = np.arange(num_params)[nonzero_idxs]
                        P_tup = TensorRepresentation(
                            data.flatten(order="F"),
                            np.tile(P.row, len(param_idxs)),
                            np.tile(P.col, len(param_idxs)),
                            np.repeat(param_idxs, len(P.data)),
                            P.shape,
                        )
                    elif block_indices is not None:
                        P_tup = self._extract_block_quad(
                            P, c_part, block_indices, num_params
                        )
                    else:
                        assert (P.col == P.row).all()
                        scaled = P @ c_part
                        ri, ci = np.nonzero(scaled)
                        P_tup = TensorRepresentation(
                            c_part[ri, ci], ri, ri.copy(), ci, P.shape
                        )

                if orig_id in coeffs:
                    if "P" in coeffs[orig_id]:
                        coeffs[orig_id]["P"] = coeffs[orig_id]["P"] + P_tup
                    else:
                        coeffs[orig_id]["P"] = P_tup
                else:
                    coeffs[orig_id] = {}
                    coeffs[orig_id]["P"] = P_tup
                    q_shape = (P_tup.shape[0], c.shape[1])
                    if num_params == 1:
                        coeffs[orig_id]["q"] = np.zeros(q_shape)
                    else:
                        coeffs[orig_id]["q"] = sp.coo_matrix(
                            ([], ([], [])), shape=q_shape
                        )
            else:
                var_offset = affine_id_map[var.id][0]
                var_size = np.prod(affine_var_shapes[var.id], dtype=int)
                if var.id in coeffs:
                    if num_params == 1:
                        coeffs[var.id]["q"] += c[var_offset : var_offset + var_size, :]
                    else:
                        coeffs[var.id]["q"] += param_coeffs[
                            var_offset : var_offset + var_size, :
                        ]
                else:
                    coeffs[var.id] = {}
                    if num_params == 1:
                        coeffs[var.id]["q"] = c[var_offset : var_offset + var_size, :]
                    else:
                        coeffs[var.id]["q"] = param_coeffs[
                            var_offset : var_offset + var_size, :
                        ]
        return coeffs, constant

    CoeffExtractor.extract_quadratic_coeffs = _patched_extract_quadratic_coeffs
