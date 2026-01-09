"""Unit tests for parse_args.py module."""

import cvxpy as cp
import numpy as np
import pytest
import torch

from cvxpylayers.utils.parse_args import LayersContext, VariableRecovery, parse_args


class TestVariableRecovery:
    """Test VariableRecovery.recover() method."""

    def test_recover_primal_unbatched(self):
        """Test primal recovery without batch dimension."""
        var_recovery = VariableRecovery(primal=slice(0, 3), dual=None, shape=(3,))
        primal_sol = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dual_sol = np.array([0.0, 0.0])

        result = var_recovery.recover(primal_sol, dual_sol)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_recover_primal_batched(self):
        """Test primal recovery with batch dimension."""
        var_recovery = VariableRecovery(primal=slice(1, 4), dual=None, shape=(3,))
        primal_sol = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
            ],
        )
        dual_sol = np.array([[0.0], [0.0]])

        result = var_recovery.recover(primal_sol, dual_sol)
        expected = np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]])
        np.testing.assert_array_equal(result, expected)

    def test_recover_dual_unbatched(self):
        """Test dual recovery without batch dimension."""
        var_recovery = VariableRecovery(primal=None, dual=slice(0, 2), shape=(2,))
        primal_sol = np.array([1.0, 2.0, 3.0])
        dual_sol = np.array([0.5, 1.5, 2.5])

        result = var_recovery.recover(primal_sol, dual_sol)
        np.testing.assert_array_equal(result, np.array([0.5, 1.5]))

    def test_recover_dual_batched(self):
        """Test dual recovery with batch dimension."""
        var_recovery = VariableRecovery(primal=None, dual=slice(1, 3), shape=(2,))
        primal_sol = np.array([[1.0], [2.0]])
        dual_sol = np.array(
            [
                [0.0, 0.5, 1.5, 2.5],
                [3.0, 3.5, 4.5, 5.5],
            ],
        )

        result = var_recovery.recover(primal_sol, dual_sol)
        # shape=(2,) with slice(1, 3) extracts 2 elements per batch
        expected = np.array([[0.5, 1.5], [3.5, 4.5]])
        np.testing.assert_array_equal(result, expected)

    def test_recover_neither_raises_error(self):
        """Test that RuntimeError is raised when both primal and dual are None."""
        var_recovery = VariableRecovery(primal=None, dual=None, shape=(1,))
        primal_sol = np.array([1.0, 2.0])
        dual_sol = np.array([0.5, 1.5])

        with pytest.raises(RuntimeError):
            var_recovery.recover(primal_sol, dual_sol)


class TestLayersContextValidateParams:
    """Test LayersContext.validate_params() method."""

    @pytest.fixture
    def simple_context(self):
        """Create a simple LayersContext for testing."""
        # Create a simple problem: minimize x subject to x >= 0
        x = cp.Variable(2)
        p = cp.Parameter(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        ctx = parse_args(problem, [x], [p], "DIFFCP")
        return ctx

    def test_validate_params_count_mismatch(self, simple_context):
        """Test that wrong number of parameters raises ValueError."""
        with pytest.raises(
            ValueError,
            match="A tensor must be provided for each CVXPY parameter",
        ):
            simple_context.validate_params(
                [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            )

    def test_validate_params_unbatched_correct_shape(self, simple_context):
        """Test validation with correct unbatched parameter."""
        batch_dims = simple_context.validate_params([torch.tensor([1.0, 2.0])])
        assert batch_dims == ()
        assert simple_context.batch_sizes == [0]

    def test_validate_params_unbatched_wrong_shape(self, simple_context):
        """Test that wrong unbatched shape raises ValueError."""
        with pytest.raises(ValueError, match="Invalid parameter shape"):
            simple_context.validate_params([torch.tensor([1.0, 2.0, 3.0])])

    def test_validate_params_batched_correct_shape(self, simple_context):
        """Test validation with correct batched parameter."""
        batched_param = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch_dims = simple_context.validate_params([batched_param])
        assert batch_dims == (3,)
        assert simple_context.batch_sizes == [3]

    def test_validate_params_batched_wrong_shape(self, simple_context):
        """Test that wrong batched shape raises ValueError."""
        batched_param = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="Invalid parameter shape"):
            simple_context.validate_params([batched_param])

    def test_validate_params_wrong_dimensionality(self, simple_context):
        """Test that wrong dimensionality raises ValueError."""
        # Parameter expects shape (2,) but we give shape (2, 2, 2)
        wrong_dim = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        with pytest.raises(ValueError, match="Invalid parameter dimensionality"):
            simple_context.validate_params([wrong_dim])

    def test_validate_params_mixed_batch_sizes_error(self):
        """Test that inconsistent batch sizes raise ValueError."""
        # Create problem with two parameters
        x = cp.Variable(2)
        p1 = cp.Parameter(2)
        p2 = cp.Parameter(3)
        # Problem must use both parameters
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p1) + cp.sum(p2)), [x >= 0])
        ctx = parse_args(problem, [x], [p1, p2], "DIFFCP")

        # Try to validate with different batch sizes
        param1_batched = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # batch size 2
        param2_batched = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )  # batch size 3

        with pytest.raises(ValueError, match="Inconsistent batch sizes"):
            ctx.validate_params([param1_batched, param2_batched])

    def test_validate_params_mixed_batched_unbatched(self):
        """Test that mixing batched and unbatched parameters works (broadcasting)."""
        # Create problem with two parameters
        x = cp.Variable(2)
        p1 = cp.Parameter(2)
        p2 = cp.Parameter(3)
        # Problem must use both parameters
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p1) + cp.sum(p2)), [x >= 0])
        ctx = parse_args(problem, [x], [p1, p2], "DIFFCP")

        # Mix batched and unbatched
        param1_batched = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # batch size 2
        param2_unbatched = torch.tensor([1.0, 2.0, 3.0])  # unbatched

        batch_dims = ctx.validate_params([param1_batched, param2_unbatched])
        assert batch_dims == (2,)
        assert ctx.batch_sizes == [2, 0]


class TestParseArgs:
    """Test parse_args() function."""

    def test_parse_args_simple_problem(self):
        """Test parse_args with a simple optimization problem."""
        x = cp.Variable(2)
        p = cp.Parameter(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        ctx = parse_args(problem, [x], [p], "DIFFCP")

        assert isinstance(ctx, LayersContext)
        assert len(ctx.parameters) == 1
        assert ctx.parameters[0] is p
        assert len(ctx.var_recover) == 1
        assert isinstance(ctx.var_recover[0], VariableRecovery)

    def test_parse_args_non_dpp_problem(self):
        """Test that non-DPP problems raise ValueError."""
        x = cp.Variable(2)
        p = cp.Parameter(2, nonneg=True)
        # Division by parameter violates DPP (non-affine use of parameter)
        problem = cp.Problem(cp.Minimize(cp.sum(x / p)), [x >= 0])

        with pytest.raises(ValueError, match="Problem must be DPP"):
            parse_args(problem, [x], [p], "DIFFCP")

    def test_parse_args_parameter_mismatch(self):
        """Test that parameter mismatch raises ValueError."""
        x = cp.Variable(2)
        p1 = cp.Parameter(2)
        p2 = cp.Parameter(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p1)), [x >= 0])

        # Try to use p2 which is not in the problem
        with pytest.raises(ValueError, match="must exactly match problem.parameters"):
            parse_args(problem, [x], [p2], "DIFFCP")

    def test_parse_args_variable_not_in_problem(self):
        """Test that variables not in problem raise ValueError."""
        x = cp.Variable(2)
        y = cp.Variable(2)
        p = cp.Parameter(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        # y is not in the problem
        with pytest.raises(ValueError, match="must be a subset of problem.variables"):
            parse_args(problem, [y], [p], "DIFFCP")

    def test_parse_args_parameters_not_list(self):
        """Test that parameters must be list or tuple."""
        x = cp.Variable(2)
        p = cp.Parameter(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        # Pass parameters as a set instead of list/tuple
        with pytest.raises(ValueError, match="must be provided as a list or tuple"):
            parse_args(problem, [x], {p}, "DIFFCP")  # type: ignore[arg-type]

    def test_parse_args_variables_not_list(self):
        """Test that variables must be list or tuple."""
        x = cp.Variable(2)
        p = cp.Parameter(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        # Pass variables as a set instead of list/tuple
        with pytest.raises(ValueError, match="must be provided as a list or tuple"):
            parse_args(problem, {x}, [p], "DIFFCP")  # type: ignore[arg-type]

    def test_parse_args_default_solver(self):
        """Test that solver defaults to DIFFCP when None."""
        x = cp.Variable(2)
        p = cp.Parameter(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        ctx = parse_args(problem, [x], [p], None)
        # Should not raise an error and should work with DIFFCP
        assert isinstance(ctx, LayersContext)

    def test_parse_args_multiple_variables(self):
        """Test parse_args with multiple variables."""
        x = cp.Variable(2)
        y = cp.Variable(3)
        p = cp.Parameter(2)
        q = cp.Parameter(3)
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(x - p) + cp.sum_squares(y - q)),
            [x >= 0, y >= 0],
        )

        ctx = parse_args(problem, [x, y], [p, q], "DIFFCP")

        assert len(ctx.var_recover) == 2
        # Check that slices are correct
        assert ctx.var_recover[0].primal is not None
        assert ctx.var_recover[1].primal is not None
        # x has size 2, y has size 3
        assert ctx.var_recover[0].primal.stop - ctx.var_recover[0].primal.start == 2
        assert ctx.var_recover[1].primal.stop - ctx.var_recover[1].primal.start == 3

    def test_parse_args_variable_recovery_slices(self):
        """Test that VariableRecovery slices are constructed correctly."""
        x = cp.Variable(5)
        p = cp.Parameter(5)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        ctx = parse_args(problem, [x], [p], "DIFFCP")

        # Should have one VariableRecovery
        assert len(ctx.var_recover) == 1
        var_rec = ctx.var_recover[0]
        assert var_rec.primal is not None
        assert var_rec.dual is None
        # Variable x has size 5
        assert var_rec.primal.stop - var_rec.primal.start == 5
