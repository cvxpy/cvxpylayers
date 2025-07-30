import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict
import diffcp
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("Unable to import tensorflow. Please install "
                      "TensorFlow >= 2.0 (https://tensorflow.org).")

tf_major_version = int(tf.__version__.split('.')[0])
if tf_major_version < 2:
    raise ImportError("cvxpylayers requires TensorFlow >= 2.0; please "
                      "upgrade your installation of TensorFlow, which is "
                      "version %s." % tf.__version__)


class CvxpyLayer(object):
    """A differentiable convex optimization layer

    A CvxpyLayer solves a parametrized convex optimization problem given by a
    CVXPY problem. It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass. The CVPXY problem must be a disciplined parametrized
    program.

    Example usage

        ```
        import cvxpy as cp
        import tensorflow as tf
        from cvxpylayers.tensorflow import CvxpyLayer


        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        A_tf = tf.Variable(tf.random.normal((m, n)))
        b_tf = tf.Variable(tf.random.normal((m,)))

        with tf.GradientTape() as tape:
          # solve the problem, setting the values of A and b to A_tf and b_tf
          solution, = cvxpylayer(A_tf, b_tf)
          summed_solution = tf.math.reduce_sum(solution)
        gradA, gradb = tape.gradient(summed_solution, [A_tf, b_tf])
        ```
    """

    def __init__(self, problem, parameters, variables, gp=False):
        """Construct a CvxpyLayer

        Args:
          problem: The CVXPY problem; must be DPP.
          parameters: A list of CVXPY Parameters in the problem; the order
                      of the Parameters determines the order in which parameter
                      values must be supplied in the forward pass.
          variables: A list of CVXPY Variables in the problem; the order of the
                     Variables determines the order of the optimal variable
                     values returned from the forward pass.
          gp: Whether to parse the problem using DGP (True or False).
        """
        if gp:
            if not problem.is_dgp(dpp=True):
                raise ValueError('Problem must be DPP.')
        else:
            if not problem.is_dcp(dpp=True):
                raise ValueError('Problem must be DPP.')
        if set(parameters) != set(problem.parameters()):
            raise ValueError("The layer's parameters must exactly match "
                             "problem.parameters")
        if not set(variables).issubset(set(problem.variables())):
            raise ValueError('Argument `variables` must be a subset of '
                             '`problem.variables()`')
        self.params = parameters
        self.gp = gp

        if self.gp:
            for param in parameters:
                if param.value is None:
                    raise ValueError("An initial value for each parameter is "
                                     "required when gp=True.")
            data, solving_chain, _ = (
                problem.get_problem_data(
                    solver=cp.SCS, gp=True,
                    solver_opts={'use_quad_obj': False}
            ))
            self.asa_maps = data[cp.settings.PARAM_PROB]
            self.dgp2dcp = solving_chain.get(cp.reductions.Dgp2Dcp)
            self.param_ids = [p.id for p in self.asa_maps.parameters]
        else:
            data, _, _ = problem.get_problem_data(
                solver=cp.SCS, solver_opts={'use_quad_obj': False})
            self.asa_maps = data[cp.settings.PARAM_PROB]
            self.param_ids = [p.id for p in self.params]

        self.cones = dims_to_solver_dict(data['dims'])
        self.vars = variables

    def __call__(self, *parameters, solver_args={}):
        """Solve problem (or a batch of problems) corresponding to `parameters`

        Args:
          parameters: a sequence of tf.Tensors; the n-th Tensor specifies
                      the value for the n-th CVXPY Parameter. These Tensors
                      can be batched: if a Tensor has 3 dimensions, then its
                      first dimension is interpreted as the batch size.
          solver_args: a dict of optional arguments, to send to `diffcp`. Keys
                       should be the names of keyword arguments.

        Returns:
          a list of optimal variable values, one for each CVXPY Variable
          supplied to the constructor.
        """
        if len(parameters) != len(self.params):
            raise ValueError('A tensor must be provided for each CVXPY '
                             'parameter; received %d tensors, expected %d' % (
                                 len(parameters), len(self.params)))
        compute = tf.custom_gradient(
            lambda *parameters: self._compute(parameters, solver_args))
        return compute(*parameters)

    def _dx_from_dsoln(self, dsoln):
        dvars_numpy = list(map(lambda x: x.numpy(), dsoln))
        del_vars = {}
        for v, dv in zip(self.vars, dvars_numpy):
            del_vars[v.id] = dv
        return self.asa_maps.split_adjoint(del_vars)

    def _problem_data_from_params(self, params):
        c, _, A, b = self.asa_maps.apply_parameters(
            dict(zip(self.param_ids, params)), keep_zeros=True)
        A = -A
        return A, b, c

    def _restrict_DT_to_dx(self, DT, batch_size, s_shape):
        if batch_size > 0:
            zeros = [np.zeros(s_shape) for _ in range(batch_size)]
        else:
            zeros = np.zeros(s_shape)
        return lambda dxs: DT(dxs, zeros, zeros)

    def _split_solution(self, x):
        soln = self.asa_maps.split_solution(x, {v.id: v for v in self.vars})
        return tuple([tf.constant(soln[v.id]) for v in self.vars])

    def _compute(self, params, solver_args={}):
        tf_params = params
        params = [p.numpy() for p in params]

        # infer whether params are batched
        batch_sizes = []
        for i, (p_in, p_signature) in enumerate(zip(params, self.params)):
            # check and extract the batch size for the parameter
            # 0 means there is no batch dimension for this parameter
            # and we assume the batch dimension is non-zero
            if p_in.ndim == p_signature.ndim:
                batch_size = 0
            elif p_in.ndim == p_signature.ndim + 1:
                batch_size = p_in.shape[0]
                if batch_size == 0:
                    raise ValueError(
                        "Invalid parameter size passed in. "
                        "Parameter {} appears to be batched, but the leading "
                        "dimension is 0".format(i))
            else:
                raise ValueError(
                    "Invalid parameter size passed in. Expected "
                    "parameter {} to have have {} or {} dimensions "
                    "but got {} dimensions".format(
                        i, p_signature.ndim, p_signature.ndim + 1,
                        p_in.ndim))
            batch_sizes.append(batch_size)

            # validate the parameter shape
            p_shape = p_in.shape if batch_size == 0 else p_in.shape[1:]
            if not np.all(p_shape == p_signature.shape):
                raise ValueError(
                    "Inconsistent parameter shapes passed in. "
                    "Expected parameter {} to have non-batched shape of "
                    "{} but got {}.".format(
                            i,
                            p_signature.shape,
                            p_signature.shape))

        batch_sizes = np.array(batch_sizes)
        any_batched = np.any(batch_sizes > 0)

        if any_batched:
            nonzero_batch_sizes = batch_sizes[batch_sizes > 0]
            batch_size = int(nonzero_batch_sizes[0])
            if np.any(nonzero_batch_sizes != batch_size):
                raise ValueError(
                    "Inconsistent batch sizes passed in. Expected "
                    "parameters to have no batch size or all the same "
                    "batch size but got sizes: {}.".format(batch_sizes))
        else:
            batch_size = 1

        if self.gp:
            old_params_to_new_params = self.dgp2dcp.canon_methods._parameters
            param_map = {}
            # construct a list of params for the DCP problem
            for param, value in zip(self.params, params):
                if param in old_params_to_new_params:
                    new_id = old_params_to_new_params[param].id
                    param_map[new_id] = np.log(value)
                else:
                    new_id = param.id
                    param_map[new_id] = value
            params = [param_map[pid] for pid in self.param_ids]

        As, bs, cs = [], [], []
        for i in range(batch_size):
            params_i = [
                p if sz == 0 else p[i] for p, sz in zip(params, batch_sizes)]
            A, b, c = self._problem_data_from_params(params_i)
            As.append(A)
            bs.append(b)
            cs.append(c)

        try:
            xs, _, ss, _, DT = diffcp.solve_and_derivative_batch(
                As=As, bs=bs, cs=cs, cone_dicts=[self.cones] * batch_size,
                **solver_args)
        except diffcp.SolverError as e:
            print(
                "Please consider re-formulating your problem so that "
                "it is always solvable or increasing the number of "
                "solver iterations.")
            raise e

        DT = self._restrict_DT_to_dx(DT, batch_size, ss[0].shape)
        solns = [self._split_solution(x) for x in xs]
        # soln[i] is a tensor with first dimension equal to batch_size, holding
        # the optimal values for variable i
        solution = [
            tf.stack([s[i] for s in solns]) for i in range(len(self.vars))]
        if not any_batched:
            solution = [tf.squeeze(s, 0) for s in solution]

        if self.gp:
            solution = [tf.exp(s) for s in solution]

        def gradient_function(*dsoln):
            if self.gp:
                dsoln = [dsoln*s for dsoln, s in zip(dsoln, solution)]

            if not any_batched:
                dsoln = [tf.expand_dims(dvar, 0) for dvar in dsoln]

            # split the batched dsoln tensors into lists, with one list
            # corresponding to each problem in the batch
            dsoln_lists = [[] for _ in range(batch_size)]
            for value in dsoln:
                tensors = tf.split(value, batch_size)
                for dsoln_list, t in zip(dsoln_lists, tensors):
                    dsoln_list.append(tf.squeeze(t))
            dxs = [self._dx_from_dsoln(dsoln_list)
                   for dsoln_list in dsoln_lists]
            dAs, dbs, dcs = DT(dxs)
            dparams_dict_unbatched = [
                self.asa_maps.apply_param_jac(dc, -dA, db) for
                (dA, db, dc) in zip(dAs, dbs, dcs)]
            dparams = []
            for pid in self.param_ids:
                dparams.append(
                    tf.constant([d[pid] for d in dparams_dict_unbatched]))

            if not any_batched:
                dparams = tuple(tf.squeeze(dparam, 0) for dparam in dparams)
            else:
                for i, sz in enumerate(batch_sizes):
                    if sz == 0:
                        dparams[i] = tf.reduce_sum(dparams[i], axis=0)

            if self.gp:
                # differentiate through the log transformation of params
                dcp_dparams = dparams
                dparams = []
                grads = {pid: g for pid, g in zip(self.param_ids, dcp_dparams)}
                old_params_to_new_params = (
                    self.dgp2dcp.canon_methods._parameters
                )
                for param, value in zip(self.params, tf_params):
                    g = 0.0 if param.id not in grads else grads[param.id]
                    if param in old_params_to_new_params:
                        dcp_param_id = old_params_to_new_params[param].id
                        # new_param.value == log(param), apply chain rule
                        g = g + (1.0 / value) * grads[dcp_param_id]
                    dparams.append(g)
            return tuple(dparams)

        return solution, gradient_function
