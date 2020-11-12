import casadi as cs
import numpy as np
from typing import List, Dict
from scipy.special import comb

class IterativeLQR:
    """
    The IterativeLQR class solves a nonlinear, unconstrained iLQR problem for a given
     - system dynamics (continuous time)
     - intermediate cost l(x,u)
     - final cost lf(x)
    """

    class LinearDynamics:

        def __init__(self, nx: int, nu: int):

            self.A = np.zeros((nx, nx))
            self.B = np.zeros((nx, nu))
            self.Fxx = np.zeros((nx*nx, nx))
            self.Fuu = np.zeros((nx*nu, nu))
            self.Fux = np.zeros((nx*nu, nx))

        def __repr__(self):
            return self.__dict__.__repr__()

    class LinearConstraint:

        def __init__(self, nx: int, nu: int, nc: int):

            self.C = np.zeros((nc, nx))
            self.D = np.zeros((nc, nu))
            self.g = np.zeros(nc)

        def __repr__(self):
            return self.__dict__.__repr__()

    class QuadraticCost:

        def __init__(self, nx: int, nu: int):
            self.qx = np.zeros(nx)
            self.Qxx = np.zeros((nx, nx))
            self.qu = np.zeros(nu)
            self.Quu = np.zeros((nu, nu))
            self.Qxu = np.zeros((nx, nu))

        def __repr__(self):
            return self.__dict__.__repr__()

    def __init__(self,
                 x: cs.SX,
                 u: cs.SX,
                 xdot: cs.SX,
                 dt: float,
                 N: int,
                 intermediate_cost: cs.SX,
                 final_cost: cs.SX,
                 intermediate_constraints=dict(),  # note: does not work currently
                 final_constraint=None,
                 sym_t=cs.SX):

        """
        Constructor
        :param x: state variable
        :param u: control variable
        :param xdot: continuous-time dynamics -> xdot = f(x, u)
        :param dt: discretization step
        :param N: horizon length
        :param intermediate_cost: intermediate cost -> l(x, u)
        :param final_cost: final cost -> lf(x)
        :param intermediate_constraints: dict constr_name -> hi(x, u) = 0
        :param final_constraint: hf(x) = 0
        :param sym_t: casadi symbol type (SX or MX)
        """

        # sym type
        self._sym_t = sym_t

        # state and control dimension
        self._nx = x.size1()
        self._nu = u.size1()

        # discretization & horizon
        self._dt = dt
        self._N = N

        # dynamics
        self._dynamics_ct = cs.Function('dynamics_ct',
                                        {'x': x, 'u': u, 'xdot': xdot},
                                        ['x', 'u'],
                                        ['xdot'])

        # cost terms
        self._diff_inter_cost = cs.Function('intermediate_cost',
                                            {'x': x, 'u': u, 'l': intermediate_cost},
                                            ['x', 'u'],
                                            ['l'])

        self._final_cost = cs.Function('final_cost',
                                       {'x': x, 'u': u, 'lf': final_cost},
                                       ['x', 'u'],
                                       ['lf'])

        self._jacobian_lf = self._final_cost.jac()
        self._hessian_lf = self._jacobian_lf.jac()

        # discrete dynamics & intermediate cost
        self._discretize()

        # constraints
        self._constrained = final_constraint is not None or len(intermediate_constraints) is not 0
        self._final_constraint = None
        self._constraint_to_go = None

        if self._constrained:

            self._constraint_to_go = self.LinearConstraint(self._nx, self._nu, 0)
            self._inter_constraints = [self.LinearConstraint(self._nx, self._nu, 0) for _ in range(self._N)]

        # final constraint
        if final_constraint is not None:

            self._final_constraint = cs.Function('final_constraint',
                                                 {'x': x, 'u': u, 'hf': final_constraint},
                                                 ['x', 'u'],
                                                 ['hf'])

            self._final_constraint_jac = self._final_constraint.jac()

        # intermediate constraints
        intermediate_constr_r_der = []  # list of intermediate constraint r-th derivatives
        intermediate_constr_r_der_des = []  # list of intermediate constraint desired r-th derivative value

        # loop over all defined constraints, fill above lists
        for name, ic in intermediate_constraints.items():

            rel_degree = 0
            hi_derivatives = [ic]  # list of current constraint derivatives

            while True:

                inter_constraint = cs.Function('intermediate_constraint',
                                               {'x': x, 'u': u, 'h': ic},
                                               ['x', 'u'],
                                               ['h'])

                inter_constraint_jac = inter_constraint.jac()

                # if constraint jacobian depends on u, break
                if inter_constraint_jac(x=x, u=u)['DhDu'].nnz() > 0:
                    break

                # otherwise, increase relative degree and do time derivative
                rel_degree += 1
                ic = inter_constraint_jac(x=x, u=u)['DhDx'] @ xdot
                hi_derivatives.append(ic)

            print('constraint "{}" relative degree is {}'.format(name, rel_degree))
            rand_x = np.random.standard_normal(self._nx)
            rand_u = np.random.standard_normal(self._nu)
            r = np.linalg.matrix_rank(inter_constraint_jac(x=rand_x, u=rand_u)['DhDu'].toarray())
            print('constraint "{}" rank at random (x, u) is {} vs dim(u) = {}'.format(name, r, self._nu))

            intermediate_constr_r_der.append(ic)
            hr_des = sym_t.zeros(ic.size1())
            constr_lambda = 0.1
            for i in range(rel_degree):
                hr_des += comb(r, i) * hi_derivatives[i] * constr_lambda**(r-i)

            intermediate_constr_r_der_des.append(hr_des)

        self._inter_constr = cs.Function('intermediate_constraint',
                                         {'x': x, 'u': u, 'h': cs.vertcat(*intermediate_constr_r_der)},
                                         ['x', 'u'],
                                         ['h'])

        self._inter_constr_des = cs.Function('intermediate_constraint_desired',
                                             {'x': x, 'u': u, 'hdes': cs.vertcat(*intermediate_constr_r_der_des)},
                                             ['x', 'u'],
                                             ['hdes'])

        self._inter_constr_jac = self._inter_constr.jac()

        self._has_inter_constr = self._inter_constr.size1_out('h') > 0

        # initalization of all internal structures
        self._state_trj = [np.zeros(self._nx) for _ in range(self._N + 1)]
        self._ctrl_trj  = [np.zeros(self._nu) for _ in range(self._N)]
        self._inter_constr_trj = [np.zeros(0) for _ in range(self._N)]
        self._lin_dynamics = [self.LinearDynamics(self._nx, self._nu) for _ in range(self._N)]
        self._inter_quad_cost = [self.QuadraticCost(self._nx, self._nu) for _ in range(self._N)]
        self._final_quad_cost = self.QuadraticCost(self._nx, 0)
        self._cost_to_go = [self.QuadraticCost(self._nx, self._nu) for _ in range(self._N)]
        self._value_function = [self.QuadraticCost(self._nx, self._nu) for _ in range(self._N)]

        self._fb_gain = [np.zeros((self._nu, self._nx)) for _ in range(self._N)]
        self._ff_u = [np.zeros(self._nu) for _ in range(self._N)]
        self._defect = [np.zeros(self._nx) for _ in range(self._N)]

        self._defect_norm = []
        self._du_norm = []
        self._dx_norm = []
        self._dcost = []

        self._use_second_order_dynamics = False
        self._use_single_shooting_state_update = False
        self._verbose = False

    @staticmethod
    def _make_jit_function(f: cs.Function):
        """
        Compiles casadi function into a shared object and return it
        :return:
        """

        import filecmp
        import os

        gen_code_path = 'ilqr_generated_{}.c'.format(f.name())
        f.generate(gen_code_path)

        gen_lib_path = 'ilqr_generated_{}.so'.format(f.name())
        gcc_cmd = 'gcc {} -shared -fPIC -O3 -o {}'.format(gen_code_path, gen_lib_path)

        if os.system(gcc_cmd) != 0:
            raise SystemError('Unable to compile function "{}"'.format(f.name()))

        jit_f = cs.external(f.name(), './' + gen_lib_path)

        os.remove(gen_code_path)
        os.remove(gen_lib_path)

        return jit_f

    def _discretize(self):
        """
        Compute discretized dynamics in the form of _F (nonlinear state transition function) and
        _jacobian_F (its jacobian)
        :return: None
        """

        x = self._sym_t.sym('x', self._nx)
        u = self._sym_t.sym('u', self._nu)

        dae = {'x': x,
               'p': u,
               'ode': self._dynamics_ct(x, u),
               'quad': self._diff_inter_cost(x, u)}

        # self._F = cs.integrator('F', 'rk', dae, {'t0': 0, 'tf': self._dt})
        self._F = cs.Function('F',
                              {'x0': x, 'p': u,
                               'xf': x + self._dt * self._dynamics_ct(x, u),
                               'qf': self._dt * self._diff_inter_cost(x, u)
                               },
                              ['x0', 'p'],
                              ['xf', 'qf'])

        # self._F = integrator.RK4(dae, {'tf': self._dt}, 'SX')
        self._jacobian_F = self._F.jac()
        self._hessian_F = self._jacobian_F.jac()

    def _linearize_quadratize(self):
        """
        Compute quadratic approximations to cost functions about the current state and control trajectories
        :return: None
        """

        jl_value = self._jacobian_lf(x=self._state_trj[-1])
        hl_value = self._hessian_lf(x=self._state_trj[-1])

        self._final_quad_cost.qx = jl_value['DlfDx'].toarray().flatten()
        self._final_quad_cost.Qxx = hl_value['DDlfDxDx'].toarray()

        for i in range(self._N):

            jode_value = self._jacobian_F(x0=self._state_trj[i],
                                          p=self._ctrl_trj[i])

            hode_value = self._hessian_F(x0=self._state_trj[i],
                                         p=self._ctrl_trj[i])

            self._inter_quad_cost[i].qu = jode_value['DqfDp'].toarray().flatten()
            self._inter_quad_cost[i].qx = jode_value['DqfDx0'].toarray().flatten()
            self._inter_quad_cost[i].Quu = hode_value['DDqfDpDp'].toarray()
            self._inter_quad_cost[i].Qxx = hode_value['DDqfDx0Dx0'].toarray()
            self._inter_quad_cost[i].Qxu = hode_value['DDqfDx0Dp'].toarray()

            self._lin_dynamics[i].A = jode_value['DxfDx0'].toarray()
            self._lin_dynamics[i].B = jode_value['DxfDp'].toarray()

            if self._use_second_order_dynamics:
                for j in range(self._nx):
                    nx = self._nx
                    nu = self._nu
                    self._lin_dynamics[i].Fxx[j*nx:(j+1)*nx, :] = hode_value['DDxfDx0Dx0'].toarray()[j::nx, :]
                    self._lin_dynamics[i].Fuu[j*nu:(j+1)*nu, :] = hode_value['DDxfDpDp'].toarray()[j::nx, :]
                    self._lin_dynamics[i].Fux[j*nu:(j+1)*nu, :] = hode_value['DDxfDpDx0'].toarray()[j::nx, :]

            if self._constrained:

                jconstr_value = self._inter_constr_jac(x=self._state_trj[i],
                                                       u=self._ctrl_trj[i])

                self._inter_constraints[i].C = jconstr_value['DhDx'].toarray()
                self._inter_constraints[i].D = jconstr_value['DhDu'].toarray()
                self._inter_constraints[i].g = self._inter_constr_des(x=self._state_trj[i],
                                                                      u=self._ctrl_trj[i])['hdes'].toarray().flatten()
                self._inter_constraints[i].g -= self._inter_constr(x=self._state_trj[i],
                                                                   u=self._ctrl_trj[i])['h'].toarray().flatten()

        if self._final_constraint is not None:

            jgf_value = self._final_constraint_jac(x=self._state_trj[-1])
            nc = self._final_constraint.size1_out('hf')
            self._constraint_to_go = self.LinearConstraint(self._nx, self._nu, nc)
            self._constraint_to_go.C = jgf_value['DhfDx'].toarray()
            self._constraint_to_go.D = np.zeros((nc, self._nu))
            self._constraint_to_go.g = self._final_constraint(x=self._state_trj[-1])['hf'].toarray().flatten()

    def _backward_pass(self):
        """
        To be implemented
        :return:
        """

        # value function at next time step (prev iteration)
        S = self._final_quad_cost.Qxx
        s = self._final_quad_cost.qx

        for i in reversed(range(self._N)):

            # variable labeling for better convenience
            nx = self._nx
            nu = self._nu
            x_integrated = self._F(x0=self._state_trj[i], p=self._ctrl_trj[i])['xf'].toarray().flatten()
            xnext = self._state_trj[i+1]
            d = x_integrated - xnext
            r = self._inter_quad_cost[i].qu
            q = self._inter_quad_cost[i].qx
            P = self._inter_quad_cost[i].Qxu.T
            R = self._inter_quad_cost[i].Quu
            Q = self._inter_quad_cost[i].Qxx
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B
            Fxx = self._lin_dynamics[i].Fxx.reshape((nx, nx, nx))
            Fuu = self._lin_dynamics[i].Fuu.reshape((nx, nu, nu))
            Fux = self._lin_dynamics[i].Fux.reshape((nx, nu, nx))

            # intermediate constraint
            # NOTE: constraint handling is experimental!
            # NOTE: intermediate constraints don't seem to work well

            # final constraint
            l_ff = np.zeros(self._nu)
            L_fb = np.zeros((self._nu, self._nx))
            Vns = np.eye(self._nu)

            constr_to_process = self._constraint_to_go is not None or self._has_inter_constr

            if self._constrained and constr_to_process:

                if self._constraint_to_go is not None:

                    # back-propagate constraint to go from next time step
                    C = self._constraint_to_go.C@A
                    D = self._constraint_to_go.C@B
                    g = self._constraint_to_go.g - self._constraint_to_go.C@d

                    # add intermediate constraint
                    C = np.vstack((C, self._inter_constraints[i].C))
                    D = np.vstack((D, self._inter_constraints[i].D))
                    g = np.hstack((g, self._inter_constraints[i].g))

                else:

                    # we only have intermediate constraints
                    C = self._inter_constraints[i].C
                    D = self._inter_constraints[i].D
                    g = self._inter_constraints[i].g




                # svd of constraint input matrix
                U, sv, V = np.linalg.svd(D)
                V = V.T

                # rotated constraint
                rot_g = U.T @ g
                rot_C = U.T @ C

                # non-zero singular values
                large_sv = sv > 1e-4

                nc = g.size  # number of currently active constraints
                nsv = len(sv)  # number of singular values
                rank = np.count_nonzero(large_sv)  # constraint input matrix rank

                # singular value inversion
                inv_sv = sv.copy()
                inv_sv[large_sv] = np.reciprocal(sv[large_sv])

                # compute constraint component of control input uc = Lc*x + lc
                l_ff = -V[:, 0:nsv] @ (inv_sv*rot_g[0:nsv])
                l_ff.flatten()
                L_fb = -V[:, 0:nsv] @ np.diag(inv_sv) @ rot_C[0:nsv, :]

                # update constraint to go
                left_constraint_dim = nc - rank

                if left_constraint_dim == 0:
                    self._constraint_to_go = None
                else:
                    self._constraint_to_go.C = rot_C[rank:, :]
                    self._constraint_to_go.D = np.zeros((left_constraint_dim, self._nu))
                    self._constraint_to_go.g = rot_g[rank:]

                nullspace_dim = self._nu - rank

                if nullspace_dim == 0:
                    Vns = np.zeros((self._nu, 0))
                else:
                    Vns = V[:, -nullspace_dim:]

                # the constraint induces a modified dynamics via u = Lx + l + Vns*z (z = new control input)
                d = d + B@l_ff
                A = A + B@L_fb
                B = B@Vns

                if self._use_second_order_dynamics:

                    tr_idx = (0, 2, 1)

                    d += 0.5*l_ff@Fuu@l_ff
                    A += l_ff@Fuu@L_fb + l_ff@Fux
                    B += l_ff@Fuu@Vns

                    Fxx = Fxx + L_fb.T @ (Fuu @ L_fb + Fux) + Fux.transpose(tr_idx)@L_fb
                    Fux = Vns.T @ (Fux + Fuu @ L_fb)
                    Fuu = Vns.T @ Fuu @ Vns

                q = q + L_fb.T@(r + R@l_ff) + P.T@l_ff
                Q = Q + L_fb.T@R@L_fb + L_fb.T@P + P.T@L_fb
                P = Vns.T @ (P + R@L_fb)

                r = Vns.T @ (r + R @ l_ff)
                R = Vns.T @ R @ Vns

            # intermediate quantities
            hx = q + A.T@(s + S@d)
            hu = r + B.T@(s + S@d)
            Huu = R + B.T@S@B
            Hux = P + B.T@S@A
            Hxx = Q + A.T@S@A

            if self._use_second_order_dynamics:

                Huu += (Fuu.T @ (s + S@d)).T
                Hux += (Fux.T @ (s + S@d)).T
                Hxx += (Fxx.T @ (s + S@d)).T

            # nullspace gain and feedforward computation
            l_Lz = -np.linalg.solve(Huu, np.hstack((hu.reshape((hu.size, 1)), Hux)))
            lz = l_Lz[:, 0]
            Lz = l_Lz[:, 1:]

            # overall gain and ffwd including constraint
            l_ff = l_ff + Vns @ lz
            L_fb = L_fb + Vns @ Lz

            # value function update
            s = hx - Lz.T@Huu@lz
            S = Hxx - Lz.T@Huu@Lz

            # save gain and ffwd
            self._fb_gain[i] = L_fb.copy()
            self._ff_u[i] = l_ff.copy()

            # save defect (for original dynamics)
            d = x_integrated - xnext
            self._defect[i] = d.copy()

    class PropagateResult:
        def __init__(self):
            self.state_trj = []
            self.ctrl_trj = []
            self.dx_norm = 0.0
            self.du_norm = 0.0
            self.cost = 0.0
            self.inter_constr = []
            self.final_constr = None

    def _forward_pass(self):
        """
        To be implemented
        :return:
        """
        x_old = self._state_trj.copy()

        defect_norm = 0
        du_norm = 0
        dx_norm = 0

        for i in range(self._N):

            xnext = self._state_trj[i+1]
            xi_upd = self._state_trj[i]
            ui = self._ctrl_trj[i]
            d = self._defect[i]
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B
            L = self._fb_gain[i]
            l = self._ff_u[i]
            dx = np.atleast_1d(xi_upd - x_old[i])

            ui_upd = ui + l + L@dx

            if self._use_single_shooting_state_update:
                xnext_upd = self._F(x0=xi_upd, p=ui_upd)['xf'].toarray().flatten()
            else:
                xnext_upd = xnext + (A + B@L)@dx + B@l + d

            self._state_trj[i+1] = xnext_upd.copy()
            self._ctrl_trj[i] = ui_upd.copy()

            defect_norm += np.linalg.norm(d, ord=1)
            du_norm += np.linalg.norm(l, ord=1)
            dx_norm += np.linalg.norm(dx, ord=1)
            self._inter_constr_trj[i] = self._inter_constr(x=xi_upd, u=ui_upd)['h'].toarray().flatten()

        self._defect_norm.append(defect_norm)
        self._du_norm.append(du_norm)
        self._dx_norm.append(dx_norm)
        self._dcost.append(self._eval_cost(self._state_trj, self._ctrl_trj))

    def _propagate(self, xtrj: List[np.array], utrj: List[np.array], alpha=1):

        N = len(utrj)
        ret = self.PropagateResult()

        ret.state_trj = xtrj.copy()
        ret.ctrl_trj = utrj.copy()

        for i in range(N):

            xnext = xtrj[i+1]
            xi = xtrj[i]
            xi_upd = ret.state_trj[i]
            ui = utrj[i]
            d = self._defect[i]
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B
            L = self._fb_gain[i]
            l = alpha * self._ff_u[i]
            dx = np.atleast_1d(xi_upd - xi)

            ui_upd = ui + l + L@dx

            if self._use_single_shooting_state_update:
                xnext_upd = self._F(x0=xi_upd, p=ui_upd)['xf'].toarray().flatten()
            else:
                xnext_upd = xnext + (A + B@L)@dx + B@l + d

            ret.state_trj[i+1] = xnext_upd.copy()
            ret.ctrl_trj[i] = ui_upd.copy()
            ret.dx_norm += np.linalg.norm(dx, ord=1)
            ret.du_norm += np.linalg.norm(ui_upd - ui, ord=1)
            ret.inter_constr.append(self._inter_constr(x=xi_upd, u=ui_upd)['h'].toarray().flatten())

        ret.final_constr = self._final_constraint(x=ret.state_trj[-1])['hf'].toarray().flatten()
        ret.cost = self._eval_cost(ret.state_trj, ret.ctrl_trj)

        return ret

    def _eval_cost(self, x_trj, u_trj):

        cost = 0.0

        for i in range(len(u_trj)):

            cost += self._F(x0=x_trj[i], p=u_trj[i])['qf'].__float__()

        cost += self._final_cost(x=x_trj[-1])['lf'].__float__()

        return cost

    def solve(self, niter: int):

        if len(self._dcost) == 0:
            self._dcost.append(self._eval_cost(self._state_trj, self._ctrl_trj))

        for i in range(niter):

            self._linearize_quadratize()
            self._backward_pass()
            self._forward_pass()

            if self._verbose:
                print('Iter #{}: cost = {}'.format(i, self._dcost[-1]))

    def setInitialState(self, x0: np.array):

        self._state_trj[0] = np.array(x0)

    def randomizeInitialGuess(self):

        self._state_trj[1:] = [np.random.randn(self._nx) for _ in range(self._N)]
        self._ctrl_trj  = [np.random.randn(self._nu) for _ in range(self._N)]

        if self._use_single_shooting_state_update:
            for i in range(self._N):
                self._state_trj[i+1] = self._F(x0=self._state_trj[i], p=self._ctrl_trj[i])['xf'].toarray().flatten()



