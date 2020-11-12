import ilqr
import casadi as cs
import rospy
import numpy as np
from matplotlib import pyplot as plt


def doit():

    # state vector is (x, y, theta)
    x = cs.SX.sym('x', 3)

    # control vector is (v, omega)
    u = cs.SX.sym('u', 2)

    # explicit variables for readability
    v = u[0]
    omega = u[1]
    theta = x[2]

    # unicycle kinematic model
    xdot = cs.vertcat(v*cs.cos(theta),
                      v*cs.sin(theta),
                      omega)

    # intermediate cost, final cost, final constraint
    l = cs.sumsqr(u)
    xf_des = np.array([0, 1, 0])
    lf = 1000.0 * cs.sumsqr(x - xf_des)
    hf = x - xf_des

    # create solver
    solver = ilqr.IterativeLQR(x=x,
                               u=u,
                               xdot=xdot,
                               dt=0.1,
                               N=100,
                               intermediate_cost=l,
                               final_cost=lf,
                               final_constraint=None)
    # solve
    solver.randomizeInitialGuess()
    solver.solve(50)

    plt.figure()
    plt.title('Total cost')
    plt.semilogy(solver._dcost, '--s')
    plt.grid()

    plt.figure()
    plt.title('State trajectory')
    plt.plot(solver._state_trj)
    plt.legend(['x', 'y', 'theta'])
    plt.grid()

    plt.figure()
    plt.title('Control trajectory')
    plt.plot(solver._ctrl_trj)
    plt.legend(['v', 'omega'])
    plt.grid()

    plt.show()


if __name__ == '__main__':
    doit()
