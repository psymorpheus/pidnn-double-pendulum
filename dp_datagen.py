import numpy as np
import sympy as smp
from scipy.integrate import odeint
from tqdm import tqdm

def iter(the_i1, the_i2, tv, gv, m1v, m2v, l1v, l2v):
    #Define Constants
    t, g = smp.symbols('t g')
    m1, m2 = smp.symbols('m1 m2')
    L1, L2 = smp.symbols('L1, L2')

    #Define variables w.r.t. time
    the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
    the1 = the1(t)
    the2 = the2(t)

    #Define differentials for the1(t) & the2(t)
    the1_d = smp.diff(the1, t)
    the2_d = smp.diff(the2, t)
    the1_dd = smp.diff(the1_d, t)
    the2_dd = smp.diff(the2_d, t)

    #Calculations for x&y cordinates of m1 and m2
    x1 = L1*smp.sin(the1)
    y1 = -L1*smp.cos(the1)
    x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
    y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)

    #Calculations for kinetic and potential energy
    # Kinetic
    T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
    T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
    T = T1+T2
    # Potential
    V1 = m1*g*y1
    V2 = m2*g*y2
    V = V1 + V2
    # Lagrangian
    L = T-V

    #Generation of Lagrangian differential equations
    LE1 = (smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t)).simplify()
    LE2 = (smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t)).simplify()

    #Problem Definition
    sols = smp.solve([LE1, LE2], (the1_dd, the2_dd),
                simplify=False, rational=False)

    #Linearization of second order differential equations
    dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the1_dd])
    dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the2_dd])
    dthe1dt_f = smp.lambdify(the1_d, the1_d)
    dthe2dt_f = smp.lambdify(the2_d, the2_d)

    #Matrix function representation of system of linear equations obtained
    def dSdt(S, t, g, m1, m2, L1, L2):
        the1, z1, the2, z2 = S
        return [
            dthe1dt_f(z1),
            dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
            dthe2dt_f(z2),
            dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        ]
    return odeint(dSdt, y0=[the_i1, 0, the_i2, 0], t=tv, args=(gv,m1v,m2v,l1v,l2v))

def double_pendulum_data(config):
    t = config['t_range']
    data = np.array([]).reshape(0,5)
    for theta1 in tqdm(config['theta_range'], desc = 'Dynamic pendulum data generation progress'):
        for theta2 in config['theta_range']:
            # Format - each row [theta1, omega1, theta2, omega2]
            solved_data = iter(theta1, theta2, t, config['g'], config['m1'], config['m2'], config['l1'], config['l2'])
            iteration_data = np.hstack([
                np.reshape(t,(-1,1)),
                theta1*np.ones(shape=(len(t),1)),
                theta2*np.ones(shape=(len(t),1)),
                solved_data[:,[0]]-theta1*np.ones(shape=(len(t),1)),
                solved_data[:,[2]]-theta2*np.ones(shape=(len(t),1))
            ])
            data = np.vstack((data, iteration_data))

    if config['save_collected']:
        np.savetxt(config['datadir'] + config['datafile'], data, delimiter=",")
