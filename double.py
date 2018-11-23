import sys
import numpy as np
from scipy.integrate import odeint
import csv


# The gravitational acceleration (m.s-2).
g = 9.81


def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def solve(L1, L2, m1, m2, tmax, dt, y0):
    t = np.arange(0, tmax+dt, dt)

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))
    theta1, theta2 = y[:,0], y[:,2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return theta1, theta2, x1, y1, x2, y2

def simulate_pendulum(theta_resolution, tmax, dt):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    # Maximum time, time point spacings (all in s).
    #tmax, dt = 30.0, 0.01

    with open('double-pendulum.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for theta1_init in np.linspace(0, 2*np.pi, theta_resolution):
            for theta2_init in np.linspace(0, 2*np.pi, theta_resolution):
                # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
                y0 = np.array([
                    theta1_init,
                    0.0,
                    theta2_init,
                    0.0
                ])

                theta1, theta2, x1, y1, x2, y2 = solve(L1, L2, m1, m2, tmax, dt, y0)
                #print theta1_init, theta2_init, theta1[-1], theta2[-1]
                spamwriter.writerow([theta1_init, theta2_init, theta1, theta2, x1, y1, x2, y2]);


def do_the_thing(theta_resolution, tmax, dt):
    simulate_pendulum(theta_resolution, tmax, dt)