import sys
import numpy as np
from scipy.integrate import odeint
import csv

#based on https://scipython.com/blog/the-double-pendulum/

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

def mappableSolve(args):
	L1, L2, m1, m2, tmax, dt, theta1_init, theta2_init = args
	y0 = np.array([
		theta1_init,
		0.0,
		theta2_init,
		0.0
	])
    #vracam i thete da znam za cega su ovi rezultati (da ih posle mogu
    #pisati u fajl)
	return (theta1_init, theta2_init, solve(L1, L2, m1, m2, tmax, dt, y0))
                


def simulate_pendulum(theta_resolution, tmax, dt, filename, parallel):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    # Maximum time, time point spacings (all in s).
    #tmax, dt = 30.0, 0.01

    theta1_inits = np.linspace(0, 2*np.pi, theta_resolution)
    theta2_inits = np.linspace(0, 2*np.pi, theta_resolution)

    import itertools
    t1t2_inits = itertools.product(theta1_inits, theta2_inits)
    params = [[L1, L2, m1, m2, tmax, dt, t1t2_i[0], t1t2_i[1]] for t1t2_i in t1t2_inits]
	
    if parallel:
        from multiprocessing import Pool
        pool = Pool(processes=4)
        solutions = pool.map(mappableSolve, params)
    else:
        solutions = map(mappableSolve, params)

    with open(filename, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["theta1_init", "theta2_init", "theta1_last", "theta2_last", "x1_last", "y1_last", "x2_last", "y2_last"])      
        for t1i, t2i, results in solutions:
           theta1, theta2, x1, y1, x2, y2 = results #to je ono sto je solve izracunao
           csvwriter.writerow([t1i, t2i, theta1[-1], theta2[-1], x1[-1], y1[-1], x2[-1], y2[-1]])

def plot_the_thing(filename):
	
	
	rows = []
	with open(filename, 'rb') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		next(csvreader, None)	#skip header
		for row in csvreader:
			rows.append(row)

	x1_lasts = [float(row[4]) for row in rows]
	y1_lasts = [float(row[5]) for row in rows]
	x2_lasts = [float(row[6]) for row in rows]
	y2_lasts = [float(row[7]) for row in rows]

	from matplotlib import pyplot as plt
	#t = range(0, len(rows[0]))
	#plt.plot(t, rows[0], label="theta 1")
	plt.plot(x1_lasts, y1_lasts, 'ro', label='1')
	plt.plot(x2_lasts, y2_lasts, 'go', label='2')
	#plt.axis([-1, 1, -1, 1])
	plt.legend()
	plt.savefig("points.png")


def do_the_thing(theta_resolution, tmax, dt, filename, graph, parallel):
    simulate_pendulum(theta_resolution, tmax, dt, filename, parallel)
    if graph:
        plot_the_thing(filename)
