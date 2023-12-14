import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity

def compute_forces(bodies):
    """
    Calculates force on each body given current positions

    :param bodies: all bodies in consideration
    :return: forces on the bodies, 1d-array of size n
    """
    G = 1
    forces = []
    for i in range(len(bodies)):
        force = np.zeros(3)
        for j in range(len(bodies)):
            if i != j:
                r = bodies[j].position - bodies[i].position
                force += G * bodies[i].mass * bodies[j].mass * r / np.linalg.norm(r)**3
        forces.append(force)
    return np.array(forces)

def compute_2_body_period(r_arr, dt):
    """
    Used especially for 2-body systems; calculates the period

    :param r_arr: position array
    :param dt: time step
    :return: None
    """
    d_lst = []  # distances between the two bodies
    for each_time in r_arr:
        # each_time refers to the positions of both bodies at each given time (3x2)
        position_diff = each_time[1] - each_time[0]  # distance between the two bodies (vector)
        d_lst.append(np.linalg.norm(position_diff))
    d_arr = np.array(d_lst)
    min_d_times = argrelextrema(d_arr, np.less)[0]  # times (as indices) when distance is (local) minimum
    if len(min_d_times) > 1:
        period = (min_d_times[1] - min_d_times[0]) * dt
        print("Period is:", period)
    else:
        print("Not enough data to calculate the period of the two-body system.")


def integrate(bodies, dt, omega):
    """
    Integrator that uses Verlet method and works in the rotating frame

    :param bodies: all bodies in consideration
    :param dt: time step
    :param omega: angular velocity of the rotating frame
    :return: None
    """
    for i in range(len(bodies)):
        # x_1 = x_0 + dt * v_(1/2)
        bodies[i].position += dt * bodies[i].velocity
    forces = compute_forces(bodies)  # F(x_1)
    for i in range(len(bodies)):
        # v_(3/2) = v_(1/2) + F(x_1) * dt - omega x x_1
        bodies[i].velocity += forces[i] / bodies[i].mass * dt
        # account for Coriolis and centrifugal forces in the rotating frame
        bodies[i].velocity -= 2 * np.cross(omega, bodies[i].velocity) * dt
        bodies[i].velocity -= np.cross(omega, np.cross(omega, bodies[i].position)) * dt

def solve_n_body_problem(bodies, dt, steps):
    """
    Solves the problem using the integrator

    :param bodies: all bodies in consideration
    :param dt: time step
    :param steps: number of steps
    :return: flat_r_arr: flattened position array
    """
    tot_m = 0    # total mass of bodies
    for body in bodies:
        tot_m += body.mass
    # initializations
    r_lst = []    # absolute positions
    r_cm_lst = []     # CM frame positions
    ini_forces = compute_forces(bodies)  # F(x_0)
    for i in range(len(bodies)):
        bodies[i].velocity += (ini_forces[i] / bodies[i].mass) * (dt / 2)  # v_(1/2)
        
    # Calculate omega for the rotating frame
    G = 1  # gravitational constant
    m1 = bodies[0].mass
    m2 = bodies[1].mass
    r = np.linalg.norm(bodies[0].position - bodies[1].position)
    omega = np.sqrt(G * (m1 + m2) / r**3)
    omega_vector = np.array([0, 0, omega])

    # integrate
    for _ in range(steps):
        current_r = []
        for body in bodies:
            current_r.append(body.position)
        r_lst.append(np.array(current_r))  # necessary to create new array object to avoid referencing same memory location
        # center of mass frame
        r_cm = 0     # position vector of center of mass
        for i in range(len(bodies)):
            r_cm += bodies[i].mass * current_r[i]
        r_cm /= tot_m
        current_r_cm = np.array(current_r) - r_cm         # positions in the CM frame
        r_cm_lst.append(current_r_cm)
        # proceed in time
        integrate(bodies, dt, omega_vector)
    r_arr = np.array(r_cm_lst)        # for observer's frame, replace by r_arr = np.array(r_lst)
    print(r_arr)  # overview of the positions
    flat_r_arr = r_arr.flatten()

    # the following can be used for testing the period of two-body systems
    if len(bodies) == 2:
        compute_2_body_period(r_arr, dt)

    return flat_r_arr

def plot_helper(bodies, flat_r_arr):
    """
    Helper function that helps plot;
    to customize size of figure or animate, use fig = plt.figure()

    :param bodies: all bodies in consideration
    :param flat_r_arr: flattened position array from solve_n_body_problem()
    :return: None
    """
    for i in range(len(bodies)):
        mask_x = np.zeros(len(flat_r_arr), dtype=int)
        mask_y = np.zeros(len(flat_r_arr), dtype=int)
        for j in range(len(flat_r_arr)):
            if j % (3 * len(bodies)) == 0:
                mask_x[j + 3 * i] = 1
            elif j % (3 * len(bodies)) == 1:
                mask_y[j + 3 * i] = 1
        print(mask_x)
        x = flat_r_arr[mask_x != 0]
        y = flat_r_arr[mask_y != 0]
        print(x, y)
        plt.scatter(x,y,s=1)
        plt.xlabel("$x$")
        plt.ylabel("$y$")

def plot_solution(bodies, dt, steps):
    """
    Plots the solution to an n-body problem

    :param bodies: all bodies in consideration
    :param dt: time step
    :param steps: number of steps
    :return: None
    """
    flat_r_arr = solve_n_body_problem(bodies, dt, steps)
    plot_helper(bodies, flat_r_arr)
    plt.show()

