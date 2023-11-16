import numpy as np

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity

def compute_forces(bodies):
    G = 6.67430e-11
    forces = []
    for i in range(len(bodies)):
        force = np.zeros(3)
        for j in range(len(bodies)):
            if i != j:
                r = bodies[j].position - bodies[i].position
                force += G * bodies[i].mass * bodies[j].mass * r / np.linalg.norm(r)**3
        forces.append(force)
    return forces

def integrate(bodies, dt):
    forces = compute_forces(bodies)
    for i in range(len(bodies)):
        bodies[i].velocity += forces[i] / bodies[i].mass * dt
        bodies[i].position += bodies[i].velocity * dt

def solve_n_body_problem(bodies, dt, steps):
    for _ in range(steps):
        integrate(bodies, dt)