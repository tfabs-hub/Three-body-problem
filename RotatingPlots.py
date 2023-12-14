import numpy as np
import matplotlib.pyplot as plt
from RotatingFrame import solve_n_body_problem, Body, plot_helper


##L1 Lagrange Points
'''bodies = [Body(50, np.array([-1.0,0.0,0.0]), np.array([0.0,3.8,0.0])),
          Body(50, np.array([1.0,0.0,0.0]), np.array([0.0,-3.8,0.0])),
          Body(2, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]))]'''




##L2/L3 Lagrange Points
'''bodies = [Body(2, np.array([0.0,-0.5,0.0]), np.array([-1,0,0.0])),
          Body(2, np.array([0.0,0.5,0.0]), np.array([1,0.0,0.0]))
          ,Body(.01, np.array([0.0,-.5992,0.0]), np.array([1.1984,0.0,0.0]))]'''

          
   
##L4/5 Lagrange Points
bodies = [Body(25, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0])),
          Body(1, np.array([1.0,0.0,0.0]), np.array([0.0,5.099,0.0])),
          Body(.01, np.array([-0.4615,0.866,0.0]), np.array([-4.33,-2.36,0.0]))]


    
#THE CODE BELOW TRANSLATES VELOCITIES TO ROTATING FRAME

# Calculate omega for the rotating frame
G = 1  # gravitational constant
m1 = bodies[0].mass
m2 = bodies[1].mass
r = np.linalg.norm(bodies[0].position - bodies[1].position)
omega = np.sqrt(G * (m1 + m2) / r**3)

# Define omega as a 3D vector
omega_vector = np.array([0, 0, omega])

# Update the velocities to the rotating frame
for body in bodies:
    body.velocity = body.velocity - np.cross(omega_vector, body.position)
    
flat_r_arr = solve_n_body_problem(bodies, 0.0001, 50000)

plot_helper(bodies, flat_r_arr)
plt.title("L4 and L5 in a Rotating Frame")
plt.show()