import numpy as np
import matplotlib.pyplot as plt
from nbody import solve_n_body_problem, Body, plot_helper


######THOMAS




##L1 Lagrange Points
bodies = [Body(50, np.array([-1.0,0.0,0.0]), np.array([0.0,3.8,0.0])),
          Body(50, np.array([1.0,0.0,0.0]), np.array([0.0,-3.8,0.0])),
          Body(2, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]))]

flat_r_arr = solve_n_body_problem(bodies, 0.0001, 50000)

plot_helper(bodies, flat_r_arr)
plt.title('L1 in Rotating Frame')
plt.show()

##L2/L3 Lagrange Points
bodies = [Body(2, np.array([0.0,-0.5,0.0]), np.array([-1.0,0,0.0])),
          Body(2, np.array([0.0,0.5,0.0]), np.array([1.0,0.0,0.0])),
          Body(.01, np.array([0.0,-.5992,0.0]), np.array([1.1984,0.0,0.0]))]
         

flat_r_arr = solve_n_body_problem(bodies, 0.0001, 50000)

plot_helper(bodies, flat_r_arr)
plt.title('L2/L3 in Rotating Frame')
plt.show()
   
##L4/5 Lagrange Points
bodies = [Body(25, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0])),
          Body(1, np.array([1.0,0.0,0.0]), np.array([0.0,5.099,0.0])),
          Body(.01, np.array([-0.4615,0.866,0.0]), np.array([-4.33,-2.36,0.0]))
          ]



flat_r_arr = solve_n_body_problem(bodies, 0.0001, 50000)

plot_helper(bodies, flat_r_arr)
plt.title('L4/L5 in Rotating Frame')
plt.show()


## Figure eight

# Define the masses and initial conditions for the figure-eight solution
mass = 1  # for equal masses
# Positions
pos1 = np.array([-0.97000436, 0.24308753, 0])
pos2 = np.array([0, 0, 0])
pos3 = np.array([0.97000436, -0.24308753, 0])

vel1 = np.array([0.466203685, 0.43236573, 0])
vel2 = np.array([-0.93240737, -0.86473146, 0])
vel3 = np.array([0.466203685, 0.43236573, 0])

# Create the bodies
body1 = Body(mass, pos1, vel1)
body2 = Body(mass, pos2, vel2)
body3 = Body(mass, pos3, vel3)

# List of bodies
bodies = [body1, body2, body3]


    
    
flat_r_arr = solve_n_body_problem(bodies, 0.0001, 50000)

plot_helper(bodies, flat_r_arr)
plt.title('Three Bodies of Similar Masses')
plt.show()