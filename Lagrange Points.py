import numpy as np
import matplotlib.pyplot as plt


G = 1  
m1 = m2 = 1  
Omega = 1 

#meshgrid
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)

#Defining Radii
r1 = np.sqrt((X + 0.5)**2 + Y**2)
r2 = np.sqrt((X - 0.5)**2 + Y**2)

# Effective potential -- NOT SURE ABOUT THIS STILL
Ueff = -G*m1/r1 - G*m2/r2 - 0.5*Omega**2*(X**2 + Y**2)


plt.figure(figsize=(6, 6))
plt.imshow(Ueff, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='RdGy', alpha=0.5)
plt.colorbar(label='Effective potential')


plt.plot(-0.5, 0, 'bo')
plt.plot(0.5, 0, 'ro')


plt.title('Effective potential and Lagrange points')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
