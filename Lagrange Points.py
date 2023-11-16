import numpy as np
import matplotlib.pyplot as plt



G = 1  
m1 = 2
m2 = 1
Omega = 1

#meshgrid
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

#effective potential
r1 = np.sqrt((X+1)**2 + Y**2)
r2 = np.sqrt((X-1)**2 + Y**2)
#Ueff = -G*m1/np.abs(np.sqrt(X**2 + Y**2)-r1) - G*m2/np.abs(np.sqrt(X**2 + Y**2)-r2) - 0.5*Omega**2*(X**2 + Y**2)
Ueff = -G * (m1 / r1 + m2 / r2) + 0.5 * Omega**2 * (X**2 + Y**2)

Ueff_min = np.min(Ueff)
Ueff_max = np.max(Ueff)



fig, ax = plt.subplots()


fig.patch.set_facecolor('black')
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Ueff, levels=np.linspace(Ueff_min, Ueff_max, 100), cmap='RdYlBu')
plt.title('Effective Potential')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Ueff')
plt.grid(True)
plt.show()