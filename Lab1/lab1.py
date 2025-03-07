import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.colors import LogNorm

# Constants
G = 6.67e-11  # gravitational constant
M_E = 5.9e24  # earth mass
M_M = 7.3e22  # moon mass
R_E = 6378e3  # earth radius
R_M = 1737e3  # moon radius
d_EM = 3.8e8  # earth to moon distance
g = 9.81  # gravity

ve = 2.4e3  # exhaust velocity 
m_dot = 1.3e4  # burn rate 
m0 = 2.4e6  # initial mass 
mf = 7.5e5  # final mass

#part 1

# 1. function for gravitational potential
def gravitational_potential(M, xM, yM, x, y):
    r = np.sqrt((x - xM)**2 + (y - yM)**2)
    if np.any(r == 0):
        return np.inf * np.ones_like(r) 
    return -G * M / r

# 2. plot
x_values = np.linspace(R_E, 1.5 * d_EM, 500)
phi_values = np.abs(gravitational_potential(M_E, 0, 0, x_values, np.zeros_like(x_values)))

plt.figure(figsize=(8,6))
plt.plot(x_values / 1e6, phi_values)
plt.yscale("log")
plt.xlabel("Distance from Earth Surface (x10^6 m)")
plt.ylabel("|Φ| (J/kg)")
plt.title("Gravitational Potential |Φ| vs Distance")
plt.grid()
plt.show()

# 3. gravitational potential plot
x = np.linspace(-1.5 * d_EM, 1.5 * d_EM, 300)
y = np.linspace(-1.5 * d_EM, 1.5 * d_EM, 300)
X, Y = np.meshgrid(x, y)

Phi = np.abs(gravitational_potential(M_E, 0, 0, X, Y))

plt.figure(figsize=(8,6))
plt.pcolormesh(X / 1e6, Y / 1e6, np.log10(Phi), shading='auto')
plt.colorbar(label="log(|Φ|) (J/kg)")
plt.xlabel("x (x10^6 m)")
plt.ylabel("y (x10^6 m)")
plt.title("Gravitational Potential of Earth")
plt.axis("equal")
plt.show()

x_moon = d_EM / np.sqrt(2)
y_moon = d_EM / np.sqrt(2)

Phi_total = np.abs(gravitational_potential(M_E, 0, 0, X, Y)) + np.abs(gravitational_potential(M_M, x_moon, y_moon, X, Y))

# Fixed 2D Color Mesh Plot for Earth-Moon Potential
plt.figure(figsize=(8,6))
plt.pcolormesh(X / 1e6, Y / 1e6, np.log10(Phi_total), shading='auto')
plt.colorbar(label="log(|Φ|) (J/kg)")
plt.xlabel("x (Million meters)")
plt.ylabel("y (Million meters)")
plt.title("Gravitational Potential of Earth-Moon System")
plt.axis("equal")
plt.show()

# Fixed Contour Plot for Earth-Moon Potential
plt.figure(figsize=(8,6))
contour = plt.contour(X / 1e6, Y / 1e6, Phi_total, levels=50, norm=LogNorm())
plt.colorbar(contour, label="|Gravitational Potential| (J/kg)")
plt.xlabel("x (Million meters)")
plt.ylabel("y (Million meters)")
plt.title("Gravitational Potential Contours (Earth-Moon System)")
plt.axis('equal')
plt.show()

#part 3
def gravitational_force(M1, x1, y1, x2, y2):
    r_vec = np.array([x2 - x1, y2 - y1])
    r_mag = np.linalg.norm(r_vec, axis=0)
    r_mag = np.where(r_mag == 0, np.inf, r_mag)  
    F_mag = G * M1 / r_mag**2
    F_vec = -F_mag * (r_vec / r_mag)
    return F_vec[0], F_vec[1]

Fx_E, Fy_E = gravitational_force(M_E, 0, 0, X, Y)
Fx_M, Fy_M = gravitational_force(M_M, x_moon, y_moon, X, Y)

Fx_total = Fx_E + Fx_M
Fy_total = Fy_E + Fy_M

# streamplot of gravitational force field
plt.figure(figsize=(8,6))
plt.streamplot(X / 1e6, Y / 1e6, Fx_total, Fy_total, density=1, color=np.log10(np.sqrt(Fx_total**2 + Fy_total**2)), cmap="plasma")
plt.colorbar(label="log(|F|) (N/kg)")
plt.xlabel("x (x10^6 m)")
plt.ylabel("y (x10^6 m)")
plt.title("Gravitational Force Field of Earth-Moon System")
plt.axis("equal")
plt.show()

#part 4
#1. burn time
T = (m0 - mf) / m_dot

#2. function definition
def delta_v(t, m0, mf, m_dot, ve, g):
    m_t = np.maximum(m0 - m_dot * t, mf)  
    dv = ve * np.log(m0 / m_t) - g * t  
    return dv

#3. altitude
altitude, _ = integrate.quad(lambda t: delta_v(t, m0, mf, m_dot, ve, g), 0, T)

print("Burn Time (T):", T)
print("Altitude:", altitude)
