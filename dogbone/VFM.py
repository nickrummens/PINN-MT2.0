import os
import numpy as np
import deepxde as dde
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

np.set_printoptions(edgeitems=30, linewidth = 1000000)

# Variables
R = 20
theta = np.arccos(15/R)
b = 10
indent_x = R - (R * np.cos(theta))
indent_y = R * np.sin(theta)
free_length = 110
L_u = 60
L_c = free_length - (2*indent_y)
H_clamp = 0
total_points_vert = 140 #see geometry mapping code
total_points_hor = 40
x_max_FEM = (2*indent_x) + b
y_max_FEM = 2*H_clamp + 2*indent_y + L_c

#ROI positioning
offs_x = indent_x
offs_y = indent_y + H_clamp + (L_c-L_u)/2
x_max_ROI = b
y_max_ROI = L_u


E_actual  = 69e3   # Actual Young's modulus 210 GPa = 210e3 N/mm^2
nu_actual = 0.33     # Actual Poisson's ratio


p_stress = 9 #360N/(20mmx2mm)

n_DIC = 10
noise_DIC = 0

def transform_coords(x):
    """
    For SPINN, if the input x is provided as a list of 1D arrays (e.g., [X_coords, Y_coords]),
    this function creates a 2D meshgrid and stacks the results into a 2D coordinate array.
    """
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
    return dde.backend.stack(x_mesh, axis=-1)

# Load data
dir_path = os.path.dirname(os.path.realpath(__file__))
fem_file = os.path.join(dir_path, r"data_fem", 'fem_solution_dogbone_experiments.dat')
data = np.loadtxt(fem_file)
X_val      = data[:, :2]
u_val      = data[:, 2:4]
strain_val = data[:, 4:7]
stress_val = data[:, 7:10]
solution_val = np.hstack((u_val, stress_val))

n_mesh_points = [total_points_hor,total_points_vert]
x_grid = np.linspace(0, x_max_FEM, n_mesh_points[0])
y_grid = np.linspace(0, y_max_FEM, n_mesh_points[1])

def create_interpolation_fn(data_array):
    num_components = data_array.shape[1]
    interpolators = []
    for i in range(num_components):
        interp = RegularGridInterpolator(
            (x_grid, y_grid),
            data_array[:, i].reshape(n_mesh_points[1], n_mesh_points[0]).T,
        )
        interpolators.append(interp)
    def interpolation_fn(x):
        x_in = transform_coords([x[0], x[1]])
        return np.array([interp((x_in[:, 0], x_in[:, 1])) for interp in interpolators]).T
    return interpolation_fn

solution_fn = create_interpolation_fn(solution_val)
strain_fn   = create_interpolation_fn(strain_val)


# Create simulated data
X_DIC_input = [np.linspace(offs_x, offs_x + x_max_ROI, n_DIC).reshape(-1, 1),
               np.linspace(offs_y, offs_y + y_max_ROI, n_DIC).reshape(-1, 1)]
X_DIC_input_ref = [np.linspace(0, 20, n_DIC).reshape(-1, 1),
               np.linspace(offs_y, offs_y + y_max_ROI, n_DIC).reshape(-1, 1)]
DIC_solution = solution_fn(X_DIC_input_ref)
DIC_data = strain_fn(X_DIC_input_ref)[:, :3]
DIC_data += np.random.normal(0, noise_DIC, DIC_data.shape)

# Generate output matrices
X1, X2 = np.meshgrid(X_DIC_input[0], X_DIC_input[1])
Eps1= DIC_data[:,0]
Eps2 = DIC_data[:,1]
Eps6 = DIC_data[:,2]
Eps1 = Eps1.reshape(X1.shape)
Eps2 = Eps2.reshape(X1.shape)
Eps6 = Eps6.reshape(X1.shape) # *2? engineering strain
X1 -= 5
X2 -= 25

# print("X1")
# print(X1)
# print("X2")
# print(X2)
# print("Eps1")
# print(Eps1)
# print("Eps2")
# print(Eps2)
# print("Eps6")
# print(Eps6)

# Constants
F = 360 #MUST CORRESPOND TO PSTRESS FOR SIMULATED DATA!!!!! CHECK FEM FILE
t = 2
w = 10
h = 60
Sd = h*w

# print(Eps1[5,5])
# print(Eps2[5,5])

# Eps1 = np.ones((10,10))*-0.000087
# Eps2 = np.ones((10,10))*0.00026
# Eps6 = np.zeros((10,10))

print(np.mean(Eps1))
print(np.mean(Eps2))
print(np.mean(Eps6))




"""CALCULATED"""
# Calculation of the components of matrix A


A = np.zeros((2, 2))

#Field 1.2: u2 = x2
A[0, 0] = np.mean(Eps2) * Sd
A[0, 1] = np.mean(Eps1) * Sd

#Field 1.1: u1 = x2(x2 - h)x1
A[1, 0] = (np.mean( ( Eps1 * X2 * (X2-h) )) * Sd) + (np.mean( Eps6 * ( ( 2 * X2 ) - h ) * X1) * Sd)
A[1, 1] = (np.mean( ( Eps2 * X2 * (X2-h) ) ) * Sd) - (np.mean( Eps6 * ( ( 2 * X2 ) - h ) * X1 ) * Sd)

# A[1, 0] = np.mean(Eps1)
# A[1, 1] = np.mean(Eps2)

# Calculation of the virtual work of the external forces
B = np.zeros(2)
B[0] = F*h/t 
B[1] = 0  

# Identification of the stiffness components
# Q = np.linalg.solve(A, B)
Q = np.linalg.inv(A) @ B.T

# E and Nu from Q
Nu = Q[1] / Q[0]
E = Q[0] * (1 - Nu**2)

# Compute the condition number of A
cond_A = np.linalg.cond(A)
print(f'Condition number of A: {cond_A:.6f}')

# Display the result
print(f'Elastic modulus (E): {E:.6f}')
print(f'Poisson ratio (Nu): {Nu:.6f}')


Nu = -(np.mean(X2*(X2-h)*Eps1) + np.mean(X1*(2*X2 - h)*Eps6)) / (np.mean(X2*(X2-h)*Eps2) - np.mean(X1*(2*X2 - h)*Eps6))
print(f'Poisson ratio (Nu): {Nu:.6f}')

E = (1-Nu**2)*F*h/(t*Sd*(np.mean(Eps2) + Nu*np.mean(Eps1)))
print(f'Elastic modulus (E): {E:.6f}')