import os
import numpy as np
import deepxde as dde
import jax

import time
import json
import argparse
import platform
import subprocess

import jax.numpy as jnp
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

np.set_printoptions(edgeitems=30, linewidth = 1000000)

# parser = argparse.ArgumentParser(description="Physics Informed Neural Networks for Linear Elastic Plate")
# parser.add_argument('--n_iter', type=int, default=int(1e10), help='Number of iterations')


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


E_actual  = 132.1353e3 #69e3   # Actual Young's modulus 210 GPa = 210e3 N/mm^2
nu_actual = 0.3120 # 0.33     # Actual Poisson's ratio


p_stress = 9 #360N/(20mmx2mm)

n_DIC = 4
noise_DIC = 1e-6

def transform_coords(x):
    """
    For SPINN, if the input x is provided as a list of 1D arrays (e.g., [X_coords, Y_coords]),
    this function creates a 2D meshgrid and stacks the results into a 2D coordinate array.
    """
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
    return dde.backend.stack(x_mesh, axis=-1)



"""
SIMULATED DATA
"""
# dir_path = os.path.dirname(os.path.realpath(__file__))
# dic_path = os.path.join(dir_path, "DIC_data")

# # SIMULATED DIC: speckle_pattern_Numerical_1_0.synthetic.tif_X_trans.csv etc.
# # REAL-WORLD DIC: Image_0020_0.tiff_X_trans.csv

# X_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_X_trans.csv"), delimiter=";",dtype=str)
# X_dic = X_dic.replace({',': '.'}, regex=True)
# X_dic = X_dic.apply(pd.to_numeric, errors='coerce')
# X_dic = X_dic.dropna(axis=1)
# X_dic = X_dic.to_numpy()
# X_dic += 5

# Y_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_Y_trans.csv"), delimiter=";",dtype=str)
# Y_dic = Y_dic.replace({',': '.'}, regex=True)
# Y_dic = Y_dic.apply(pd.to_numeric, errors='coerce')
# Y_dic = Y_dic.dropna(axis=1)
# Y_dic = Y_dic.to_numpy()
# # #real world data:
# # Y_dic += 25

# E_xx_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_exx.csv"), delimiter=";",dtype=str)
# E_xx_dic = E_xx_dic.replace({',': '.'}, regex=True)
# E_xx_dic = E_xx_dic.apply(pd.to_numeric, errors='coerce')
# E_xx_dic = E_xx_dic.dropna(axis=1)
# E_xx_dic = E_xx_dic.to_numpy()
# # E_xx_dic = E_xx_dic.reshape(-1,1)


# E_yy_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_eyy.csv"), delimiter=";",dtype=str)
# E_yy_dic = E_yy_dic.replace({',': '.'}, regex=True)
# E_yy_dic = E_yy_dic.apply(pd.to_numeric, errors='coerce')
# E_yy_dic = E_yy_dic.dropna(axis=1)
# E_yy_dic = E_yy_dic.to_numpy()
# # E_yy_dic = E_yy_dic.reshape(-1,1)


# E_xy_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_exy.csv"), delimiter=";",dtype=str)
# E_xy_dic = E_xy_dic.replace({',': '.'}, regex=True)
# E_xy_dic = E_xy_dic.apply(pd.to_numeric, errors='coerce')
# E_xy_dic = E_xy_dic.dropna(axis=1)
# E_xy_dic = E_xy_dic.to_numpy()
# # E_xy_dic = E_xy_dic.reshape(-1,1)

# rows_in_range_y = []
# rows_in_range_x = []
# rows_in_range_exx = []
# rows_in_range_eyy = []
# rows_in_range_exy = []

# for row_y, row_x, row_exx, row_eyy, row_exy in zip(Y_dic, X_dic, E_xx_dic, E_yy_dic, E_xy_dic):
#     for value in row_y:
#         if 25 <= value <= 85:
#             rows_in_range_y.append(row_y)
#             rows_in_range_x.append(row_x)
#             rows_in_range_exx.append(row_exx)
#             rows_in_range_eyy.append(row_eyy)
#             rows_in_range_exy.append(row_exy)
#             break
# Y_dic = np.array(rows_in_range_y)
# X_dic = np.array(rows_in_range_x)
# E_xx_dic = np.array(rows_in_range_exx)
# E_yy_dic = np.array(rows_in_range_eyy)
# E_xy_dic = np.array(rows_in_range_exy)

# # Y_dic = np.array(rows_in_range_y)[::20, ::5]
# # X_dic = np.array(rows_in_range_x)[::20, ::5]
# # E_xx_dic = np.array(rows_in_range_exx)[::20, ::5]
# # E_yy_dic = np.array(rows_in_range_eyy)[::20, ::5]
# # E_xy_dic = np.array(rows_in_range_exy)[::20, ::5]

# # Y_dic = Y_dic[::10, ::10]
# # X_dic = X_dic[::10, ::10]
# # E_xx_dic = E_xx_dic[::10, ::10]
# # E_yy_dic = E_yy_dic[::10, ::10]
# # E_xy_dic = E_xy_dic[::10, ::10]


# print(Y_dic.shape)

# E_xx_dic = E_xx_dic.reshape(-1,1)
# E_yy_dic = E_yy_dic.reshape(-1,1)
# E_xy_dic = E_xy_dic.reshape(-1,1)

# x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
# y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)
# X_DIC_input = [x_values, y_values]
# DIC_data = np.hstack([E_xx_dic, E_yy_dic, E_xy_dic])


"""
REAL WORLD DATA
"""
dir_path = os.path.dirname(os.path.realpath(__file__))
dic_path = os.path.join(dir_path, "DIC_data")

# SIMULATED DIC: speckle_pattern_Numerical_1_0.synthetic.tif_X_trans.csv etc.
# REAL-WORLD DIC: Image_0020_0.tiff_X_trans.csv

X_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0_corr.tiff_X_trans.csv"), delimiter=";",dtype=str)
X_dic = X_dic.replace({',': '.'}, regex=True)
X_dic = X_dic.apply(pd.to_numeric, errors='coerce')
X_dic = X_dic.dropna(axis=1)
X_dic = X_dic.to_numpy()
X_dic += 5.5

Y_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0_corr.tiff_Y_trans.csv"), delimiter=";",dtype=str)
Y_dic = Y_dic.replace({',': '.'}, regex=True)
Y_dic = Y_dic.apply(pd.to_numeric, errors='coerce')
Y_dic = Y_dic.dropna(axis=1)
Y_dic = Y_dic.to_numpy()
#real world data:
Y_dic += 25.5

E_xx_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0_corr.tiff_exx.csv"), delimiter=";",dtype=str)
E_xx_dic = E_xx_dic.replace({',': '.'}, regex=True)
E_xx_dic = E_xx_dic.apply(pd.to_numeric, errors='coerce')
E_xx_dic = E_xx_dic.dropna(axis=1)
E_xx_dic = E_xx_dic.to_numpy()
# E_xx_dic = E_xx_dic.reshape(-1,1)


E_yy_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0_corr.tiff_eyy.csv"), delimiter=";",dtype=str)
E_yy_dic = E_yy_dic.replace({',': '.'}, regex=True)
E_yy_dic = E_yy_dic.apply(pd.to_numeric, errors='coerce')
E_yy_dic = E_yy_dic.dropna(axis=1)
E_yy_dic = E_yy_dic.to_numpy()
# E_yy_dic = E_yy_dic.reshape(-1,1)


E_xy_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0_corr2.tiff_exy.csv"), delimiter=";",dtype=str)
E_xy_dic = E_xy_dic.replace({',': '.'}, regex=True)
E_xy_dic = E_xy_dic.apply(pd.to_numeric, errors='coerce')
E_xy_dic = E_xy_dic.dropna(axis=1)
E_xy_dic = E_xy_dic.to_numpy()
# E_xy_dic = E_xy_dic.reshape(-1,1)

rows_in_range_y = []
rows_in_range_x = []
rows_in_range_exx = []
rows_in_range_eyy = []
rows_in_range_exy = []

for row_y, row_x, row_exx, row_eyy, row_exy in zip(Y_dic, X_dic, E_xx_dic, E_yy_dic, E_xy_dic):
    for value in row_y:
        if 25 <= value <= 85:
            rows_in_range_y.append(row_y)
            rows_in_range_x.append(row_x)
            rows_in_range_exx.append(row_exx)
            rows_in_range_eyy.append(row_eyy)
            rows_in_range_exy.append(row_exy)
            break
Y_dic = np.array(rows_in_range_y)
X_dic = np.array(rows_in_range_x)
E_xx_dic = np.array(rows_in_range_exx)
E_yy_dic = np.array(rows_in_range_eyy)
E_xy_dic = np.array(rows_in_range_exy)

# Y_dic = np.array(rows_in_range_y)[::20, ::5]
# X_dic = np.array(rows_in_range_x)[::20, ::5]
# E_xx_dic = np.array(rows_in_range_exx)[::20, ::5]
# E_yy_dic = np.array(rows_in_range_eyy)[::20, ::5]
# E_xy_dic = np.array(rows_in_range_exy)[::20, ::5]

# Y_dic = Y_dic[::10, ::10]
# X_dic = X_dic[::10, ::10]
# E_xx_dic = E_xx_dic[::10, ::10]
# E_yy_dic = E_yy_dic[::10, ::10]
# E_xy_dic = E_xy_dic[::10, ::10]


print(Y_dic.shape)

E_xx_dic = E_xx_dic.reshape(-1,1)
E_yy_dic = E_yy_dic.reshape(-1,1)
E_xy_dic = E_xy_dic.reshape(-1,1)

x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)
X_DIC_input = [x_values, y_values]
DIC_data = np.hstack([E_xx_dic, E_yy_dic, E_xy_dic])

"""
ARTIFICIAL DATA
"""

# # Load data
# dir_path = os.path.dirname(os.path.realpath(__file__))
# fem_file = os.path.join(dir_path, r"data_fem", 'fem_solution_dogbone_experiments.dat')
# data = np.loadtxt(fem_file)
# X_val      = data[:, :2]
# u_val      = data[:, 2:4]
# strain_val = data[:, 4:7]
# stress_val = data[:, 7:10]
# solution_val = np.hstack((u_val, stress_val))

# n_mesh_points = [total_points_hor,total_points_vert]
# x_grid = np.linspace(0, x_max_FEM, n_mesh_points[0])
# y_grid = np.linspace(0, y_max_FEM, n_mesh_points[1])

# def create_interpolation_fn(data_array):
#     num_components = data_array.shape[1]
#     interpolators = []
#     for i in range(num_components):
#         interp = RegularGridInterpolator(
#             (x_grid, y_grid),
#             data_array[:, i].reshape(n_mesh_points[1], n_mesh_points[0]).T,
#         )
#         interpolators.append(interp)
#     def interpolation_fn(x):
#         x_in = transform_coords([x[0], x[1]])
#         return np.array([interp((x_in[:, 0], x_in[:, 1])) for interp in interpolators]).T
#     return interpolation_fn

# solution_fn = create_interpolation_fn(solution_val)
# strain_fn   = create_interpolation_fn(strain_val)


# # Create simulated data
# X_DIC_input = [np.linspace(offs_x, offs_x + x_max_ROI, n_DIC).reshape(-1, 1),
#                np.linspace(offs_y, offs_y + y_max_ROI, n_DIC).reshape(-1, 1)]
# X_DIC_input_ref = [np.linspace(0, 20, n_DIC).reshape(-1, 1),
#                np.linspace(offs_y, offs_y + y_max_ROI, n_DIC).reshape(-1, 1)]
# DIC_solution = solution_fn(X_DIC_input_ref)
# DIC_data = strain_fn(X_DIC_input_ref)[:, :3]
# DIC_data += np.random.normal(0, noise_DIC, DIC_data.shape)


"""
VFM
"""


# Generate output matrices
X1, X2 = np.meshgrid(X_DIC_input[0], X_DIC_input[1])
Eps1= DIC_data[:,0]
Eps2 = DIC_data[:,1]
Eps6 = DIC_data[:,2]
Eps1 = Eps1.reshape(X1.shape)
Eps2 = Eps2.reshape(X1.shape)
Eps6 = Eps6.reshape(X1.shape) # *2? engineering strain

# #Simulated data
# X1 -= 5
# X2 -= 25


# Constants
# #Simumated data
# F = 360 #MUST CORRESPOND TO PSTRESS FOR SIMULATED DATA!!!!! CHECK FEM FILE

#RW data
#F = 375
F = 190


#DIC
# F = 450 #image 25

t = 2
w = 10
h = 60
Sd = h*w

print("np.mean(Eps1)")
print(np.mean(Eps1))
print("np.mean(Eps2)")
print(np.mean(Eps2))
print("np.mean(Eps6)")
print(np.mean(Eps6))
print(X1[:,-1])
print(X2[0,:])

"""CALCULATION"""
# Calculation of the components of matrix A


A = np.zeros((2, 2))

#Field 1.2: u2 = x2
A[0, 0] = np.mean(Eps2) * Sd
A[0, 1] = np.mean(Eps1) * Sd

#Field 1.1: u1 = x2(x2 - h)x1
# A[1, 0] = (np.mean( ( Eps1 * X2 * (X2-h) )) * Sd) + (np.mean( Eps6 * ( ( 2 * X2 ) - h ) * X1) * Sd)
# A[1, 1] = (np.mean( ( Eps2 * X2 * (X2-h) ) ) * Sd) - (np.mean( Eps6 * ( ( 2 * X2 ) - h ) * X1 ) * Sd)

A[1, 0] = np.mean(Eps1)
A[1, 1] = np.mean(Eps2)

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

rel_err_E = np.abs(E_actual-E)*100/E
print(f'rel error (E): {rel_err_E:.6f} %')
rel_err_nu = np.abs(nu_actual-Nu)*100/Nu
print(f'rel error (Nu): {rel_err_nu:.6f} %')

