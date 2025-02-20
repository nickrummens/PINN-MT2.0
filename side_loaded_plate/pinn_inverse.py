"""
PINN-MT2.0: Inverse identification for the side loaded plate example

This script replicates the example from Martin et al. (2019) using a FEM reference
solution and a Physics-Informed Neural Network (PINN) to identify material properties.
"""

import os
import time
import json
import argparse
import platform
import subprocess

import numpy as np
import jax
import jax.numpy as jnp
import deepxde as dde
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# 1. Utility Function: Coordinate Transformation for SPINN
# =============================================================================
def transform_coords(x):
    """
    For SPINN, if the input x is provided as a list of 1D arrays (e.g., [X_coords, Y_coords]),
    this function creates a 2D meshgrid and stacks the results into a 2D coordinate array.
    """
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
    return dde.backend.stack(x_mesh, axis=-1)

# =============================================================================
# 2. Parse Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Physics Informed Neural Networks for Linear Elastic Plate")
parser.add_argument('--n_iter', type=int, default=int(1e10), help='Number of iterations')
parser.add_argument('--log_every', type=int, default=1000, help='Log every n steps')
parser.add_argument('--available_time', type=int, default=2, help='Available time in minutes')
parser.add_argument('--log_output_fields', nargs='*', default=['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy'], help='Fields to log')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_fn', nargs='+', default=['MSE']*5+["mean l2 relative error"], help='Loss functions')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1,1,1,1e3,1e3], help='Loss weights (more on DIC points)')
parser.add_argument('--num_point_PDE', type=int, default=10000, help='Number of collocation points for PDE evaluation')
parser.add_argument('--num_point_test', type=int, default=100000, help='Number of test points')

parser.add_argument('--net_width', type=int, default=32, help='Width of the network')
parser.add_argument('--net_depth', type=int, default=5, help='Depth of the network')
parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='tanh', help='Activation function')
parser.add_argument('--optimizer', choices=['adam'], default='adam', help='Optimizer')
parser.add_argument('--mlp', choices=['mlp', 'modified_mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--initialization', choices=['Glorot uniform', 'He normal'], default='Glorot uniform', help='Initialization method')

parser.add_argument('--measurments_type', choices=['displacement','strain','DIC'], default='strain', help='Type of measurements')
parser.add_argument('--n_measurments', type=int, default=16, help='Number of measurements (should be a perfect square)')
parser.add_argument('--noise_magnitude', type=float, default=1e-6, help='Gaussian noise magnitude')
parser.add_argument('--u_0', nargs='+', type=float, default=[0,0], help='Displacement scaling factor for Ux and Uy, default(=0) use measurements norm')
parser.add_argument('--params_iter_speed', nargs='+', type=float, default=[1,1], help='Scale iteration step for each parameter')

parser.add_argument('--FEM_dataset', type=str, default='3mm_200points.dat', help='Path to FEM data')
parser.add_argument('--DIC_dataset', choices=['3mm_0noise', '3mm_1_5noise'], default='3mm_0noise', help='Only for DIC measurements')
parser.add_argument('--results_path', type=str, default='results_inverse', help='Path to save results')

args = parser.parse_args()

if len(args.log_output_fields[0]) == 0:
    args.log_output_fields = [] # Empty list for no logging

# For strain measurements, extend loss weights
if args.measurments_type == "strain":
    args.loss_fn.append(args.loss_fn[-1])
    args.loss_weights.append(args.loss_weights[-1])

dde.config.set_default_autodiff("forward")

# =============================================================================
# 3. Global Constants, Geometry, and Material Parameters
# =============================================================================
L_max     = 3.0
E_actual  = 210e3   # Actual Young's modulus
nu_actual = 0.3     # Actual Poisson's ratio
E_init    = 100e3   # Initial guess for Young's modulus
nu_init   = 0.2     # Initial guess for Poisson's ratio
m, b = 10, 50  # Side-load parameters

def side_load(y):
    return m * y + b

# Create trainable scaling factors (one per parameter)
params_factor = [dde.Variable(1 / s) for s in args.params_iter_speed]
trainable_variables = params_factor

# =============================================================================
# 4. Load FEM Data and Build Interpolation Functions
# =============================================================================
dir_path = os.path.dirname(os.path.realpath(__file__))
fem_file = os.path.join(dir_path, r"fem_data", args.FEM_dataset)
data = np.loadtxt(fem_file)
X_val      = data[:, :2]
u_val      = data[:, 2:4]
strain_val = data[:, 4:7]
stress_val = data[:, 7:10]
solution_val = np.hstack((u_val, stress_val))

n_mesh_points = int(np.sqrt(X_val.shape[0]))
x_grid = np.linspace(0, L_max, n_mesh_points)
y_grid = np.linspace(0, L_max, n_mesh_points)

def create_interpolation_fn(data_array):
    num_components = data_array.shape[1]
    interpolators = []
    for i in range(num_components):
        interp = RegularGridInterpolator(
            (x_grid, y_grid),
            data_array[:, i].reshape(n_mesh_points, n_mesh_points).T
        )
        interpolators.append(interp)
    def interpolation_fn(x):
        x_in = transform_coords([x[0], x[1]])
        return np.array([interp((x_in[:, 0], x_in[:, 1])) for interp in interpolators]).T
    return interpolation_fn

solution_fn = create_interpolation_fn(solution_val)
strain_fn   = create_interpolation_fn(strain_val)

# =============================================================================
# 5. Setup Measurement Data Based on Type (Displacement, Strain, DIC)
# =============================================================================
args.n_measurments = int(np.sqrt(args.n_measurments))**2
if args.measurments_type == "displacement":    
    X_DIC_input = [np.linspace(0, L_max, args.n_measurments).reshape(-1, 1)] * 2
    DIC_data = solution_fn(X_DIC_input)[:, :2]
    DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)
    measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1],
                                          lambda x, f, x_np: f[0][:, 0:1])
    measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2],
                                          lambda x, f, x_np: f[0][:, 1:2])
    bcs = [measure_Ux, measure_Uy]

elif args.measurments_type == "strain":
    def strain_from_output(x, f):
        """
        Compute strain components from the network output for strain measurements.
        """
        x = transform_coords(x)
        E_xx = dde.grad.jacobian(f, x, i=0, j=0)[0]
        E_yy = dde.grad.jacobian(f, x, i=1, j=1)[0]
        E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1)[0] + dde.grad.jacobian(f, x, i=1, j=0)[0])
        return jnp.hstack([E_xx, E_yy, E_xy])

    X_DIC_input = [np.linspace(0, L_max, args.n_measurments).reshape(-1, 1)] * 2
    DIC_data = strain_fn(X_DIC_input)
    DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)
    measure_Exx = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 0:1])
    measure_Eyy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 1:2])
    measure_Exy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 2:3],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 2:3])
    bcs = [measure_Exx, measure_Eyy, measure_Exy]

elif args.measurments_type == "DIC":
    import pandas as pd
    dic_path = os.path.join(dir_path, f"dic_data/{args.DIC_dataset}")
    X_dic = pd.read_csv(os.path.join(dic_path, "X_trans", "pattern_2MP_Numerical_1_0.synthetic.tif_X_trans.csv"),
                        delimiter=";").dropna(axis=1).to_numpy()
    Y_dic = pd.read_csv(os.path.join(dic_path, "Y_trans", "pattern_2MP_Numerical_1_0.synthetic.tif_Y_trans.csv"),
                        delimiter=";").dropna(axis=1).to_numpy()
    Ux_dic = pd.read_csv(os.path.join(dic_path, "U_trans", "pattern_2MP_Numerical_1_0.synthetic.tif_U_trans.csv"),
                         delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
    Uy_dic = pd.read_csv(os.path.join(dic_path, "V_trans", "pattern_2MP_Numerical_1_0.synthetic.tif_V_trans.csv"),
                         delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
    x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
    y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)
    X_DIC_input = [x_values, y_values]
    
    if args.n_measurments != x_values.shape[0] * y_values.shape[0]:
        print(f"For this DIC dataset, the number of measurements is fixed to {x_values.shape[0] * y_values.shape[0]}")
    args.n_measurments = x_values.shape[0] * y_values.shape[0]

    measure_Ux = dde.PointSetOperatorBC(X_DIC_input, Ux_dic,
                                          lambda x, f, x_np: f[0][:, 0:1])
    measure_Uy = dde.PointSetOperatorBC(X_DIC_input, Uy_dic,
                                          lambda x, f, x_np: f[0][:, 1:2])
    bcs = [measure_Ux, measure_Uy]

# Use measurements norm as the default scaling factor
if args.measurments_type == "DIC":
    DIC_norms = np.linalg.norm(np.hstack([Ux_dic, Uy_dic]), axis=0)
else:
    DIC_norms = np.linalg.norm(solution_fn(X_DIC_input)[:, :2], axis=0)
args.u_0 = [DIC_norms[i] if not args.u_0[i] else args.u_0[i] for i in range(2)]

# =============================================================================
# 6. PINN Implementation: Boundary Conditions and PDE Residual
# =============================================================================
# Define the domain geometry
geom = dde.geometry.Rectangle([0, 0], [L_max, L_max])

def HardBC(x, f, x_max=L_max):
    """
    Apply hard boundary conditions via transformation.
    If x is provided as a list of 1D arrays, transform it to a 2D meshgrid.
    """
    if isinstance(x, list):
        x = transform_coords(x)
    Ux  = f[:, 0] * x[:, 0] * args.u_0[0] 
    Uy  = f[:, 1] * x[:, 1] * args.u_0[1]
    Sxx = f[:, 2] * (x_max - x[:, 0]) + side_load(x[:, 1])
    Syy = f[:, 3] * (x_max - x[:, 1])
    Sxy = f[:, 4] * x[:, 0] * (x_max - x[:, 0]) * x[:, 1] * (x_max - x[:, 1])
    return dde.backend.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

def pde(x, f, unknowns=params_factor):
    """
    Define the PDE residuals for the linear elastic plate.
    """
    x = transform_coords(x)
    param_factors = [u * s for u, s in zip(unknowns, args.params_iter_speed)]
    E = E_init * param_factors[0]
    nu = nu_init * param_factors[1]
    lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)[0]
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)[0]
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1)[0] + dde.grad.jacobian(f, x, i=1, j=0)[0])
    
    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)[0]
    Syy_y = dde.grad.jacobian(f, x, i=3, j=1)[0]
    Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)[0]
    Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)[0]
    
    momentum_x = Sxx_x + Sxy_y
    momentum_y = Sxy_x + Syy_y
    
    f_internal = f[0]
    stress_x  = S_xx - f_internal[:, 2:3]
    stress_y  = S_yy - f_internal[:, 3:4]
    stress_xy = S_xy - f_internal[:, 4:5]
    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

# =============================================================================
# 7. Define Neural Network, Data, and Model
# =============================================================================
layers = [2] + [args.net_width] * args.net_depth + [5]
net = dde.nn.SPINN(layers, args.activation, args.initialization, args.mlp)
num_point_PDE = args.num_point_PDE
batch_size = num_point_PDE + args.n_measurments
num_params = sum(p.size for p in jax.tree.leaves(net.init(jax.random.PRNGKey(0), jnp.ones(layers[0]))))
num_test = 100000

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=num_point_PDE,
    solution=solution_fn,
    num_test=num_test,
    is_SPINN=True,
)
net.apply_output_transform(HardBC)

model = dde.Model(data, net)
model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
              loss_weights=args.loss_weights, external_trainable_variables=trainable_variables)

# =============================================================================
# 8. Setup Callbacks for Logging
# =============================================================================
results_path = os.path.join(dir_path, args.results_path)
folder_name = f"{args.measurments_type}_x{args.n_measurments}_{args.noise_magnitude}noise_{args.available_time if args.available_time else args.n_iter}{'min' if args.available_time else 'iter'}"
existing_folders = [f for f in os.listdir(results_path) if f.startswith(folder_name)]
if existing_folders:
    suffixes = [int(f.split("-")[-1]) for f in existing_folders if f != folder_name]
    folder_name = f"{folder_name}-{max(suffixes)+1}" if suffixes else f"{folder_name}-1"
new_folder_path = os.path.join(results_path, folder_name)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

callbacks = []
if args.available_time:
    callbacks.append(dde.callbacks.Timer(args.available_time))
callbacks.append(dde.callbacks.VariableValue(params_factor, period=args.log_every,
                                               filename=os.path.join(new_folder_path, "variables_history.dat"),
                                               precision=8))
X_plot = [np.linspace(0, L_max, 100).reshape(-1, 1)] * 2
for i, field in enumerate(args.log_output_fields): # Log output fields
    callbacks.append(
        dde.callbacks.OperatorPredictor(
            X_plot,
            lambda x, output, i=i: output[0][:, i],
            period=args.log_every,
            filename=os.path.join(new_folder_path, f"{field}_history.dat"),
            precision=6
        )
    )

# =============================================================================
# 9. Training
# =============================================================================
start_time = time.time()
print(f"E: {E_init * params_factor[0].value * args.params_iter_speed[0]:.3f}, nu: {nu_init * params_factor[1].value * args.params_iter_speed[1]:.3f}")
losshistory, train_state = model.train(iterations=args.n_iter, callbacks=callbacks, display_every=args.log_every)
elapsed = time.time() - start_time

# =============================================================================
# 10. Logging
# =============================================================================
dde.utils.save_loss_history(losshistory, os.path.join(new_folder_path, "loss_history.dat"))

params_init = [E_init, nu_init]
variables_history_path = os.path.join(new_folder_path, "variables_history.dat")

# Read the variables history
with open(variables_history_path, "r") as f:
    lines = f.readlines()

# Update the variables history with scaled values
with open(variables_history_path, "w") as f:
    for line in lines:
        step, value = line.strip().split(' ', 1)
        values = [scale * init * val for scale, init, val in zip(args.params_iter_speed, params_init, eval(value))]
        f.write(f"{step} " + dde.utils.list_to_str(values, precision=8) + "\n")

# Read the variables history
with open(variables_history_path, "r") as f:
    lines = f.readlines()
# Final E and nu values as the average of the last 10 values 
E_final = np.mean([eval(line.strip().split(' ', 1)[1])[0] for line in lines[-10:]])
nu_final = np.mean([eval(line.strip().split(' ', 1)[1])[1] for line in lines[-10:]])
print(f"Final E: {E_final:.3f}, nu: {nu_final:.3f}")

def log_config(fname):
    """
    Save configuration and execution details to a JSON file, grouped by category.
    """
    system_info = {"OS": platform.system(), "Release": platform.release()}
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        gpu_name, total_memory_mb = output.stdout.strip().split(", ")
        total_memory_gb = round(float(total_memory_mb.split(' ')[0]) / 1024, 2)
        gpu_info = {"GPU": gpu_name, "Total GPU Memory": f"{total_memory_gb:.2f} GB"}
    except subprocess.CalledProcessError:
        gpu_info = {"GPU": "No GPU found", "Total GPU Memory": "N/A"}
    
    execution_info = {
        "n_iter": train_state.epoch,
        "elapsed": elapsed,
        "iter_per_sec": train_state.epoch / elapsed,
        "backend": dde.backend.backend_name,
    }
    network_info = {
        "net_width": args.net_width,
        "net_depth": args.net_depth,
        "num_params": num_params,
        "activation": args.activation,
        "mlp_type": args.mlp,
        "optimizer": args.optimizer,
        "initializer": args.initialization,
        "batch_size": batch_size,
        "lr": args.lr,
        "loss_weights": args.loss_weights,
        "params_iter_speed": args.params_iter_speed,
        "u_0": args.u_0,
        "logged_fields": args.log_output_fields,
    }
    problem_info = {
        "L_max": L_max,
        "E_actual": E_actual,
        "nu_actual": nu_actual,
        "E_init": E_init,
        "nu_init": nu_init,
        "E_final": E_final,
        "nu_final": nu_final,
    }
    data_info = {
        "n_measurments": (int(np.sqrt(args.n_measurments)))**2,
        "noise_magnitude": args.noise_magnitude,
        "measurments_type": args.measurments_type,
    }
    info = {"system": system_info, "gpu": gpu_info, "execution": execution_info,
            "network": network_info, "problem": problem_info, "data": data_info}
    with open(fname, "w") as f:
        json.dump(info, f, indent=4)

log_config(os.path.join(new_folder_path, "config.json"))