"""
We replicate the side loaded plate example from Martin et al. (2019)
The data used for training is generated using the FEM solution of the plate, and can either be displacement, strain or DIC data (processed from FEM deformed images using MatchID FEDEF software)
"""
import deepxde as dde
import numpy as np
import jax.numpy as jnp
import jax
import time
import os
from scipy.interpolate import RegularGridInterpolator
import argparse

parser = argparse.ArgumentParser(description='Physics Informed Neural Networks for Linear Elastic Plate')

# Training parameters
parser.add_argument('--n_iter', type=int, default=int(1e10), help='Number of iterations')
parser.add_argument('--log_every', type=int, default=1000, help='Log every n steps')
parser.add_argument('--available_time', type=int, default=2, help='Available time in minutes')
parser.add_argument('--log_output_fields', nargs='+', default=['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy'], help='Fields to log')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1,1,1,1e8,1e8], help='Loss weights (more on DIC points)')
parser.add_argument('--num_point_PDE', type=int, default=10000, help='Number of collocation points for the PDE evaluation')

# Network parameters
parser.add_argument('--net_width', type=int, default=32, help='Width of the network')
parser.add_argument('--net_depth', type=int, default=5, help='Depth of the network')
parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='tanh', help='Activation function')
parser.add_argument('--optimizer', choices=['adam'], default='adam', help='Optimizer')
parser.add_argument('--mlp', choices=['mlp', 'modified_mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--initialization', choices=['Glorot uniform', 'He normal'], default='Glorot uniform', help='Initialization method')

# Inverse problem parameters
parser.add_argument('--measurments_type', choices=['displacement','strain','DIC'], default='DIC', help='Type of measurments (simulated DIC from FEM deformed image)')
parser.add_argument('--n_measurments', type=int, default=16, help='Number of measurments, should be a square number (=50*48 for the DIC data)') # =
parser.add_argument('--noise_magnitude', type=float, default=0, help='Gaussian noise magnitude added to the displacement/strain or equal to 0 or 1.5% of dynamic range for DIC images') 
parser.add_argument('--u_0', type=float, default=1e-4, help='Displacement scaling factor')
parser.add_argument('--params_iter_speed', nargs='+', type=float, default=[1,1], help='Scale iteration step for each parameter')

args = parser.parse_args()

n_iter = args.n_iter
log_every = args.log_every
available_time = args.available_time
log_output_fields = {i: field for i, field in enumerate(args.log_output_fields)}
mlp = args.mlp
n_measurments = int(np.sqrt(args.n_measurments))**2 # ensure it is a square number
if n_measurments != args.n_measurments:
    print(f"Warning: provided number of measurements ({args.n_measurments}) is not a perfect square. Using {n_measurments} points instead.")
noise_magnitude = args.noise_magnitude
lr = args.lr
u_0 = args.u_0
loss_weights = args.loss_weights
params_iter_speed = args.params_iter_speed
measurments_type = args.measurments_type
if measurments_type == "strain":
    loss_weights.append(loss_weights[-1])
    
dde.config.set_default_autodiff("forward")

L_max = 3.0
E_actual = 210e3  # Young's modulus
nu_actual = 0.3  # Poisson's ratio

E_init = 100e3 #100e3  # Initial guess for Young's modulus
nu_init = 0.2  # Initial guess for Poisson's ratio

params_factor = [dde.Variable(1/scale) for scale in params_iter_speed]
trainable_variables = params_factor

# Load
m = 10
b = 50
def side_load(y):
    return m * y + b

sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

geom = dde.geometry.Rectangle([0, 0], [L_max, L_max])

def HardBC(x, f, x_max=L_max):
    if isinstance(x, list):
        """For SPINN, the input x is a list of 1D arrays (X_coords, Y_coords)
        that need to be converted to a 2D meshgrid of same shape as the output f"""
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
        x = stack(x_mesh, axis=-1)

    Ux = f[:, 0] * x[:, 0]*u_0 
    Uy = f[:, 1] * x[:, 1]*u_0

    Sxx = f[:, 2] * (x_max - x[:, 0]) + side_load(x[:, 1])
    Syy = f[:, 3] * (x_max - x[:, 1])
    Sxy = f[:, 4] * x[:, 0]*(x_max - x[:, 0])*x[:, 1]*(x_max - x[:, 1])
    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

def jacobian(f, x, i, j):
    return dde.grad.jacobian(f, x, i=i, j=j)[0]  # second element is the function used by jax to compute the gradients

def pde(x, f, unknowns=params_factor):
    """For SPINN, the input x is a list of 1D arrays (X_coords, Y_coords)
    that need to be converted to a 2D meshgrid of same shape as the output f"""
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
    x = stack(x_mesh, axis=-1)

    param_factors = [unknown*scale for unknown, scale in zip(unknowns, params_iter_speed)]

    E, nu = E_init*param_factors[0], nu_init*param_factors[1]
    lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
    mu = E / (2 * (1 + nu))  # Lame's second parameter

    E_xx = jacobian(f, x, i=0, j=0) 
    E_yy = jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jacobian(f, x, i=2, j=0)
    Syy_y = jacobian(f, x, i=3, j=1)
    Sxy_x = jacobian(f, x, i=4, j=0)
    Sxy_y = jacobian(f, x, i=4, j=1)

    momentum_x = Sxx_x + Sxy_y 
    momentum_y = Sxy_x + Syy_y 

    f = f[0]  # f[1] is the function used by jax to compute the gradients

    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

# Load FEM reference solution
dir_path = os.path.dirname(os.path.realpath(__file__))
data = np.loadtxt(os.path.join(dir_path, r"fem_data/fem_solution_200_points.dat"))
X_val = data[:, :2]
u_val = data[:, 2:4]
strain_val = data[:, 4:7]
stress_val = data[:, 7:10]

solution_val = np.hstack((u_val, stress_val))

n_mesh_points = int(np.sqrt(X_val.shape[0]))

# Interpolate solution
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
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
        x_in = stack(x_mesh, axis=-1)
        return np.array([interp((x_in[:, 0], x_in[:, 1])) for interp in interpolators]).T

    return interpolation_fn

solution_fn = create_interpolation_fn(solution_val)
strain_fn = create_interpolation_fn(strain_val)

X_DIC_input = [np.linspace(0, L_max, int(np.sqrt(n_measurments))).reshape(-1, 1)]*2

if measurments_type == "displacement":
    DIC_data = solution_fn(X_DIC_input)[:,:2]
    DIC_data += np.random.normal(0, noise_magnitude, DIC_data.shape)

    measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1], lambda x, f, x_np: f[0][:, 0:1])
    measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2], lambda x, f, x_np: f[0][:, 1:2])
    bcs = [measure_Ux, measure_Uy]

elif measurments_type == "strain":
    DIC_data = strain_fn(X_DIC_input)
    DIC_data += np.random.normal(0, noise_magnitude, DIC_data.shape)
    def strain_from_output(x,f):
        """For SPINN, the input x is a list of 1D arrays (X_coords, Y_coords)
        that need to be converted to a 2D meshgrid of same shape as the output f"""
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
        x = stack(x_mesh, axis=-1)

        E_xx = jacobian(f, x, i=0, j=0) 
        E_yy = jacobian(f, x, i=1, j=1)
        E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))
        return jnp.hstack([E_xx, E_yy, E_xy])
        
    measure_Exx = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1], lambda x, f, x_np: strain_from_output(x,f)[:, 0:1])
    measure_Eyy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2], lambda x, f, x_np: strain_from_output(x,f)[:, 1:2])
    measure_Exy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 2:3], lambda x, f, x_np: strain_from_output(x,f)[:, 2:3])
    bcs = [measure_Exx, measure_Eyy, measure_Exy]

elif measurments_type == "DIC":

    import pandas as pd
    dic_dataset = "200fem_0_noise" if noise_magnitude == 0 else "200fem_1_5_noise" 
    dic_path = os.path.join(dir_path, f"dic_data/{dic_dataset}")

    X_dic = pd.read_csv(f"{dic_path}/X_trans/pattern_2MP_Numerical_1_0.synthetic.tif_X_trans.csv", delimiter=";").dropna(axis=1).to_numpy() 
    Y_dic = pd.read_csv(f"{dic_path}/Y_trans/pattern_2MP_Numerical_1_0.synthetic.tif_Y_trans.csv", delimiter=";").dropna(axis=1).to_numpy() 
    Ux_dic = pd.read_csv(f"{dic_path}/U_trans/pattern_2MP_Numerical_1_0.synthetic.tif_U_trans.csv", delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1,1)
    Uy_dic = pd.read_csv(f"{dic_path}/V_trans/pattern_2MP_Numerical_1_0.synthetic.tif_V_trans.csv", delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1,1)

    x_values = np.mean(X_dic, axis=0).reshape(-1,1)
    y_values = np.mean(Y_dic, axis=1).reshape(-1,1)
    X_DIC_input = [x_values, y_values]
    n_measurments = x_values.shape[0] * y_values.shape[0] # number of DIC points = 50*48

    measure_Ux = dde.PointSetOperatorBC(X_DIC_input, Ux_dic, lambda x, f, x_np: f[0][:, 0:1])
    measure_Uy = dde.PointSetOperatorBC(X_DIC_input, Uy_dic, lambda x, f, x_np: f[0][:, 1:2])
    bcs = [measure_Ux, measure_Uy]


activation = args.activation
initializer = args.initialization
optimizer = args.optimizer
layers = [2] + [args.net_width] * args.net_depth + [5]
net = dde.nn.SPINN(layers, activation, initializer, mlp)
num_point_PDE = args.num_point_PDE
batch_size = num_point_PDE + n_measurments 
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

net.apply_output_transform(HardBC) # Apply the hard boundary condition

dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = os.path.join(dir_path, "results_inverse", measurments_type)
folder_name = f"E-{E_init}_nu-{nu_init}_nDIC-{n_measurments}_noise-{noise_magnitude}_{available_time if available_time else n_iter}{'min' if available_time else 'iter'}"

# Check if any folders with the same name exist
existing_folders = [f for f in os.listdir(results_path) if f.startswith(folder_name)]

# If there are existing folders, find the highest number suffix
if existing_folders:
    suffixes = [int(f.split("-")[-1]) for f in existing_folders if f != folder_name]
    if suffixes:
        max_suffix = max(suffixes)
        folder_name = f"{folder_name}-{max_suffix + 1}"
    else:
        folder_name = f"{folder_name}-1"

# Create the new folder
new_folder_path = os.path.join(results_path, folder_name)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)


callbacks = [dde.callbacks.Timer(available_time)] if available_time else []
callbacks.append(dde.callbacks.VariableValue(params_factor, period=log_every, filename=os.path.join(new_folder_path, "variables_history.dat"), precision=10))

X_plot = [np.linspace(0, L_max, 100).reshape(-1,1)] * 2
for i, field in log_output_fields.items():
    callbacks.append(dde.callbacks.OperatorPredictor(X_plot, lambda x, output, i=i: output[0][:, i], period=log_every, filename=os.path.join(new_folder_path, f"{field}_history.dat")))

model = dde.Model(data, net)
model.compile(optimizer, lr=lr, metrics=["l2 relative error"], loss_weights=loss_weights, external_trainable_variables=trainable_variables)

start_time = time.time()
print(f"E: {E_init*params_factor[0].value*params_iter_speed[0]:.3f}, nu: {nu_init*params_factor[1].value*params_iter_speed[1]:.3f}")
losshistory, train_state = model.train(
    iterations=n_iter, callbacks=callbacks, display_every=log_every
)
print(f"E: {E_init*params_factor[0].value*params_iter_speed[0]:.3f}, nu: {nu_init*params_factor[1].value*params_iter_speed[1]:.3f}")

elapsed = time.time() - start_time

def log_config(fname):
    import json
    import platform
    import subprocess
    system_info = {
        "OS": platform.system(),
        "Release": platform.release(),
    }
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], 
                            capture_output=True, text=True, check=True)
        gpu_name, total_memory_mb = output.stdout.strip().split(", ")
        total_memory_gb = round(float(total_memory_mb.split(' ')[0])/1024) # Convert MB to GB

        gpu_info = {
            "GPU": gpu_name,
            "Total GPU Memory": f"{total_memory_gb:.2f} GB",
        }
    except subprocess.CalledProcessError:
        gpu_info = {
            "GPU": "No GPU found",
            "Total GPU Memory": "N/A",
        }

    execution_info = {
        "n_iter": train_state.epoch,
        "elapsed": elapsed,
        "iter_per_sec": train_state.epoch / elapsed,
        "backend": dde.backend.backend_name,
        "batch_size": batch_size,
        "num_params": num_params,
        "activation": activation,
        "initializer": initializer,
        "optimizer": optimizer,
        "mlp": mlp,
        "logged_fields": log_output_fields,
        "lr": lr,
        "E_actual": E_actual,
        "nu_actual": nu_actual,
        "E_init": E_init,
        "nu_init": nu_init,
        "E_final": float(E_init*params_factor[0].value*params_iter_speed[0]),
        "nu_final": float(nu_init*params_factor[1].value*params_iter_speed[1]),
        "params_iter_speed": params_iter_speed,
        "loss_weights": loss_weights,
        "L_max": L_max,
        "u_0": u_0,
        "n_measurments": (int(np.sqrt(n_measurments)))**2, 
        "noise_magnitude": noise_magnitude,
        "measurments_type": measurments_type,
    }

    info = {**system_info, **gpu_info, **execution_info}
    info_json = json.dumps(info, indent=4)

    with open(fname, "w") as f:
        f.write(info_json)


log_config(os.path.join(new_folder_path, "config.json"))
dde.utils.save_loss_history(
    losshistory, os.path.join(new_folder_path, "loss_history.dat")
)

#correct saved variable values with the training factor
params_init = [E_init, nu_init]
with open(os.path.join(new_folder_path, "variables_history.dat"), "r") as f:
    lines = f.readlines()
with open(os.path.join(new_folder_path, "variables_history.dat"), "w") as f:
    for line in lines:
        step, value = line.strip().split(' ', 1)
        values = [scale_i*iter_speed_i*value_i for scale_i, iter_speed_i, value_i in zip(params_iter_speed, params_init, eval(value))]
        f.write(f"{step} "+dde.utils.list_to_str(values, precision=3)+"\n")