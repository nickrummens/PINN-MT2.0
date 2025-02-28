import os
import sys
import subprocess

executable_path = os.path.join(os.path.dirname(__file__), "../pinn_inverse.py")
num_runs = 1

args = {
    "FEM_dataset": "100x100mm.dat",
    "n_iter": 40000,
    "available_time": 0,
    "loss_weights": [1,1,1,1,1,1e2,1e2],
    "log_every": 250,
}

# Flatten the args dictionary into a list of command line arguments
def flatten_args(args):
    args_list = []
    for key, value in args.items():
        if isinstance(value, list):
            args_list.extend([f"--{key}"] + [str(v) for v in value])
        else:
            args_list.append(f"--{key}={value}")
    return args_list

# 5 runs for displacement and strain without noise
args["DIC_dataset_path"] = r"2_noise_study/data_dic/0noise"
args["DIC_dataset_number"] = 1 # 0 is reference image
args["results_path"] = r"2_noise_study/results/0noise"
for measurement_type in ["displacement"]:
    args["measurments_type"] = measurement_type
    args["n_iter"] = 80000 if measurement_type == "displacement" else 40000
    for run in range(5):
        if run == 0: # log all fields for the first run
            args["log_output_fields"] = ['Ux', 'Uy', 'Exx', 'Eyy', 'Exy', 'Sxx', 'Syy', 'Sxy']
        else:
            args["log_output_fields"] = ['']
        try:
            print(f"Run number {run+1}/5 for {measurement_type} without noise")
            subprocess.check_call([sys.executable, executable_path] + flatten_args(args))
        except subprocess.CalledProcessError as e:
            print(f"Run number {run+1}/5 failed")
            print(e)
            sys.exit(1)

# 10 runs for displacement and strain with noise
args["DIC_dataset_path"] = r"2_noise_study/data_dic/1noise"
args["results_path"] = r"2_noise_study/results/1noise"
for measurement_type in ["displacement", "strain"]:
    args["measurments_type"] = measurement_type
    args["n_iter"] = 80000 if measurement_type == "displacement" else 40000
    for dic_number in range(1, 11):
        if run == 0: # log all fields for the first run
            args["log_output_fields"] = ['Ux', 'Uy', 'Exx', 'Eyy', 'Exy', 'Sxx', 'Syy', 'Sxy']
        else:
            args["log_output_fields"] = ['']
        args["DIC_dataset_number"] = dic_number
        try:
            print(f"Run number {dic_number}/10 for {measurement_type} with noise")
            subprocess.check_call([sys.executable, executable_path] + flatten_args(args))
        except subprocess.CalledProcessError as e:
            print(f"Run number {run+1}/10 failed")
            print(e)
            sys.exit(1)