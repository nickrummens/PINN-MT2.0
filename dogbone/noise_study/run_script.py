import os
import sys
import subprocess

executable_path = os.path.join(os.path.dirname(__file__), "../dogbone_inverse.py")
num_runs = 1

args = {
    # "FEM_dataset": "fem_solution_dogbone_experiments.dat",
    # # "n_iter": 40000,
    # "available_time": 3,
    # # "loss_weights": [1,1,1,1,1,1,1,1],
    # "log_every": 250,
    # # "num_point_PDE": 100**2,
    # # "noise_magnitude": 1e-6
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

# 1 5min run for strain with noise
#args["DIC_dataset_path"] = f"2_noise_study/data_dic/{camera_resolution}/1noise"
#args["DIC_dataset_number"] = 1
# args["results_path"] = r"noise_study/results/artificial_displacement_1e-4_E200nu60"
# args["measurments_type"] = "strain"
# args["available_time"] = 3




# try:
#     print("Run number 1/1 for strain with noise")
#     subprocess.check_call([sys.executable, executable_path] + flatten_args(args))
# except subprocess.CalledProcessError as e:
#     print("Run number 1/1 failed")
#     print(e)
#     sys.exit(1)




#5 runs for displacement and strain without noise
#args["DIC_dataset_path"] = f"2_noise_study/data_dic/{camera_resolution}/0noise"
#args["DIC_dataset_number"] = 1 # 0 is reference image
args["results_path"] = r"noise_study/results/simulated_displacement_BFSU350S5MC_E200nu60"
# args["available_time"] = 3
for run in range(10):
    if run == 0: # log all fields for the first run
        args["log_output_fields"] = ['Ux', 'Uy', 'Exx', 'Eyy', 'Exy', 'Sxx', 'Syy', 'Sxy']
    else:
        args["log_output_fields"] = ['']
    try:
        print(f"Run number {run+1}/10 for strain with noise")
        subprocess.check_call([sys.executable, executable_path] + flatten_args(args))
    except subprocess.CalledProcessError as e:
        print(f"Run number {run+1}/5 failed")
        print(e)
        sys.exit(1)
