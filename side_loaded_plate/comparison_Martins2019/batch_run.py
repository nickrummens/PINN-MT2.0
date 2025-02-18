import os
import sys
import subprocess

executable_path = os.path.join(os.path.dirname(__file__), "../pinn_inverse.py")
num_runs = 1

args = {
    "available_time": 5,
    "measurments_type": "strain",
    "noise_magnitude": 1e-6,
    "results_path": "comparison_Martins2019/results",
}
args_list = [f"--{key}={value}" for key, value in args.items()] 

for run in range(num_runs):
    try:
        print(f"Run number {run}/{num_runs}")
        subprocess.check_call([sys.executable, executable_path] + args_list)
    except subprocess.CalledProcessError as e:
        print(f"Run number {run}/{num_runs} failed")
        print(e)
        sys.exit(1)