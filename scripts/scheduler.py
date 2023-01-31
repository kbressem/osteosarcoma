import argparse
import os
import time

parser = argparse.ArgumentParser(description="Schedule trainings with slurm")
parser.add_argument("--config", type=str, required=True, help="path to the config file")
parser.add_argument(
    "--train", action="store_true", required=False, help="Run simple training. Invokes train.py"
)
parser.add_argument(
    "--tune",
    action="store_true",
    required=False,
    help="Run a hyperparameter tuning. Invokes tune.py",
)
parser.add_argument(
    "--jobs", type=int, required=False, default=1, help="Number of jobs that should be scheduled"
)
args = parser.parse_args()

with open("scheduler.sh", "w+") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=sarcoma\n")  # Specify job name
    f.write("#SBATCH --partition=gpu\n")  # Specify partition name
    f.write("#SBATCH --mem=0\n")  # Use entire memory of node
    f.write("#SBATCH --gres=gpu:1\n")  # Generic resources; 1 GPU
    f.write("#SBATCH --exclusive\n")  # Do not share node
    f.write("#SBATCH --time=120:00:00\n")  # Set a limit on the total run time
    f.write("#SBATCH --output=logs_sarcoma22.o%j\n")  # File name for standard output
    f.write("#SBATCH --error=errors_sarcoma22.e%j\n")  # File name for standard error output
    f.write("\n\n")
    f.write("source /home/bressekk/miniconda3/etc/profile.d/conda.sh\n")
    f.write("conda activate monai\n")
    f.write("\n\n")
    f.write("cd /sc-scratch/sc-scratch-dha/bone-tumor\n")
    if args.train:
        f.write(f"python scripts/train.py --config {args.config}\n")
    elif args.tune:
        f.write(f"python scripts/tune.py --config {args.config}\n")
    else:
        print("Specify either --:train or --tune")

for _ in range(0, args.jobs):
    os.system("sbatch scheduler.sh")
    time.sleep(30)  # give trainlib some time to set up correct filepaths

os.remove("scheduler.sh")
