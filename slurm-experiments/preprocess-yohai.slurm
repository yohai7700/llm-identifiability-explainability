#! /bin/sh

#SBATCH --job-name=preprocess-yohai # job name
#SBATCH --output=/home/sharifm/teaching/tml-0368-4075/2024-spring/students/yohaimazuz/slurm-results/preprocess-yohai.out # redirect stdout
#SBATCH --error=/home/sharifm/teaching/tml-0368-4075/2024-spring/students/yohaimazuz/slurm-results/preprocess-yohai.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=60 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=100000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=8 # GPUs in total


python main.py --task=preprocess --cache_user=yohai