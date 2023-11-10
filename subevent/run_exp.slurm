#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample4       #Set the job name to "JobExample4"
#SBATCH --time=07:30:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                   #Request 2560MB (2.5GB) per node
#SBATCH --output=Example4Out.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=122713724024             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 

#First Executable Line
source maven-ere/bin/activate
module load GCCcore/12.2.0
module load git/2.38.1-nodocs
module load git-lfs/2.11.0
cd /scratch/user/kangda/MAVEN-ERE/roberta-base
git lfs install
git lfs pull
cd /scratch/user/kangda/MAVEN-ERE/subevent
CUDA_VISIBLE_DEVICES=0 python -u main.py --epochs 20 --eval_steps 100 --log_steps 50