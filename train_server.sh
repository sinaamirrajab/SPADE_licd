#!/bin/bash                                                                                                                                                                                     
#SBATCH --partition=bme.gpuresearch.q      		# Partition name
#SBATCH --nodelist=bme-gpuA001
#SBATCH --nodes=1                        		# Use one node                                                                                                                                             
#SBATCH --time=10:00:00                  		# Time limit hrs:min:sec
#SBATCH --output=./log/output_%A_Training.out         	# Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --job-name=score_recon


module load cuda11.8/toolkit
python ./train_lidc.py