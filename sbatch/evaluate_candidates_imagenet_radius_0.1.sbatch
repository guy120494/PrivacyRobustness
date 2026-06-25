#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main                                     ### partition name where to run a job. Use 'main' unless qos is required. qos partitions 'rtx3090' 'rtx2080' 'gtx1080'
#SBATCH --time 1-00:00:00                                    ### limit the time of job running. Format: D-H:MM:SS
#SBATCH --job-name eval_candidates_imagenet_r0.1             ### name of the job
#SBATCH --output eval_candidates_imagenet_r0.1-%J.out        ### output log for running job - %J is the job number variable
#SBATCH --mail-user=smorodin@post.bgu.ac.il                  ### user's email for sending job status notifications
#SBATCH --mail-type=END,FAIL                                 ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1                                             ### number of GPUs
#SBATCH --mem=16G                                            ### amount of RAM memory
#SBATCH --tasks=1                                            # 1 process
### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####
module load anaconda
source activate privacyrobustness
CUDA_VISIBLE_DEVICES=0
python evaluate_candidates.py \
  --folder weights-imagenet_d250_imagenet_radius_0.1 \
  --problem imagenet \
  --data_per_class_train 250 \
  --final_only false\
  --save_matrices
