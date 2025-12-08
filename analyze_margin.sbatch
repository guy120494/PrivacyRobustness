#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main                        ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 3-23:59:59                      ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name analyze_margin                   ### name of the job. replace my_job with your desired job name
#SBATCH --output analyze_margin-%J.out                ### output log for running job - %J is the job number variable
#SBATCH --mail-user=smorodin@post.bgu.ac.il      ### user’s email for sending job status notifications
#SBATCH --mail-type=FAIL             ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1  ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team’s permission
#SBATCH --mem=9G ### amount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --tasks=1  # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise
# ## Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####
module load anaconda              ### load anaconda module
source activate privacyrobustness            ### activate a conda environment, replace my_env with your conda environment
CUDA_VISIBLE_DEVICES=0
python margin_evaluations.py --train_file=/home/smorodin/robustnessPrivacy/PrivacyRobustness/datasets/cifar-10-batches-py/train_batch.pt --reconstruction_folder=/home/smorodin/robustnessPrivacy/PrivacyRobustness/results/weights-cifar10_vehicles_animals_d250_cifar10_vehicles_animals