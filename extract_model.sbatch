#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main                        ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 6-23:59:59                      ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name train_mnist_model                   ### name of the job. replace my_job with your desired job name
#SBATCH --output train_mnist_model-%J.out                ### output log for running job - %J is the job number variable
#SBATCH --mail-user=smorodin@post.bgu.ac.il      ### user’s email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL             ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1  ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team’s permission
#SBATCH --mem=8G ### amount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --tasks=1  # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise
# ## Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####
module load anaconda              ### load anaconda module
source activate robustnessPrivacy            ### activate a conda environment, replace my_env with your conda environment
CUDA_VISIBLE_DEVICES=0
python Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=50000 --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=weights-mnist_odd_even_d250_mnist_odd_even.pth --problem=mnist_odd_even --run_mode=reconstruct --wandb_active=True --wandb_login=4b2718fafa76b966ed5649355f33aa2aa9bcf65b