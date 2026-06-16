#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main                        ### partition name where to run a job. Use 'main' unless qos is required. qos partitions 'rtx3090' 'rtx2080' 'gtx1080'
#SBATCH --time 3-23:59:59                      ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name best3_r0.5_run1             ### name of the job
#SBATCH --output best3_r0.5_run1-%J.out        ### output log for running job - %J is the job number variable
#SBATCH --mail-user=smorodin@post.bgu.ac.il    ### user's email for sending job status notifications
#SBATCH --mail-type=END,FAIL                   ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1                               ### number of GPUs
#SBATCH --mem=8G                               ### amount of RAM memory
#SBATCH --tasks=1                              # 1 process
### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####
module load anaconda
source activate privacyrobustness
CUDA_VISIBLE_DEVICES=0
# Rank 1 — genial-sweep-2 (id=5gb3hc1i) — LPIPS=0.1522
python Main.py \
  --cuda true --seed 1 --precision single --run_mode reconstruct \
  --wandb_active true --wandb_project_name Dataset_Extraction \
  --save_args_files true --train_save_model false \
  --extract_save_results true --extract_save_results_every 2500 \
  --problem celeba_male_female --data_per_class_train 250 --data_reduce_mean true \
  --pretrained_model_path weights-celeba_male_female_d250_celeba_male_female_radius_0.5.pth \
  --model_type mlp --model_hidden_list '[1000, 1000]' --model_use_bias true \
  --extraction_model_activation modifiedrelu --extraction_loss_type kkt \
  --extraction_random_init true --extraction_data_amount_per_class 500 \
  --extraction_epochs 50000 --extraction_evaluate_rate 1000 --extraction_stop_threshold 5000 \
  --extraction_lr 3.934550808571756 \
  --extraction_alpha_kkt 1.6109041976929892 \
  --extraction_lambda_lr 0.0002635714471575537 \
  --extraction_init_scale 0.005957635121834364 \
  --extraction_min_lambda 0.058815760533135066 \
  --extraction_alpha_prior 1.9295784490001928 \
  --extraction_model_relu_alpha 83.39242852877892 \
  --extraction_alpha_positive_lambdas 1.505511708250798
