#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main                        ### partition name where to run a job. Use 'main' unless qos is required. qos partitions 'rtx3090' 'rtx2080' 'gtx1080'
#SBATCH --time 3-23:59:59                      ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name best3_imagenet_r0_run2      ### name of the job
#SBATCH --output best3_imagenet_r0_run2-%J.out ### output log for running job - %J is the job number variable
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
# Rank 2 — quiet-sweep-6 (id=ovocjxxd) — LPIPS=0.3017
python Main.py \
  --cuda true --seed 1 --precision single --run_mode reconstruct \
  --wandb_active true --wandb_project_name Dataset_Extraction \
  --save_args_files true --train_save_model false \
  --extract_save_results true --extract_save_results_every 2500 \
  --problem imagenet --data_per_class_train 250 --data_reduce_mean true \
  --pretrained_model_path weights-imagenet_d250_imagenet_radius_0.pth \
  --model_type mlp --model_hidden_list '[1000, 1000]' --model_use_bias true \
  --extraction_model_activation modifiedrelu --extraction_loss_type kkt \
  --extraction_random_init true --extraction_data_amount_per_class 500 \
  --extraction_epochs 50000 --extraction_evaluate_rate 1000 --extraction_stop_threshold 5000 \
  --extraction_lr 1.4506091613937198 \
  --extraction_alpha_kkt 1.455044070263979 \
  --extraction_lambda_lr 0.00031910104233188064 \
  --extraction_init_scale 0.0032444495949496516 \
  --extraction_min_lambda 0.36552205171115393 \
  --extraction_alpha_prior 5.381258392521712 \
  --extraction_model_relu_alpha 36.69324487055219 \
  --extraction_alpha_positive_lambdas 0.28072594447081695
