#!/bin/bash
cd ..

# wandb logger & workers
wandb_project='ddsp-piano'
num_workers=8
checkpoint_interval=2000

# paths (relative to this script directory)
WORKSPACE_DIR="../../202601_midisemi_data/ddsp-piano-pytorch/workspaces"
maestro_cache_path="$WORKSPACE_DIR/data_cache"
exp_dir="$WORKSPACE_DIR/models"

# phase1 training parameters
phase_1_batch_size=6
phase_1_n_epochs=7
phase_1_learning_rate=0.001
phase_1_wandb_run_name='phase1_gpu1_5090'

# phase2 training parameters
phase_2_batch_size=3
phase_2_n_epochs=3
phase_2_learning_rate=0.00001
phase_2_wandb_run_name='phase2_gpu1_5090'

# phase3 training parameters
phase_3_batch_size=6
phase_3_n_epochs=10
phase_3_learning_rate=0.001
phase_3_wandb_run_name='phase3_gpu1_5090'

python3 train.py \
	--batch_size $phase_1_batch_size \
	--epochs $phase_1_n_epochs \
	--lr $phase_1_learning_rate \
	--phase 1 \
	--save_interval $checkpoint_interval \
	--num_workers $num_workers \
	$maestro_cache_path \
	$exp_dir \
	--wandb_project "$wandb_project" \
	--wandb_run_name "$phase_1_wandb_run_name" \
	--debug_mode
	# --continue_from /media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-piano-pytorch/workspaces/models/rtx3090_phase1_gpu1_debug/phase_1/ckpts/ddsp-piano_epoch_2_params.pt 

python3 train.py \
	--batch_size $phase_2_batch_size \
	--epochs $phase_2_n_epochs \
	--lr $phase_2_learning_rate \
	--phase 2 \
	--save_interval $checkpoint_interval \
	--num_workers $num_workers \
	$maestro_cache_path \
	$exp_dir \
	--wandb_project "$wandb_project" \
	--wandb_run_name "$phase_2_wandb_run_name" \
	--debug_mode \
	--init_from /media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-piano-pytorch/workspaces/models/phase_1/ckpts/ddsp-piano_epoch_2_params.pt 

python3 train.py \
	--batch_size $phase_3_batch_size \
	--epochs $phase_3_n_epochs \
	--lr $phase_3_learning_rate \
	--phase 3 \
	--save_interval $checkpoint_interval \
	--num_workers $num_workers \
	$maestro_cache_path \
	$exp_dir \
	--wandb_project "$wandb_project" \
	--wandb_run_name "$phase_3_wandb_run_name" \
	--debug_mode \
	--init_from /media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-piano-pytorch/workspaces/models/phase_2/ckpts/ddsp-piano_epoch_2_params.pt 