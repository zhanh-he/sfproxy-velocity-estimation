#!/bin/bash
#SBATCH --job-name=ddsp
#SBATCH --output=ddsp_progress_%j.log
#SBATCH --error=ddsp_error_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au


# Load any required modules (if needed) - module load cuda/11.8 gcc/9.4.0
module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
module list
source activate bark_env

echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
# echo "SLURM job ID: $SLURM_JOB_ID"

# Make W&B monitor exactly the GPUs exposed to this job
export WANDB__SERVICE__GPU_MONITOR_POLICY=visible
export WANDB__SERVICE__GPU_MONITOR_DEVICES="$CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=1

# Dual GPU settings
export GPUS_PER_NODE=2
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-$((15000 + RANDOM % 20000))}

# Paths
FOLDER_NAME=${SLURM_JOB_ID}
PROJECT_NAME=${PROJECT_NAME:-202604_midiproxy}
DATA_PROJECT=${DATA_PROJECT:-202604_midiproxy_data}
EXECUTABLE=${EXECUTABLE:-$HOME/$PROJECT_NAME/synthesizer/ddsp-piano-pytorch}
SCRATCH=${SCRATCH:-$MYSCRATCH/$PROJECT_NAME/ddsp-piano-pytorch/$FOLDER_NAME}
RESULTS=${RESULTS:-$MYGROUP/${PROJECT_NAME}_results/ddsp-piano-pytorch/$FOLDER_NAME}

# Creates a unique directory in the SCRATCH and GROUP directory 
mkdir -p $SCRATCH $RESULTS
echo SCRATCH is $SCRATCH
echo RESULTS dir is $RESULTS

# Copy code
echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r $EXECUTABLE $SCRATCH
cd $SCRATCH/ddsp-piano-pytorch

# Link dataset (parent exists, target does not → correct ln -s usage)
DATA_SRC=${DATA_SRC:-$MYSCRATCH/$DATA_PROJECT/ddsp-piano-pytorch/workspaces/data_cache}
DATA_VIEW=$SCRATCH/ddsp-piano-pytorch/workspaces/data_cache
ln -s $DATA_SRC $DATA_VIEW

# Link trained models view (to load previous phase checkpoints when needed)
MODELS_SRC=${MODELS_SRC:-$MYSCRATCH/$DATA_PROJECT/ddsp-piano-pytorch/workspaces/models}
MODELS_VIEW=$SCRATCH/ddsp-piano-pytorch/workspaces/models
ln -s $MODELS_SRC $MODELS_VIEW

# Experiment output
EXP_BASE=$SCRATCH/ddsp-piano-pytorch/workspaces/models

# TRAIN_BATCH=6 # Option A: keep previous single-GPU batch (per-GPU batch=6 => global 12)
TRAIN_BATCH=3   # Option B: uncomment to match legacy global batch (per-GPU batch=3 => global 6)
TRAIN_EPOCHS=7
TRAIN_LR=0.001
TRAIN_PHASE=1             # 1/2/3 
NUM_WORKERS=8
DEBUG_MODE=0              # 1 to limit to 20 batches, 0 to disable
WANDB_PROJECT='ddsp-piano'
WANDB_RUN_NAME="phase${TRAIN_PHASE}_dual_gpu-bz${TRAIN_BATCH}-${SLURM_JOB_ID}"

# Init-from controls
# INIT_EPOCH: -1 to use latest from previous phase; otherwise use the specified epoch number
INIT_EPOCH=-1
INIT_FROM=""
if [ "$TRAIN_PHASE" -gt 1 ]; then
  PREV_PHASE=$((TRAIN_PHASE - 1))
  CAND_DIR="$MODELS_VIEW/phase_${PREV_PHASE}/ckpts"
  if [ "$INIT_EPOCH" -eq -1 ]; then
    [ -d "$CAND_DIR" ] && INIT_FROM=$(ls -1 "$CAND_DIR"/ddsp-piano_epoch_*_params.pt 2>/dev/null | sort -V | tail -n 1 || true)
  else
    INIT_FROM="$CAND_DIR/ddsp-piano_epoch_${INIT_EPOCH}_params.pt"
  fi
fi

LAUNCHER="torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS_PER_NODE}"
CMD="$LAUNCHER train.py \
  --use_ddp \
  --batch_size $TRAIN_BATCH \
  --epochs $TRAIN_EPOCHS \
  --lr $TRAIN_LR \
  --phase $TRAIN_PHASE \
  --num_workers $NUM_WORKERS \
  --save_interval 2000 \
  --logs_interval 20 \
  --wandb_project $WANDB_PROJECT \
  --wandb_run_name $WANDB_RUN_NAME \
  $DATA_VIEW \
  $EXP_BASE"

if [ "$DEBUG_MODE" = "1" ]; then
  CMD+=" --debug_mode"
fi
if [ -n "$INIT_FROM" ]; then
  CMD+=" --init_from $INIT_FROM"
fi

echo "Running: $CMD"
eval $CMD

# Collect results & Cleanup
mv $EXP_BASE ${RESULTS}/
cd $HOME
rm -rf $SCRATCH
source deactivate
echo "ddsp job ${SLURM_JOB_ID} finished at $(date)"
