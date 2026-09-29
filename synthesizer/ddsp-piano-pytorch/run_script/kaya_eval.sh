#!/bin/bash
#SBATCH --job-name=ddsp-eval
#SBATCH --output=ddsp_eval_progress_%j.log
#SBATCH --error=ddsp_eval_error_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

# Modules + env
module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
module list
source activate bark_env

echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# W&B GPU visibility
export WANDB__SERVICE__GPU_MONITOR_POLICY=visible
export WANDB__SERVICE__GPU_MONITOR_DEVICES="$CUDA_VISIBLE_DEVICES"

# Paths
FOLDER_NAME=${SLURM_JOB_ID}
PROJECT_NAME=${PROJECT_NAME:-202604_midiproxy}
DATA_PROJECT=${DATA_PROJECT:-202604_midiproxy_data}
EXECUTABLE=${EXECUTABLE:-$HOME/$PROJECT_NAME/synthesizer/ddsp-piano-pytorch}
SCRATCH=${SCRATCH:-$MYSCRATCH/$PROJECT_NAME/ddsp-piano-pytorch/$FOLDER_NAME}
RESULTS=${RESULTS:-$MYGROUP/${PROJECT_NAME}_results/ddsp-piano-pytorch/$FOLDER_NAME}

mkdir -p $SCRATCH $RESULTS
echo SCRATCH is $SCRATCH
echo RESULTS dir is $RESULTS

# Copy code to scratch
cp -r $EXECUTABLE $SCRATCH
cd $SCRATCH/ddsp-piano-pytorch

# Link dataset cache into scratch workspace
DATA_SRC=${DATA_SRC:-$MYSCRATCH/$DATA_PROJECT/ddsp-piano-pytorch/workspaces/data_cache}
DATA_VIEW=$SCRATCH/ddsp-piano-pytorch/workspaces/data_cache
ln -s $DATA_SRC $DATA_VIEW

# Link trained models into scratch workspace
MODELS_SRC=${MODELS_SRC:-$MYSCRATCH/$DATA_PROJECT/ddsp-piano-pytorch/workspaces/models}
MODELS_VIEW=$SCRATCH/ddsp-piano-pytorch/workspaces/models
ln -s $MODELS_SRC $MODELS_VIEW

# Eval configuration
PHASE=1                   # 1/2/3
SPLIT="validation"        # validation/test
ORDER="epoch"             # epoch/step
BATCH_SIZE=4
NUM_WORKERS=14
USE_CUDA=1
DEBUG_MODE=0              # 1 to limit to 20 batches, 0 to disable
WANDB_PROJECT='ddsp-piano'
WANDB_RUN_NAME="eval_phase${PHASE}_${SPLIT}_in_${ORDER}_${SLURM_JOB_ID}"

# Checkpoints folder to scan and output dir for metrics
CHECKPOINT="$MODELS_VIEW/phase_${PHASE}/ckpts"
OUTPUT_DIR="$SCRATCH/ddsp-piano-pytorch/workspaces/models/eval_phase_${PHASE}_${SPLIT}"

CMD="python evaluate.py \
  --checkpoint $CHECKPOINT \
  --output_dir $OUTPUT_DIR \
  --phase $PHASE \
  --split $SPLIT \
  --ckpt_order $ORDER \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --cuda $USE_CUDA \
  --wandb_project $WANDB_PROJECT --wandb_run_name $WANDB_RUN_NAME \
  $DATA_VIEW"

if [ "$DEBUG_MODE" = "1" ]; then
  CMD+=" --debug_mode"
fi

echo "Running: $CMD"
eval $CMD

# Collect results & Cleanup
mkdir -p "$RESULTS/eval"
mv "$OUTPUT_DIR" "$RESULTS/eval/"
cd "$HOME"
rm -rf "$SCRATCH"
source deactivate
echo "ddsp eval job ${SLURM_JOB_ID} finished at $(date)"
