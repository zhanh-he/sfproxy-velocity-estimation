#!/bin/bash
#SBATCH --job-name=route3_ablation
#SBATCH --output=route3_ablation_progress_%A_%a.log
#SBATCH --error=route3_ablation_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
module list
source activate bark_env


echo "Running on host: $(hostname)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM ID: ${SLURM_ARRAY_ID:-N/A} ${SLURM_ARRAY_TASK_ID:-N/A}"

FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PROJECT_NAME=${PROJECT_NAME:-sfproxy-velocity-estimation}
DATA_PROJECT=${DATA_PROJECT:-202604_midiproxy_data}
EXECUTABLE=$HOME/${PROJECT_NAME}
SCRATCH=$MYSCRATCH/${PROJECT_NAME}/$FOLDER_NAME
RESULTS=$MYGROUP/${PROJECT_NAME}_results/$FOLDER_NAME

mkdir -p "$SCRATCH" "$RESULTS"
echo "SCRATCH is $SCRATCH"
echo "RESULTS dir is $RESULTS"

echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r "$EXECUTABLE" "$SCRATCH"
cd "$SCRATCH/$PROJECT_NAME/score_hpt"

WORKSPACE_DIR=workspaces
rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

DATA_ROOT="$MYSCRATCH/${DATA_PROJECT}"
HDF5_SRC="$DATA_ROOT/score_hpt/workspaces/hdf5s"
HDF5_VIEW="$WORKSPACE_DIR/hdf5s"
DDSP_ROOT="$DATA_ROOT/ddsp-piano-pytorch"
PRETRAINED_ROOT_SRC="$DATA_ROOT/score_hpt/workspaces/pretrained_checkpoints"
PRETRAINED_ROOT_VIEW="pretrained_checkpoints"

ln -s "$HDF5_SRC" "$HDF5_VIEW"
ln -s "$PRETRAINED_ROOT_SRC" "$PRETRAINED_ROOT_VIEW"

# frontend
TRAIN_SET=${TRAIN_SET:-maestro}
TEST_SET=${TEST_SET:-maestro}
EVAL_SETS=${EVAL_SETS:-[train,maestro,smd]}
EVAL_SPLIT=${EVAL_SPLIT:-validation}
MODEL_VARIANTS_STR=${MODEL_VARIANTS:-"hpt_note_editor"} # "hpt_note_editor filmunet"
FRONTEND_PRETRAIN_MODES_STR=${FRONTEND_PRETRAIN_MODES:-"scratch"} # "scratch route2_piano_auto"
FRONTEND_PRETRAINED=${FRONTEND_PRETRAINED:-}
LOSS_TYPE=${LOSS_TYPE:-kim_bce_l1}

# backend
SEGMENTS_STR=${SEGMENT_LIST:-"2 5"}
AUDIO_LOSSES_STR=${AUDIO_LOSSES:-"piano_ssm_spectral piano_ssm_spectral_plus_log_rms"}
SUPERVISED_WEIGHT=${SUPERVISED_WEIGHT:-0.0}
BACKEND_WEIGHT=${BACKEND_WEIGHT:-1.0}
PRIOR_WEIGHT=${PRIOR_WEIGHT:-0.2}
SATURATION_WEIGHT=${SATURATION_WEIGHT:-0.4}
WEIGHT_RECIPES_STR=${WEIGHT_RECIPES:-"1.0,1.0,0.1,0.2 0.5,0.5,0.1,0.2"}
DDSP_PHASE=${DDSP_PHASE:-1}
DDSP_CKPT_EPOCH=${DDSP_CKPT_EPOCH:-7}
LOGRMS_WEIGHT=${LOGRMS_WEIGHT:-0.05}
DIFFSYNTH_LOUDNESS_WEIGHT=${DIFFSYNTH_LOUDNESS_WEIGHT:-0.05}

# Evaluation
ENABLE_AUDIO_METRICS=${ENABLE_AUDIO_METRICS:-0}
INSTRUMENT_PATH=${INSTRUMENT_PATH:-}
AUDIO_METRIC_MAX_SEGMENTS=${AUDIO_METRIC_MAX_SEGMENTS:-4}

read -r -a MODEL_VARIANTS <<< "$MODEL_VARIANTS_STR"
read -r -a FRONTEND_PRETRAIN_MODES <<< "$FRONTEND_PRETRAIN_MODES_STR"
read -r -a SEGMENTS <<< "$SEGMENTS_STR"
read -r -a AUDIO_LOSSES <<< "$AUDIO_LOSSES_STR"
read -r -a WEIGHT_RECIPES <<< "$WEIGHT_RECIPES_STR"

resolve_ddsp_ckpt() {
  local segment_seconds="$1"
  local ckpt_dir="$DDSP_ROOT/workspaces_unified_${segment_seconds}s/models/phase_${DDSP_PHASE}/ckpts"
  local ckpt

  if [ "$DDSP_CKPT_EPOCH" = "latest" ]; then
    ckpt=$(find "$ckpt_dir" -maxdepth 1 -type f -name 'ddsp-piano_epoch_*_params.pt' | sort -V | tail -n 1)
  else
    ckpt="$ckpt_dir/ddsp-piano_epoch_${DDSP_CKPT_EPOCH}_params.pt"
  fi

  printf '%s\n' "$ckpt"
}

EXP_NAME=()
EXP_MODEL_VARIANT=()
EXP_FRONTEND_PRETRAIN_MODE=()
EXP_SEGMENT=()
EXP_AUDIO_LOSS=()
EXP_DDSP_CKPT=()
EXP_SUPERVISED_WEIGHT=()
EXP_BACKEND_WEIGHT=()
EXP_PRIOR_WEIGHT=()
EXP_SATURATION_WEIGHT=()

for SEGMENT_SECONDS in "${SEGMENTS[@]}"; do
  DDSP_CKPT="$(resolve_ddsp_ckpt "$SEGMENT_SECONDS")"
  for AUDIO_LOSS in "${AUDIO_LOSSES[@]}"; do
    for MODEL_VARIANT in "${MODEL_VARIANTS[@]}"; do
      for FRONTEND_PRETRAIN_MODE in "${FRONTEND_PRETRAIN_MODES[@]}"; do
        if [ -n "$WEIGHT_RECIPES_STR" ]; then
          for WEIGHT_RECIPE in "${WEIGHT_RECIPES[@]}"; do
            IFS=, read -r RECIPE_SUPERVISED_WEIGHT RECIPE_BACKEND_WEIGHT RECIPE_PRIOR_WEIGHT RECIPE_SATURATION_WEIGHT <<< "$WEIGHT_RECIPE"
            RECIPE_PRIOR_WEIGHT="${RECIPE_PRIOR_WEIGHT:-0.0}"
            RECIPE_SATURATION_WEIGHT="${RECIPE_SATURATION_WEIGHT:-0.0}"
            sup_tag=${RECIPE_SUPERVISED_WEIGHT/./p}
            backend_tag=${RECIPE_BACKEND_WEIGHT/./p}
            prior_tag=${RECIPE_PRIOR_WEIGHT/./p}
            sat_tag=${RECIPE_SATURATION_WEIGHT/./p}
            EXP_NAME+=("route3_${MODEL_VARIANT}_${FRONTEND_PRETRAIN_MODE}_${SEGMENT_SECONDS}s_${AUDIO_LOSS}_sup${sup_tag}_backend${backend_tag}_prior${prior_tag}_sat${sat_tag}")
            EXP_MODEL_VARIANT+=("$MODEL_VARIANT")
            EXP_FRONTEND_PRETRAIN_MODE+=("$FRONTEND_PRETRAIN_MODE")
            EXP_SEGMENT+=("$SEGMENT_SECONDS")
            EXP_AUDIO_LOSS+=("$AUDIO_LOSS")
            EXP_DDSP_CKPT+=("$DDSP_CKPT")
            EXP_SUPERVISED_WEIGHT+=("$RECIPE_SUPERVISED_WEIGHT")
            EXP_BACKEND_WEIGHT+=("$RECIPE_BACKEND_WEIGHT")
            EXP_PRIOR_WEIGHT+=("$RECIPE_PRIOR_WEIGHT")
            EXP_SATURATION_WEIGHT+=("$RECIPE_SATURATION_WEIGHT")
          done
        else
          sup_tag=${SUPERVISED_WEIGHT/./p}
          backend_tag=${BACKEND_WEIGHT/./p}
          prior_tag=${PRIOR_WEIGHT/./p}
          sat_tag=${SATURATION_WEIGHT/./p}
          EXP_NAME+=("route3_${MODEL_VARIANT}_${FRONTEND_PRETRAIN_MODE}_${SEGMENT_SECONDS}s_${AUDIO_LOSS}_sup${sup_tag}_backend${backend_tag}_prior${prior_tag}_sat${sat_tag}")
          EXP_MODEL_VARIANT+=("$MODEL_VARIANT")
          EXP_FRONTEND_PRETRAIN_MODE+=("$FRONTEND_PRETRAIN_MODE")
          EXP_SEGMENT+=("$SEGMENT_SECONDS")
          EXP_AUDIO_LOSS+=("$AUDIO_LOSS")
          EXP_DDSP_CKPT+=("$DDSP_CKPT")
          EXP_SUPERVISED_WEIGHT+=("$SUPERVISED_WEIGHT")
          EXP_BACKEND_WEIGHT+=("$BACKEND_WEIGHT")
          EXP_PRIOR_WEIGHT+=("$PRIOR_WEIGHT")
          EXP_SATURATION_WEIGHT+=("$SATURATION_WEIGHT")
        fi
      done
    done
  done
done

TOTAL_JOBS=${#EXP_NAME[@]}

IDX=$SLURM_ARRAY_TASK_ID
EXP_TAG=${EXP_NAME[$IDX]}
MODEL_VARIANT=${EXP_MODEL_VARIANT[$IDX]}
FRONTEND_PRETRAIN_MODE=${EXP_FRONTEND_PRETRAIN_MODE[$IDX]}
SEGMENT_SECONDS=${EXP_SEGMENT[$IDX]}
AUDIO_LOSS=${EXP_AUDIO_LOSS[$IDX]}
DDSP_CKPT=${EXP_DDSP_CKPT[$IDX]}
SUPERVISED_WEIGHT=${EXP_SUPERVISED_WEIGHT[$IDX]}
BACKEND_WEIGHT=${EXP_BACKEND_WEIGHT[$IDX]}
PRIOR_WEIGHT=${EXP_PRIOR_WEIGHT[$IDX]}
SATURATION_WEIGHT=${EXP_SATURATION_WEIGHT[$IDX]}

MODEL_TYPE="filmunet"
RUN_SCORE_METHOD="direct"
MODEL_INPUT2="null"
if [ "$MODEL_VARIANT" = "hpt_note_editor" ]; then
  MODEL_TYPE="hpt"
  RUN_SCORE_METHOD="note_editor"
  MODEL_INPUT2="onset"
fi

FRONTEND_PRETRAINED_ARG=""
if [ "$FRONTEND_PRETRAIN_MODE" = "route2_piano_specific" ] && [ -n "$FRONTEND_PRETRAINED" ]; then
  case "$FRONTEND_PRETRAINED" in
    /*|pretrained_checkpoints/*) FRONTEND_PRETRAINED_ARG="$FRONTEND_PRETRAINED" ;;
    *) FRONTEND_PRETRAINED_ARG="$PRETRAINED_ROOT_VIEW/$FRONTEND_PRETRAINED" ;;
  esac
fi

if [ ! -f "$DDSP_CKPT" ]; then
  echo "Resolved DDSP checkpoint does not exist: $DDSP_CKPT"
  exit 1
fi
if [ -n "$FRONTEND_PRETRAINED_ARG" ] && [ ! -f "$FRONTEND_PRETRAINED_ARG" ]; then
  echo "Resolved frontend checkpoint does not exist: $FRONTEND_PRETRAINED_ARG"
  exit 1
fi

echo "Experiment       : $EXP_TAG"
echo "Model variant    : $MODEL_VARIANT"
echo "Score method     : $RUN_SCORE_METHOD"
echo "Frontend mode    : $FRONTEND_PRETRAIN_MODE"
echo "Input2 / Input3  : $MODEL_INPUT2 / null"
echo "Velocity loss    : $LOSS_TYPE"
echo "Supervised weight: $SUPERVISED_WEIGHT"
echo "Backend weight   : $BACKEND_WEIGHT"
echo "Prior weight     : $PRIOR_WEIGHT"
echo "Saturation weight: $SATURATION_WEIGHT"
echo "Backend seg (s)  : $SEGMENT_SECONDS"
echo "Backend objective: $AUDIO_LOSS"
echo "DDSP checkpoint  : $DDSP_CKPT"

EXTRA_ARGS=()
if [ "$ENABLE_AUDIO_METRICS" = "1" ] || [ "$ENABLE_AUDIO_METRICS" = "true" ]; then
  EXTRA_ARGS+=(
    "train_eval.audio_metrics.enabled=true"
    "train_eval.audio_metrics.instrument_path=$INSTRUMENT_PATH"
    "train_eval.audio_metrics.max_segments=$AUDIO_METRIC_MAX_SEGMENTS"
  )
fi
if [ -n "$FRONTEND_PRETRAINED_ARG" ]; then
  EXTRA_ARGS+=("model.frontend_pretrained_mode=route2_piano_specific")
  EXTRA_ARGS+=("model.frontend_pretrained=$FRONTEND_PRETRAINED_ARG")
fi
if [ "$FRONTEND_PRETRAIN_MODE" = "route2_piano_auto" ]; then
  EXTRA_ARGS+=("model.frontend_pretrained_mode=route2_piano_auto")
fi
if [ "$FRONTEND_PRETRAIN_MODE" = "scratch" ]; then
  EXTRA_ARGS+=("model.frontend_pretrained_mode=scratch")
fi

python pytorch/train_ddsp.py \
  exp.workspace="$WORKSPACE_DIR" \
  dataset.train_set="$TRAIN_SET" \
  dataset.test_set="$TEST_SET" \
  "dataset.eval_sets=$EVAL_SETS" \
  dataset.eval_split="$EVAL_SPLIT" \
  model.type="$MODEL_TYPE" \
  model.input2="$MODEL_INPUT2" \
  model.input3=null \
  score_informed.method="$RUN_SCORE_METHOD" \
  loss.loss_type="$LOSS_TYPE" \
  loss.supervised_weight="$SUPERVISED_WEIGHT" \
  loss.backend_weight="$BACKEND_WEIGHT" \
  loss.velocity_prior_weight="$PRIOR_WEIGHT" \
  loss.velocity_saturation_weight="$SATURATION_WEIGHT" \
  backend.enabled=true \
  backend.type=diffsynth_piano \
  backend.project_root=../diff-synth/ddsp-piano-pytorch \
  backend.checkpoint="$DDSP_CKPT" \
  backend.backend_segment_seconds="$SEGMENT_SECONDS" \
  backend.audio_loss.type="$AUDIO_LOSS" \
  backend.audio_loss.piano_ssm_spectral_plus_log_rms.log_rms_weight="$LOGRMS_WEIGHT" \
  backend.audio_loss.piano_ssm_spectral_plus_diffsynth_loudness.loudness_weight="$DIFFSYNTH_LOUDNESS_WEIGHT" \
  wandb.comment="$EXP_TAG" \
  "${EXTRA_ARGS[@]}"

[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"

cd "$HOME"
rm -r "$SCRATCH"
conda deactivate 2>/dev/null || source deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo route3_ablation $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
