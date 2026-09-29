#!/bin/bash
#SBATCH --job-name=route4_ablation
#SBATCH --output=route4_ablation_progress_%A_%a.log
#SBATCH --error=route4_ablation_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
module list
source activate bark_env
echo "Running on host: $(hostname)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM ID: ${SLURM_ARRAY_ID:-N/A} ${SLURM_ARRAY_TASK_ID:-N/A}"

FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PROJECT_NAME=${PROJECT_NAME:-202604_midiproxy}
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
SFPROXY_ROOT="$DATA_ROOT/synth-proxy"
PRETRAINED_ROOT_SRC="$DATA_ROOT/score_hpt/workspaces/pretrained_checkpoints"
PRETRAINED_ROOT_VIEW="pretrained_checkpoints"

ln -s "$HDF5_SRC" "$HDF5_VIEW"
ln -s "$PRETRAINED_ROOT_SRC" "$PRETRAINED_ROOT_VIEW"

# frontend
TRAIN_SET=${TRAIN_SET:-gaps}
TEST_SET=${TEST_SET:-gaps}
EVAL_SETS=${EVAL_SETS:-[train,francoisleduc,gaps]}
EVAL_SPLIT=${EVAL_SPLIT:-validation} 
MODEL_VARIANTS_STR=${MODEL_VARIANTS:-"hpt_note_editor"}  # filmunet hpt_note_editor
FRONTEND_PRETRAIN_MODES_STR=${FRONTEND_PRETRAIN_MODES:-"scratch"} # scratch route2_piano_auto
FRONTEND_PRETRAINED=${FRONTEND_PRETRAINED:-}
LOSS_TYPE=${LOSS_TYPE:-kim_bce_l1}

# backend
SAMPLERS_STR=${SAMPLERS:-"mixed"}
SEGMENTS_STR=${SEGMENT_LIST:-"2 5"}
PROXY_LOSSES_STR=${LOSS_TYPES:-"smooth_l1 l1"}
SUPERVISED_WEIGHT=${SUPERVISED_WEIGHT:-0.0}
BACKEND_WEIGHT=${BACKEND_WEIGHT:-1.0}
PRIOR_WEIGHT=${PRIOR_WEIGHT:-0.2}
SATURATION_WEIGHT=${SATURATION_WEIGHT:-0.4}
WEIGHT_RECIPES_STR=${WEIGHT_RECIPES:-"0.0,1.0,0.1,0.2 0.0,1.0,0.2,0.4 0.0,1.0,0.4,0.8"} #
BACKEND_WARMUP_ITERATIONS=${BACKEND_WARMUP_ITERATIONS:-5000}
USE_ONSET_WINDOW_POOLING=${USE_ONSET_WINDOW_POOLING:-true}
ONSET_WINDOW_LEFT=${ONSET_WINDOW_LEFT:-1}
ONSET_WINDOW_RIGHT=${ONSET_WINDOW_RIGHT:-2}
ONSET_WINDOW_REDUCTION=${ONSET_WINDOW_REDUCTION:-max}
SFPROXY_FEATURE_WEIGHTS=${SFPROXY_FEATURE_WEIGHTS:-[1.0,0.25]}

# Evaluation
ENABLE_AUDIO_METRICS=${ENABLE_AUDIO_METRICS:-0}
INSTRUMENT_PATH=${INSTRUMENT_PATH:-}
AUDIO_METRIC_MAX_SEGMENTS=${AUDIO_METRIC_MAX_SEGMENTS:-4}

read -r -a MODEL_VARIANTS <<< "$MODEL_VARIANTS_STR"
read -r -a FRONTEND_PRETRAIN_MODES <<< "$FRONTEND_PRETRAIN_MODES_STR"
read -r -a SAMPLERS <<< "$SAMPLERS_STR"
read -r -a SEGMENTS <<< "$SEGMENTS_STR"
read -r -a PROXY_LOSSES <<< "$PROXY_LOSSES_STR"
read -r -a WEIGHT_RECIPES <<< "$WEIGHT_RECIPES_STR"

sampler_dir() {
  local sampler="$1"
  local segment="$2"
  case "$sampler" in
    coverage) echo "piano_salamander_piano_coverage_v2_b0_c1_r0_s0_${segment}s_default" ;;
    mixed) echo "piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_${segment}s_default" ;;
    realism) echo "piano_salamander_piano_realism_v2_b0_c0_r1_s0_${segment}s_default" ;;
    *) echo "piano_salamander_piano_${sampler}_v2_b0_c0_r0_s0_${segment}s_default" ;;
  esac
}

resolve_sfproxy_ckpt() {
  local sampler="$1"
  local segment="$2"
  local run_dir
  local base_dir
  local latest

  run_dir="$(sampler_dir "$sampler" "$segment")"
  base_dir="$SFPROXY_ROOT/proxy/checkpoints/salamander_piano/${run_dir}"
  if [ ! -d "$base_dir" ]; then
    echo "SFProxy checkpoint directory does not exist: $base_dir"
    exit 1
  fi

  latest="$(find "$base_dir" -maxdepth 1 -type f -name '*last*.ckpt' | sort -V | tail -n 1)"
  if [ -n "$latest" ]; then
    printf '%s\n' "$latest"
    return
  fi

  latest="$(find "$base_dir" -maxdepth 1 -type f -name '*loss*.ckpt' | sort -V | tail -n 1)"
  if [ -n "$latest" ]; then
    printf '%s\n' "$latest"
    return
  fi

  latest="$(find "$base_dir" -maxdepth 1 -type f -name '*.ckpt' | sort -V | tail -n 1)"
  if [ -n "$latest" ]; then
    printf '%s\n' "$latest"
  fi
}

EXP_NAME=()
EXP_MODEL_VARIANT=()
EXP_FRONTEND_PRETRAIN_MODE=()
EXP_SAMPLER=()
EXP_SEGMENT=()
EXP_PROXY_LOSS=()
EXP_PROXY_CKPT=()
EXP_SUPERVISED_WEIGHT=()
EXP_BACKEND_WEIGHT=()
EXP_PRIOR_WEIGHT=()
EXP_SATURATION_WEIGHT=()

for SAMPLER in "${SAMPLERS[@]}"; do
  for SEGMENT_SECONDS in "${SEGMENTS[@]}"; do
    PROXY_CKPT="$(resolve_sfproxy_ckpt "$SAMPLER" "$SEGMENT_SECONDS")"
    for PROXY_LOSS in "${PROXY_LOSSES[@]}"; do
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
              EXP_NAME+=("route4_${SAMPLER}_${SEGMENT_SECONDS}s_${PROXY_LOSS}_${MODEL_VARIANT}_${FRONTEND_PRETRAIN_MODE}_sup${sup_tag}_backend${backend_tag}_prior${prior_tag}_sat${sat_tag}")
              EXP_MODEL_VARIANT+=("$MODEL_VARIANT")
              EXP_FRONTEND_PRETRAIN_MODE+=("$FRONTEND_PRETRAIN_MODE")
              EXP_SAMPLER+=("$SAMPLER")
              EXP_SEGMENT+=("$SEGMENT_SECONDS")
              EXP_PROXY_LOSS+=("$PROXY_LOSS")
              EXP_PROXY_CKPT+=("$PROXY_CKPT")
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
            EXP_NAME+=("route4_${SAMPLER}_${SEGMENT_SECONDS}s_${PROXY_LOSS}_${MODEL_VARIANT}_${FRONTEND_PRETRAIN_MODE}_sup${sup_tag}_backend${backend_tag}_prior${prior_tag}_sat${sat_tag}")
            EXP_MODEL_VARIANT+=("$MODEL_VARIANT")
            EXP_FRONTEND_PRETRAIN_MODE+=("$FRONTEND_PRETRAIN_MODE")
            EXP_SAMPLER+=("$SAMPLER")
            EXP_SEGMENT+=("$SEGMENT_SECONDS")
            EXP_PROXY_LOSS+=("$PROXY_LOSS")
            EXP_PROXY_CKPT+=("$PROXY_CKPT")
            EXP_SUPERVISED_WEIGHT+=("$SUPERVISED_WEIGHT")
            EXP_BACKEND_WEIGHT+=("$BACKEND_WEIGHT")
            EXP_PRIOR_WEIGHT+=("$PRIOR_WEIGHT")
            EXP_SATURATION_WEIGHT+=("$SATURATION_WEIGHT")
          fi
        done
      done
    done
  done
done

TOTAL_JOBS=${#EXP_NAME[@]}

IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM array index $IDX exceeds configured job count $TOTAL_JOBS"
  exit 1
fi

EXP_TAG=${EXP_NAME[$IDX]}
MODEL_VARIANT=${EXP_MODEL_VARIANT[$IDX]}
FRONTEND_PRETRAIN_MODE=${EXP_FRONTEND_PRETRAIN_MODE[$IDX]}
SAMPLER=${EXP_SAMPLER[$IDX]}
SEGMENT_SECONDS=${EXP_SEGMENT[$IDX]}
PROXY_LOSS=${EXP_PROXY_LOSS[$IDX]}
PROXY_CKPT=${EXP_PROXY_CKPT[$IDX]}
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
if [ "$FRONTEND_PRETRAIN_MODE" = "route2_piano_specific" ]; then
  if [ -z "$FRONTEND_PRETRAINED" ]; then
    echo "FRONTEND_PRETRAINED must be set when FRONTEND_PRETRAIN_MODE=route2_piano_specific"
    exit 1
  fi
  case "$FRONTEND_PRETRAINED" in
    /*|pretrained_checkpoints/*) FRONTEND_PRETRAINED_ARG="$FRONTEND_PRETRAINED" ;;
    *) FRONTEND_PRETRAINED_ARG="$PRETRAINED_ROOT_VIEW/$FRONTEND_PRETRAINED" ;;
  esac
fi

if [ ! -f "$PROXY_CKPT" ]; then
  echo "Resolved DiffProxy checkpoint does not exist: $PROXY_CKPT"
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
echo "Sampler          : $SAMPLER"
echo "Backend seg (s)  : $SEGMENT_SECONDS"
echo "Backend loss     : $PROXY_LOSS"
echo "DiffProxy ckpt   : $PROXY_CKPT"
echo "Warmup iters     : $BACKEND_WARMUP_ITERATIONS"
echo "Onset pooling    : $USE_ONSET_WINDOW_POOLING (${ONSET_WINDOW_LEFT},${ONSET_WINDOW_RIGHT},${ONSET_WINDOW_REDUCTION})"
echo "Feature weights  : $SFPROXY_FEATURE_WEIGHTS"

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

python pytorch/train_proxy.py \
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
  backend.type=diffproxy \
  backend.project_root=../synth-proxy \
  backend.checkpoint="$PROXY_CKPT" \
  backend.backend_segment_seconds="$SEGMENT_SECONDS" \
  backend.warmup_iterations="$BACKEND_WARMUP_ITERATIONS" \
  backend.supervision.hop_size=221 \
  backend.diffproxy.instrument_name=salamander_piano \
  backend.diffproxy.loss_type="$PROXY_LOSS" \
  backend.diffproxy.use_gt_aligned_note_events=true \
  backend.diffproxy.use_onset_window_pooling="$USE_ONSET_WINDOW_POOLING" \
  backend.diffproxy.onset_window_left="$ONSET_WINDOW_LEFT" \
  backend.diffproxy.onset_window_right="$ONSET_WINDOW_RIGHT" \
  backend.diffproxy.onset_window_reduction="$ONSET_WINDOW_REDUCTION" \
  "backend.diffproxy.feature_weights=$SFPROXY_FEATURE_WEIGHTS" \
  backend.diffproxy.feature.hop=221 \
  wandb.comment="$EXP_TAG" \
  "${EXTRA_ARGS[@]}"

[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"

cd "$HOME"
rm -r "$SCRATCH"
conda deactivate 2>/dev/null || source deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo route4_ablation $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
